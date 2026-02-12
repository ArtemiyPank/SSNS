// ssns-benchmark — CLI subprocess spawned by /api/run_training.  Mirrors
// the contract of /home/artemiypank/Programming/Python/SSNS_mvp/scripts/
// benchmark_fhe_vs_plain.py: same flags, same output paths, same status
// + log JSON schemas.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <ssns/io/logger.hpp>
#include <ssns/io/status_file.hpp>
#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/activations.hpp>
#include <ssns/nn/client.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/lr_schedule.hpp>
#include <ssns/nn/server.hpp>
#include <ssns/protocol/training.hpp>

namespace {

struct Args {
    std::size_t t_input          = 64;
    std::size_t t_hidden         = 64;
    std::size_t s_input          = 64;
    std::size_t s_hidden         = 192;
    std::size_t output_dim       = 0;     // required
    std::size_t cluster_size     = 5;
    std::size_t batch_size       = 64;
    long        epochs           = 0;     // required
    double      dz               = 0.03;
    double      lr_max           = 0.01;
    double      warmup_frac      = 0.05;
    long        snapshot_interval = 0;    // 0 -> auto-derive epochs/10
    long        samples_to_log    = 20;
    std::filesystem::path output_path  = "ui_data/training_log.json";
    std::filesystem::path status_path  = "ui_data/training_status.json";
    std::uint64_t seed             = 2024;
    std::uint64_t teacher_seed     = 42;
    std::uint64_t bfa_seed         = 43;
    bool          skip_fhe_bench   = false;   // accepted, currently no-op
    bool          use_fhe          = false;   // Phase 7
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "ssns-benchmark: " << msg << "\n";
    std::exit(2);
}

bool match(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

template <typename T>
T parse_int(const char* s) {
    char* end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (!end || *end != '\0') die(std::string("not an integer: ") + s);
    return static_cast<T>(v);
}

double parse_double(const char* s) {
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (!end || *end != '\0') die(std::string("not a number: ") + s);
    return v;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char* k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) die(std::string("missing value for ") + k);
            return argv[++i];
        };
        if      (match(k, "--t-input"))           a.t_input  = parse_int<std::size_t>(next());
        else if (match(k, "--t-hidden"))          a.t_hidden = parse_int<std::size_t>(next());
        else if (match(k, "--s-input"))           a.s_input  = parse_int<std::size_t>(next());
        else if (match(k, "--s-hidden"))          a.s_hidden = parse_int<std::size_t>(next());
        else if (match(k, "--output-dim"))        a.output_dim = parse_int<std::size_t>(next());
        else if (match(k, "--cluster-size"))      a.cluster_size = parse_int<std::size_t>(next());
        else if (match(k, "--batch-size"))        a.batch_size = parse_int<std::size_t>(next());
        else if (match(k, "--epochs"))            a.epochs = parse_int<long>(next());
        else if (match(k, "--dz"))                a.dz = parse_double(next());
        else if (match(k, "--lr-max"))            a.lr_max = parse_double(next());
        else if (match(k, "--warmup-frac"))       a.warmup_frac = parse_double(next());
        else if (match(k, "--snapshot-interval")) a.snapshot_interval = parse_int<long>(next());
        else if (match(k, "--samples-to-log"))    a.samples_to_log = parse_int<long>(next());
        else if (match(k, "--output-path"))       a.output_path = next();
        else if (match(k, "--status-path"))       a.status_path = next();
        else if (match(k, "--seed"))              a.seed = parse_int<std::uint64_t>(next());
        else if (match(k, "--teacher-seed"))      a.teacher_seed = parse_int<std::uint64_t>(next());
        else if (match(k, "--bfa-seed"))          a.bfa_seed = parse_int<std::uint64_t>(next());
        else if (match(k, "--skip-fhe-bench"))    a.skip_fhe_bench = true;
        else if (match(k, "--use-fhe"))           a.use_fhe = true;
        else die(std::string("unknown flag: ") + k);
    }
    if (a.epochs <= 0)     die("--epochs is required and must be > 0");
    if (a.output_dim == 0) die("--output-dim is required and must be > 0");
    if (a.s_input != a.t_input) die("--s-input must equal --t-input");
    if (a.output_dim % a.cluster_size != 0)
        die("--output-dim must be divisible by --cluster-size");
    if (a.snapshot_interval <= 0) {
        a.snapshot_interval = std::max<long>(1, a.epochs / 10);
    }
    if (a.use_fhe) {
        // Phase 7 will hook this up; until then refuse loudly.
        die("--use-fhe is not implemented in this phase");
    }
    return a;
}

// Compute Teacher's hidden activation for visualisation only.
ssns::linalg::Matrix teacher_hidden(
    const ssns::nn::CleanServer& server,
    const ssns::linalg::Matrix& X)
{
    auto H_pre = ssns::linalg::matmul(X, server.teacher().W1());
    return ssns::nn::relu(H_pre);
}

}  // namespace

int main(int argc, char** argv) try {
    auto a = parse_args(argc, argv);

    using namespace ssns;

    nn::CleanClientConfig client_cfg{};
    client_cfg.input_dim          = a.s_input;
    client_cfg.hidden_dim         = a.s_hidden;
    client_cfg.output_dim         = a.output_dim;
    client_cfg.lr_max             = a.lr_max;
    client_cfg.lr_total_steps     = a.epochs;
    client_cfg.lr_warmup_frac     = a.warmup_frac;
    client_cfg.seed               = a.seed;
    client_cfg.grad_clip_max_norm = 1.0;

    nn::CleanClient client(client_cfg);
    nn::CleanServer server(a.t_input, a.s_hidden, a.t_hidden, a.output_dim,
                            a.teacher_seed, a.bfa_seed);

    io::LoggerConfig log_cfg{};
    log_cfg.snapshot_interval = a.snapshot_interval;
    log_cfg.samples_to_log    = a.samples_to_log;
    log_cfg.T_input    = a.t_input;
    log_cfg.T_hidden   = a.t_hidden;
    log_cfg.S_input    = a.s_input;
    log_cfg.S_hidden   = a.s_hidden;
    log_cfg.output_dim = a.output_dim;
    log_cfg.cluster_size = a.cluster_size;
    log_cfg.batch_size = a.batch_size;
    log_cfg.epochs     = a.epochs;
    log_cfg.dz         = a.dz;
    log_cfg.lr_max     = a.lr_max;
    log_cfg.warmup_frac = a.warmup_frac;
    log_cfg.teacher_seed = a.teacher_seed;
    log_cfg.bfa_seed     = a.bfa_seed;
    io::TrainingLogger logger(log_cfg);

    nn::Rng rng(a.seed + 1);

    const auto started_at = std::chrono::system_clock::now();
    io::write_starting_status(a.status_path, a.epochs);

    const long status_interval = std::max<long>(1, a.epochs / 100);
    double last_loss = 0.0;
    double last_lr   = 0.0;

    for (long ep = 1; ep <= a.epochs; ++ep) {
        auto step = protocol::clean_train_step(client, server,
                                               a.batch_size, a.s_input, rng);
        const double lr_now = nn::warmup_cosine_lr(
            client.step_count() - 1,
            a.epochs,
            a.warmup_frac,
            a.lr_max,
            0.0);
        last_loss = step.loss;
        last_lr   = lr_now;

        // Compute Teacher hidden + per-batch error for the logger sample slice.
        auto H_T = teacher_hidden(server, step.X);
        // error = (Y_pred - Y_true) / batch  (matches the Python schema)
        auto err = linalg::sub(step.Y_pred, step.Y_true);
        err.scale_in_place(1.0 / static_cast<double>(a.batch_size));

        logger.maybe_record(
            ep, step.loss, lr_now,
            server.teacher().W1(), server.teacher().W2(),
            client.W1(), client.W2(),
            server.b_fa(),
            step.X, H_T, step.H,
            step.Y_true, step.Y_pred, err);

        if (ep % status_interval == 0 || ep == a.epochs) {
            io::write_progress(a.status_path, ep, a.epochs,
                               last_loss, last_lr,
                               started_at, /*running=*/true,
                               /*completed_at=*/std::nullopt);
        }
    }

    // IMPORTANT: log first, status flip second — matches the Python reference's
    // ordering so any frontend that polls running:false is guaranteed to read
    // the freshly-written training_log.json on the very next request.
    logger.save(a.output_path);
    io::write_progress(a.status_path, a.epochs, a.epochs,
                       last_loss, last_lr,
                       started_at, /*running=*/false,
                       /*completed_at=*/std::chrono::system_clock::now());

    return 0;
} catch (const std::exception& e) {
    std::cerr << "ssns-benchmark: " << e.what() << "\n";
    return 2;
}
