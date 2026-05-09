// ssns-benchmark cli subprocess spawned by /api/run_training
// same flags output paths status/log json schemas as python ref at SSNS_mvp/scripts/benchmark_fhe_vs_plain.py
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <ssns/ckks/backend.hpp>
#include <ssns/io/logger.hpp>
#include <ssns/io/status_file.hpp>
#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/activations.hpp>
#include <ssns/nn/client.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/lr_schedule.hpp>
#include <ssns/nn/server.hpp>
#include <ssns/protocol/bit_extract.hpp>
#include <ssns/protocol/training.hpp>

namespace {

// all cli args defaults match python ref
struct Args {
    // net shape, required output_dim epochs
    std::size_t t_input          = 64;
    std::size_t t_hidden         = 64;
    std::size_t s_input          = 64;
    std::size_t s_hidden         = 192;
    std::size_t output_dim       = 0;     // required
    std::size_t cluster_size     = 5;
    std::size_t batch_size       = 64;
    long        epochs           = 0;     // required

    // train sched + dz
    double      dz               = 0.03;
    double      lr_max           = 0.01;
    double      warmup_frac      = 0.05;

    // logging cadence + out paths
    long        snapshot_interval = 0;    // 0 means auto epochs/10
    long        samples_to_log    = 20;
    std::filesystem::path output_path  = "ui_data/training_log.json";
    std::filesystem::path status_path  = "ui_data/training_status.json";

    // rng seeds
    std::uint64_t seed             = 2024;
    std::uint64_t teacher_seed     = 42;
    std::uint64_t bfa_seed         = 43;

    // mode + teacher tuning
    bool          use_fhe          = false;
    double        teacher_w2_scale = 1.0;
    double        bimodality_alpha = 0.0;
    double        simulate_fhe_noise = 0.0;   // 0 disables, plain + noise = fast fhe proxy

    // inline stress test after training
    long          stress_trials      = 0;     // 0 disables N>0 runs N rand trials
    long          stress_seed        = 7;     // rng seed for stress
    long          stress_runs        = 1;     // run stress N times with seeds stress_seed +1 +2 ...

    // inline key confirmation sim
    long          confirm_trials     = 0;     // 0 disables N>0 runs N rounds of T/S hash compare
    long          confirm_seed       = 11;    // rng seed for X_key sampling
    long          confirm_min_bits   = 2;     // need at least this many shared bits else no-key retry

    // pre filters that can reject seed before full training
    double        prefilter_min_hit_rate  = 0.0;   // 0 disables reject if teacher bimodality too low
    double        prefilter_dz            = 0.20;  // dz for bimodality estimate
    std::size_t   prefilter_batch         = 2000;  // sample count for bimodality estimate
    long          prefilter_plain_epochs  = 0;     // 0 disables N plain epochs as cheap fhe proxy
    double        prefilter_plain_max_loss = 1.0;  // reject if loss above this after N plain epochs
};

// print err and exit 2
[[noreturn]] void die(const std::string& msg) {
    std::cerr << "ssns-benchmark: " << msg << "\n";
    std::exit(2);
}

// strcmp wrapper true when both c strings equal
bool match(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

// parse base 10 int or die
template <typename T>
T parse_int(const char* s) {
    char* end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (!end || *end != '\0') die(std::string("not an integer: ") + s);
    return static_cast<T>(v);
}

// parse double or die
double parse_double(const char* s) {
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (!end || *end != '\0') die(std::string("not a number: ") + s);
    return v;
}

// walk argv fill Args, unknown flag -> die
Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char* k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) die(std::string("missing value for ") + k);
            return argv[++i];
        };
        // net shape flags
        if      (match(k, "--t-input"))           a.t_input  = parse_int<std::size_t>(next());
        else if (match(k, "--t-hidden"))          a.t_hidden = parse_int<std::size_t>(next());
        else if (match(k, "--s-input"))           a.s_input  = parse_int<std::size_t>(next());
        else if (match(k, "--s-hidden"))          a.s_hidden = parse_int<std::size_t>(next());
        else if (match(k, "--output-dim"))        a.output_dim = parse_int<std::size_t>(next());
        else if (match(k, "--cluster-size"))      a.cluster_size = parse_int<std::size_t>(next());
        else if (match(k, "--batch-size"))        a.batch_size = parse_int<std::size_t>(next());
        else if (match(k, "--epochs"))            a.epochs = parse_int<long>(next());
        // train sched + dz
        else if (match(k, "--dz"))                a.dz = parse_double(next());
        else if (match(k, "--lr-max"))            a.lr_max = parse_double(next());
        else if (match(k, "--warmup-frac"))       a.warmup_frac = parse_double(next());
        // logging + output paths
        else if (match(k, "--snapshot-interval")) a.snapshot_interval = parse_int<long>(next());
        else if (match(k, "--samples-to-log"))    a.samples_to_log = parse_int<long>(next());
        else if (match(k, "--output-path"))       a.output_path = next();
        else if (match(k, "--status-path"))       a.status_path = next();
        // seeds
        else if (match(k, "--seed"))              a.seed = parse_int<std::uint64_t>(next());
        else if (match(k, "--teacher-seed"))      a.teacher_seed = parse_int<std::uint64_t>(next());
        else if (match(k, "--bfa-seed"))          a.bfa_seed = parse_int<std::uint64_t>(next());
        // mode + teacher tuning
        else if (match(k, "--use-fhe"))             a.use_fhe = true;
        else if (match(k, "--teacher-w2-scale"))    a.teacher_w2_scale = parse_double(next());
        else if (match(k, "--bimodality-alpha"))        a.bimodality_alpha = parse_double(next());
        else if (match(k, "--simulate-fhe-noise"))      a.simulate_fhe_noise = parse_double(next());
        // inline stress test
        else if (match(k, "--stress-trials"))           a.stress_trials = parse_int<long>(next());
        else if (match(k, "--stress-seed"))             a.stress_seed = parse_int<long>(next());
        else if (match(k, "--stress-runs"))             a.stress_runs = parse_int<long>(next());
        // inline key confirm
        else if (match(k, "--key-confirmation"))        a.confirm_trials = parse_int<long>(next());
        else if (match(k, "--key-confirmation-seed"))   a.confirm_seed = parse_int<long>(next());
        else if (match(k, "--key-confirmation-min-bits")) a.confirm_min_bits = parse_int<long>(next());
        // prefilters
        else if (match(k, "--prefilter-min-hit-rate"))   a.prefilter_min_hit_rate = parse_double(next());
        else if (match(k, "--prefilter-dz"))              a.prefilter_dz = parse_double(next());
        else if (match(k, "--prefilter-batch"))           a.prefilter_batch = parse_int<std::size_t>(next());
        else if (match(k, "--prefilter-plain-epochs"))    a.prefilter_plain_epochs = parse_int<long>(next());
        else if (match(k, "--prefilter-plain-max-loss"))  a.prefilter_plain_max_loss = parse_double(next());
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
    return a;
}

// estimate fraction of teacher sigmoid outputs outside dz around 0.5
// run on n_samples random N(0,1) inputs
// good teacher pushes most outs toward 0 or 1, bad one clusters near 0.5 and breaks key extraction
double teacher_hit_rate(
    const ssns::nn::Teacher& teacher,
    std::size_t input_dim,
    std::size_t n_samples,
    double dz,
    ssns::nn::Rng& rng)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    auto& g = rng.engine();
    ssns::linalg::Matrix X(n_samples, input_dim);
    double* d = X.data();
    for (std::size_t i = 0; i < X.size(); ++i) d[i] = dist(g);

    ssns::linalg::Matrix Y = ssns::nn::sigmoid(teacher.forward(X));
    const double lo = 0.5 - dz;
    const double hi = 0.5 + dz;
    std::size_t hits = 0;
    const double* y = Y.data();
    for (std::size_t i = 0; i < Y.size(); ++i)
        if (y[i] < lo || y[i] > hi) ++hits;
    return static_cast<double>(hits) / static_cast<double>(Y.size());
}

// inline stress test runs N rand trials of T/S key extract counts perfect matches and total mm
// output line format consumed by run_fhe_*.sh scripts
void run_inline_stress(
    const ssns::nn::CleanServer& server,
    const ssns::nn::CleanClient& client,
    std::size_t input_dim,
    std::size_t output_dim,
    std::size_t cluster_size,
    double dz,
    long n_trials,
    long stress_seed)
{
    using namespace ssns;
    std::mt19937_64 rng(static_cast<std::uint64_t>(stress_seed));
    std::normal_distribution<double> nd(0.0, 1.0);

    linalg::Matrix X(static_cast<std::size_t>(n_trials), input_dim);
    double* xd = X.data();
    for (std::size_t i = 0; i < X.size(); ++i) xd[i] = nd(rng);

    auto Y_T = nn::sigmoid(server.teacher().forward(X));
    auto Y_S = nn::sigmoid(linalg::matmul(nn::relu(linalg::matmul(X, client.W1())), client.W2()));

    const std::size_t n_clusters = output_dim / cluster_size;
    const double lo_thr = 0.5 - dz;
    const double hi_thr = 0.5 + dz;

    long perfect = 0, total_shared = 0, total_mm = 0, max_mm = 0;
    long max_shared = 0;
    std::vector<long> shared_per_trial(n_trials, 0);
    for (long r = 0; r < n_trials; ++r) {
        long sh = 0, mm = 0;
        for (std::size_t k = 0; k < n_clusters; ++k) {
            double sum_T = 0.0, sum_S = 0.0;
            for (std::size_t j = 0; j < cluster_size; ++j) {
                sum_T += Y_T(static_cast<std::size_t>(r), k * cluster_size + j);
                sum_S += Y_S(static_cast<std::size_t>(r), k * cluster_size + j);
            }
            const double mt = sum_T / static_cast<double>(cluster_size);
            const double ms = sum_S / static_cast<double>(cluster_size);
            const bool conf_T = (mt >= hi_thr) || (mt <= lo_thr);
            const bool conf_S = (ms >= hi_thr) || (ms <= lo_thr);
            const int  bit_T  = (mt >= hi_thr) ? 1 : 0;
            const int  bit_S  = (ms >= hi_thr) ? 1 : 0;
            if (conf_T && conf_S) {
                ++sh;
                if (bit_T != bit_S) ++mm;
            }
        }
        shared_per_trial[r] = sh;
        if (sh > max_shared) max_shared = sh;
        if (mm == 0) ++perfect;
        if (mm > max_mm) max_mm = mm;
        total_shared += sh;
        total_mm     += mm;
    }
    const double mean_shared = static_cast<double>(total_shared) / static_cast<double>(n_trials);
    const double mean_mm     = static_cast<double>(total_mm)     / static_cast<double>(n_trials);
    const bool sync_full = (max_mm == 0 && perfect == n_trials);

    std::cout << "STRESS perfect=" << perfect << "/" << n_trials
              << " shared_mean=" << mean_shared
              << " shared_max=" << max_shared
              << " mm_mean=" << mean_mm
              << " mm_max=" << max_mm
              << " SYNC=" << (sync_full ? "FULL" : "PARTIAL") << "\n";
}

// real key confirmation sim
// both parties derive bits from fresh X_key keep only shared confident clusters hash via sha-256 (hex_key) compare hashes
// match means protocol succeeds first try
// mismatch means hash exchange detects fail and we retry
void run_key_confirmation(
    const ssns::nn::CleanServer& server,
    const ssns::nn::CleanClient& client,
    std::size_t input_dim,
    std::size_t cluster_size,
    double dz,
    long n_trials,
    long key_seed,
    long min_bits)
{
    using namespace ssns;
    std::mt19937_64 rng(static_cast<std::uint64_t>(key_seed));
    std::normal_distribution<double> nd(0.0, 1.0);

    long matches = 0;        // hashes equal and shared >= min_bits
    long mismatches = 0;     // hashes differ retry needed but no silent corruption
    long no_key = 0;         // shared < min_bits not enough bits transparent retry
    long silent_fail = 0;    // hashes equal but bits differ impossible under sha-256
    long min_shared_bits = LONG_MAX, max_shared_bits = 0;
    double sum_shared = 0.0;

    for (long t = 0; t < n_trials; ++t) {
        // fresh X_key one rand row
        linalg::Matrix X(1, input_dim);
        for (std::size_t c = 0; c < input_dim; ++c) X(0, c) = nd(rng);

        // teacher and student forward then sigmoid
        auto Y_T = nn::sigmoid(server.teacher().forward(X));
        auto Y_S = nn::sigmoid(linalg::matmul(
            nn::relu(linalg::matmul(X, client.W1())), client.W2()));

        // flatten to plain vecs for bit extract
        std::vector<double> vT(Y_T.cols()), vS(Y_S.cols());
        for (std::size_t i = 0; i < Y_T.cols(); ++i) {
            vT[i] = Y_T(0, i);
            vS[i] = Y_S(0, i);
        }

        const auto extT = protocol::extract_with_indices(
            vT, static_cast<int>(cluster_size), dz);
        const auto extS = protocol::extract_with_indices(
            vS, static_cast<int>(cluster_size), dz);

        // keep only clusters where both T and S confident in ascending cluster idx order
        std::vector<int> bitsT_shared, bitsS_shared;
        std::size_t i = 0, j = 0;
        while (i < extT.indices.size() && j < extS.indices.size()) {
            if (extT.indices[i] == extS.indices[j]) {
                bitsT_shared.push_back(extT.bits[i]);
                bitsS_shared.push_back(extS.bits[j]);
                ++i; ++j;
            } else if (extT.indices[i] < extS.indices[j]) {
                ++i;
            } else {
                ++j;
            }
        }

        const long n_shared = static_cast<long>(bitsT_shared.size());
        sum_shared += static_cast<double>(n_shared);
        if (n_shared < min_shared_bits) min_shared_bits = n_shared;
        if (n_shared > max_shared_bits) max_shared_bits = n_shared;

        // not enough confident shared bits -> no-key trial
        // both parties just sample fresh X_key, not a security failure
        if (n_shared < min_bits) {
            ++no_key;
            continue;
        }

        // hash each side shared bits, this is what each party sends as confirmation digest
        // sha-256 collision rate is 2^-128 so silent_fail counts code bugs not real collisions
        const std::string hT = protocol::hex_key(bitsT_shared);
        const std::string hS = protocol::hex_key(bitsS_shared);
        if (hT == hS) {
            // if hashes match bits should match exact
            if (bitsT_shared == bitsS_shared) ++matches;
            else                              ++silent_fail;
        } else {
            ++mismatches;
        }
    }

    if (min_shared_bits == LONG_MAX) min_shared_bits = 0;
    const double mean_shared = sum_shared / static_cast<double>(n_trials);
    const long retries = mismatches + no_key;  // both look like a transparent retry to user
    std::cout << "KEY_CONFIRM trials=" << n_trials
              << " matched=" << matches
              << " mismatched_hash=" << mismatches
              << " no_key=" << no_key
              << " silent_fail=" << silent_fail
              << " retries=" << retries
              << " shared_min=" << min_shared_bits
              << " shared_max=" << max_shared_bits
              << " shared_mean=" << mean_shared
              << " success_first_try=" << (static_cast<double>(matches) /
                                            static_cast<double>(n_trials))
              << "\n";
}

// teacher hidden activation, viz only
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
                            a.teacher_seed, a.bfa_seed, a.teacher_w2_scale);

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

    // prefilter check teacher bimodality before commit to full run
    // exit code 3 means teacher rejected so caller can try another seed
    if (a.prefilter_min_hit_rate > 0.0) {
        nn::Rng filter_rng(a.teacher_seed ^ 0xDEAD1234ULL);
        const double hr = teacher_hit_rate(
            server.teacher(), a.t_input,
            a.prefilter_batch, a.prefilter_dz, filter_rng);
        std::cerr << "PREFILTER teacher_seed=" << a.teacher_seed
                  << " hit_rate=" << hr
                  << " threshold=" << a.prefilter_min_hit_rate
                  << " dz=" << a.prefilter_dz << "\n";
        if (hr < a.prefilter_min_hit_rate) {
            std::cerr << "PREFILTER REJECTED\n";
            std::exit(3);
        }
        std::cerr << "PREFILTER ACCEPTED\n";
    }

    // plain convergence prefilter run N cheap plain epochs as proxy for fhe convergence
    // seed that cant drop loss below threshold in plain almost certainly wont converge in fhe within budget
    // uses fresh client so real training state stays untouched
    if (a.prefilter_plain_epochs > 0) {
        nn::CleanClientConfig pf_cfg = client_cfg;
        // keep lr_total_steps = a.epochs so lr sched matches real run we just stop after prefilter_plain_epochs steps
        nn::CleanClient pf_client(pf_cfg);
        nn::CleanServer pf_server(a.t_input, a.s_hidden, a.t_hidden, a.output_dim,
                                   a.teacher_seed, a.bfa_seed, a.teacher_w2_scale);
        nn::Rng pf_rng(a.seed ^ 0xBEEF1234ULL);
        double pf_loss = 0.0;
        for (long ep = 1; ep <= a.prefilter_plain_epochs; ++ep) {
            auto step = protocol::clean_train_step(pf_client, pf_server,
                                                    a.batch_size, a.s_input,
                                                    pf_rng, a.bimodality_alpha);
            pf_loss = step.loss;
        }
        std::cerr << "PREFILTER_PLAIN teacher_seed=" << a.teacher_seed
                  << " epochs=" << a.prefilter_plain_epochs
                  << " loss=" << pf_loss
                  << " max_loss=" << a.prefilter_plain_max_loss << "\n";
        if (pf_loss > a.prefilter_plain_max_loss) {
            std::cerr << "PREFILTER_PLAIN REJECTED\n";
            std::exit(3);
        }
        std::cerr << "PREFILTER_PLAIN ACCEPTED\n";
    }

    // build fhe backend only when --use-fhe set
    // keygen is cheap few ms, cost lives in per step ciphertext arithmetic
    std::optional<ckks::Backend> fhe_backend;
    if (a.use_fhe) {
        fhe_backend = ckks::Backend::create(a.seed);
    }

    const auto started_at = std::chrono::system_clock::now();
    io::write_starting_status(a.status_path, a.epochs);

    // status write cadence
    // with fhe write every epoch (each step is expensive, user wants progress)
    // else write about once per 1% of total epochs
    // atomic write is around 1 ms negligible compared to a step
    const long status_interval = a.use_fhe
        ? 1L
        : std::max<long>(1, a.epochs / 100);
    double last_loss = 0.0;
    double last_lr   = 0.0;

    for (long ep = 1; ep <= a.epochs; ++ep) {
        auto step = a.use_fhe
            ? protocol::clean_train_step_fhe(client, server, *fhe_backend,
                                              a.batch_size, a.s_input, rng,
                                              a.bimodality_alpha)
            : (a.simulate_fhe_noise > 0.0
                ? protocol::clean_train_step_noisy(client, server,
                                                    a.batch_size, a.s_input, rng,
                                                    a.bimodality_alpha,
                                                    a.simulate_fhe_noise)
                : protocol::clean_train_step(client, server,
                                              a.batch_size, a.s_input, rng,
                                              a.bimodality_alpha));
        const double lr_now = nn::warmup_cosine_lr(
            client.step_count() - 1,
            a.epochs,
            a.warmup_frac,
            a.lr_max,
            0.0);
        last_loss = step.loss;
        last_lr   = lr_now;

        // teacher hidden + per batch error for logger sample slice
        auto H_T = teacher_hidden(server, step.X);
        // err = (Y_pred - Y_true) / batch matches json schema
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

    // important order write log first then flip running:false
    // Иначе фронт, опросивший running:false, может прочитать ещё старый training_log.json и показать неактуальные снимки
    logger.save(a.output_path);
    io::write_progress(a.status_path, a.epochs, a.epochs,
                       last_loss, last_lr,
                       started_at, /*running=*/false,
                       /*completed_at=*/std::chrono::system_clock::now());

    if (a.stress_trials > 0) {
        for (long r = 0; r < a.stress_runs; ++r) {
            run_inline_stress(server, client, a.s_input, a.output_dim,
                              a.cluster_size, a.dz, a.stress_trials,
                              a.stress_seed + r);
        }
    }

    if (a.confirm_trials > 0) {
        run_key_confirmation(server, client, a.s_input,
                             a.cluster_size, a.dz, a.confirm_trials,
                             a.confirm_seed, a.confirm_min_bits);
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "ssns-benchmark: " << e.what() << "\n";
    return 2;
}
