// e2e key agreement, train then assert shared confident bits and zero mismatches
// other FHE tests only check plumbing, this one checks the actual protocol output
// tagged [slow], opt-in for CI
#include <catch.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <ssns/ckks/backend.hpp>
#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/activations.hpp>
#include <ssns/nn/client.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/server.hpp>
#include <ssns/protocol/training.hpp>

using ssns::ckks::Backend;
using ssns::linalg::Matrix;
using ssns::linalg::matmul;
using ssns::nn::CleanClient;
using ssns::nn::CleanClientConfig;
using ssns::nn::CleanServer;
using ssns::nn::Rng;
using ssns::nn::Teacher;
using ssns::nn::relu;
using ssns::nn::sigmoid;
using ssns::protocol::clean_train_step;
using ssns::protocol::clean_train_step_fhe;

namespace {

// student forward, mirrors http server, no dropout
Matrix student_sigmoid_forward(const Matrix& X,
                                const Matrix& W1, const Matrix& W2) {
    auto pre  = matmul(X, W1);
    auto post = relu(pre);
    auto raw  = matmul(post, W2);
    return sigmoid(raw);
}

// teacher forward + sigmoid
Matrix teacher_sigmoid_forward(const Teacher& t, const Matrix& X) {
    auto raw = t.forward(X);
    return sigmoid(raw);
}

// stress stats, total_mismatches must be 0 for the protocol to be useful
struct StressStats {
    double mean_shared_bits;
    double mean_teacher_confident;
    double mean_student_confident;
    long   total_mismatches;
    long   total_shared_bits;
    long   trials_with_zero_shared;
};

// run n_trials random N(0,1) inputs, count confident bits + mismatches per cluster
StressStats run_stress(const Matrix& W1, const Matrix& W2,
                        const Teacher& teacher,
                        std::size_t input_dim,
                        std::size_t cluster_size, double dz,
                        std::size_t n_trials, std::uint64_t seed)
{
    const std::size_t n_clusters = teacher.W2().cols() / cluster_size;
    REQUIRE(teacher.W2().cols() % cluster_size == 0);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    Matrix X(n_trials, input_dim);
    for (std::size_t r = 0; r < n_trials; ++r) {
        for (std::size_t c = 0; c < input_dim; ++c) X(r, c) = nd(rng);
    }
    auto Y_T = teacher_sigmoid_forward(teacher, X);
    auto Y_S = student_sigmoid_forward(X, W1, W2);

    const double lo = 0.5 - dz;
    const double hi = 0.5 + dz;

    StressStats out{0.0, 0.0, 0.0, 0, 0, 0};
    long sum_shared = 0, sum_T = 0, sum_S = 0;
    for (std::size_t r = 0; r < n_trials; ++r) {
        long ct = 0, cs = 0, sh = 0;
        for (std::size_t k = 0; k < n_clusters; ++k) {
            double sT = 0.0, sS = 0.0;
            for (std::size_t j = 0; j < cluster_size; ++j) {
                sT += Y_T(r, k * cluster_size + j);
                sS += Y_S(r, k * cluster_size + j);
            }
            const double mt = sT / cluster_size;
            const double ms = sS / cluster_size;
            const bool confT = (mt >= hi) || (mt <= lo);
            const bool confS = (ms >= hi) || (ms <= lo);
            const int  bitT  = (mt >= hi) ? 1 : 0;
            const int  bitS  = (ms >= hi) ? 1 : 0;
            if (confT) ++ct;
            if (confS) ++cs;
            if (confT && confS) {
                ++sh;
                if (bitT != bitS) ++out.total_mismatches;
            }
        }
        sum_T += ct; sum_S += cs; sum_shared += sh;
        if (sh == 0) ++out.trials_with_zero_shared;
    }
    out.mean_teacher_confident = double(sum_T)      / n_trials;
    out.mean_student_confident = double(sum_S)      / n_trials;
    out.mean_shared_bits       = double(sum_shared) / n_trials;
    out.total_shared_bits      = sum_shared;
    return out;
}

}  // namespace

TEST_CASE("key agreement: plaintext train at FHE preset config gives shared bits",
          "[protocol][key_agreement][slow]") {
    // mirrors IDE FHE preset, what the user gets if they hit that button in plaintext
    CleanClientConfig cfg{};
    cfg.input_dim          = 4;
    cfg.hidden_dim         = 16;
    cfg.output_dim         = 20;        // 4 clusters * 5
    cfg.lr_max             = 0.01;
    cfg.lr_total_steps     = 200;
    cfg.lr_warmup_frac     = 0.05;
    cfg.seed               = 2024;
    cfg.grad_clip_max_norm = 1.0;

    CleanClient client(cfg);
    CleanServer server(/*input=*/4, /*S_h=*/16, /*T_h=*/16,
                       /*output=*/20, /*teacher_seed=*/42, /*bfa_seed=*/43);
    Rng rng(2025);

    // pre-train baseline so we can prove training improves things
    auto baseline = run_stress(client.W1(), client.W2(), server.teacher(),
                                /*input_dim=*/4, /*cluster=*/5, /*dz=*/0.10,
                                /*n_trials=*/500, /*seed=*/0xCAFE);
    INFO("BASELINE: shared/trial=" << baseline.mean_shared_bits
         << " T_conf=" << baseline.mean_teacher_confident
         << " S_conf=" << baseline.mean_student_confident
         << " mismatches=" << baseline.total_mismatches);

    for (long ep = 1; ep <= 200; ++ep) {
        (void)clean_train_step(client, server, /*batch=*/8, /*input=*/4, rng);
    }

    auto trained = run_stress(client.W1(), client.W2(), server.teacher(),
                                4, 5, 0.10, 500, 0xCAFE);
    INFO("TRAINED:  shared/trial=" << trained.mean_shared_bits
         << " T_conf=" << trained.mean_teacher_confident
         << " S_conf=" << trained.mean_student_confident
         << " mismatches=" << trained.total_mismatches
         << " zero_shared=" << trained.trials_with_zero_shared);

    REQUIRE(trained.total_mismatches == 0);            // no false agreement
    REQUIRE(trained.mean_shared_bits > baseline.mean_shared_bits);  // training helps

    // sweep evidence: T_h=16 preset gives 1.84 shared/trial plaintext, FHE comes within 1.0
    // floor at 1.0 to enforce that training works and the wider Teacher choice was worth it
    REQUIRE(trained.mean_shared_bits >= 1.0);
}

// denser config (T_h=16, Y=40) at 2000 epochs, exercises the yield/epochs trade-off
//   ep=200,  dz=0.10  yield 1.91/trial, 1 mm / 4000 cbits
//   ep=200,  dz=0.13  yield 1.19/trial, 1 mm / 4000 cbits
//   ep=2000, dz=0.13  yield >= 2.0/trial, 0 mm  (this test)
TEST_CASE("key agreement: plaintext train at denser config gives 2 shared bits/trial",
          "[protocol][key_agreement][slow]") {
    constexpr long EPOCHS = 2000;        // plaintext is fast so afford full convergence

    CleanClientConfig cfg{};
    cfg.input_dim          = 8;
    cfg.hidden_dim         = 32;
    cfg.output_dim         = 40;        // 8 clusters * 5
    cfg.lr_max             = 0.01;
    cfg.lr_total_steps     = EPOCHS;
    cfg.lr_warmup_frac     = 0.05;
    cfg.seed               = 2024;
    cfg.grad_clip_max_norm = 1.0;

    CleanClient client(cfg);
    CleanServer server(8, 32, /*T_h=*/16, 40, 42, 43);
    Rng rng(2025);

    for (long ep = 1; ep <= EPOCHS; ++ep) {
        (void)clean_train_step(client, server, 8, 8, rng);
    }

    auto stats = run_stress(client.W1(), client.W2(), server.teacher(),
                              8, 5, /*dz=*/0.13, 500, 0xCAFE);
    INFO("dense plaintext (" << EPOCHS << " ep): shared/trial="
         << stats.mean_shared_bits
         << " T_conf=" << stats.mean_teacher_confident
         << " S_conf=" << stats.mean_student_confident
         << " mismatches=" << stats.total_mismatches);

    REQUIRE(stats.total_mismatches == 0);
    REQUIRE(stats.mean_shared_bits >= 1.5);
}

// FHE training at the user-facing preset, ~10 min on 16 threads, hidden by [.benchmark]
// preset matches ui/js/app.js applyFhePreset: T=4/16 S=16 Y=20 batch=4 ep=400 dz=0.10
// sweep predicts 1.97 shared/trial in plaintext, FHE adds CKKS noise so floor at 1.2
TEST_CASE("key agreement: FHE-trained at preset gives non-trivial shared bits",
          "[.benchmark][protocol][key_agreement][fhe]") {
    CleanClientConfig cfg{};
    cfg.input_dim          = 4;
    cfg.hidden_dim         = 16;
    cfg.output_dim         = 20;
    cfg.lr_max             = 0.01;
    cfg.lr_total_steps     = 400;
    cfg.lr_warmup_frac     = 0.05;
    cfg.seed               = 2024;
    cfg.grad_clip_max_norm = 1.0;

    CleanClient client(cfg);
    CleanServer server(/*input=*/4, /*S_h=*/16, /*T_h=*/16, /*output=*/20,
                       /*teacher_seed=*/42, /*bfa_seed=*/43);
    Backend backend = Backend::create(0xCAFE);
    Rng rng(2025);

    for (long ep = 1; ep <= 400; ++ep) {
        (void)clean_train_step_fhe(client, server, backend, /*batch=*/4, 4, rng);
    }

    auto stats = run_stress(client.W1(), client.W2(), server.teacher(),
                              4, 5, 0.10, 500, 0xC0FFEE);
    INFO("FHE preset (B=4 ep=400): shared/trial=" << stats.mean_shared_bits
         << " mismatches=" << stats.total_mismatches);

    REQUIRE(stats.total_mismatches == 0);
    REQUIRE(stats.mean_shared_bits >= 1.2);  // sweep predicts 1.97 plaintext
}

// plaintext config sweep, runs only when explicitly requested ([.sweep])
// reports yield + correctness across (config, dz, epochs) to inform preset choices
TEST_CASE("key agreement: plaintext config sweep (informational)",
          "[.sweep][protocol][key_agreement]") {
    struct Pt { std::size_t T_in, T_h, S_h, Y_clusters, batch; long epochs; double dz; };
    const std::vector<Pt> grid = {
        // T_in x T_h sweep, T is plaintext so free in FHE budget
        {4,  4,  16, 4, 8,  200, 0.10},   // baseline / OLD preset
        {4, 16,  16, 4, 8,  200, 0.10},   // current preset
        {4, 32,  16, 4, 8,  200, 0.10},   // 2x wider Teacher
        {4, 64,  16, 4, 8,  200, 0.10},   // 4x wider Teacher
        {8, 16,  16, 4, 8,  200, 0.10},   // 2x input
        {8, 32,  16, 4, 8,  200, 0.10},
        // S_h sweep, matters for FHE budget (2x S_h ~ 2x per-epoch)
        {4, 16,   8, 4, 8,  200, 0.10},   // half S, half cost
        {4, 16,  32, 4, 8,  200, 0.10},   // 2x S
        // Y sweep, matters for budget too
        {4, 16,  16, 8, 8,  200, 0.10},   // 8 clusters at ep=200 (risky)
        {4, 16,  16, 8, 8,  500, 0.10},   // 8 clusters at ep=500
        // batch sweep
        {4, 16,  16, 4, 4,  200, 0.10},   // half batch, half cost
        {4, 16,  16, 4, 4,  400, 0.10},   // half batch + 2x ep, same cost more SGD steps
        {4, 16,  16, 4, 4,  800, 0.10},   // half batch + 4x ep, 2x cost (still ~10 min FHE)
        {4, 16,  16, 4, 2,  200, 0.10},   // tiny batch, quarter cost
        {4, 16,  16, 4, 2,  800, 0.10},   // tiny batch + 4x ep, same cost as preset
        {4, 16,  16, 4, 2, 1600, 0.10},   // tiny batch + 8x ep, 2x cost
        {4, 16,  16, 4,16,  200, 0.10},   // 2x batch
        // dz sweep at current preset
        {4, 16,  16, 4, 8,  200, 0.05},   // narrow dead zone
        {4, 16,  16, 4, 8,  200, 0.15},   // wide dead zone
        // longer training for fuller convergence
        {4, 16,  16, 4, 8, 1000, 0.10},
        {4, 16,  16, 8, 8, 1000, 0.10},
    };

    std::cout << "\n" << std::setw(20) << std::left << "config"
              << "shared  T_conf  S_conf  mm   zero%\n";
    std::cout << std::string(60, '-') << "\n";
    for (const auto& p : grid) {
        const std::size_t Y = p.Y_clusters * 5;
        CleanClientConfig cfg{};
        cfg.input_dim          = p.T_in;
        cfg.hidden_dim         = p.S_h;
        cfg.output_dim         = Y;
        cfg.lr_max             = 0.01;
        cfg.lr_total_steps     = p.epochs;
        cfg.lr_warmup_frac     = 0.05;
        cfg.seed               = 2024;
        cfg.grad_clip_max_norm = 1.0;

        CleanClient client(cfg);
        CleanServer server(p.T_in, p.S_h, p.T_h, Y, 42, 43);
        Rng rng(2025);
        for (long ep = 0; ep < p.epochs; ++ep) {
            (void)clean_train_step(client, server, p.batch, p.T_in, rng);
        }
        auto stats = run_stress(client.W1(), client.W2(), server.teacher(),
                                  p.T_in, 5, p.dz, 1000, 0xCAFE);
        const double zero_pct = 100.0
            * static_cast<double>(stats.trials_with_zero_shared) / 1000.0;
        std::ostringstream tag;
        tag << "T" << p.T_in << "/" << p.T_h
            << " S" << p.S_h << " Y" << Y << " ep" << p.epochs
            << " dz" << p.dz;
        std::cout << std::setw(20) << std::left << tag.str()
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << stats.mean_shared_bits
                  << std::setw(8) << stats.mean_teacher_confident
                  << std::setw(8) << stats.mean_student_confident
                  << std::setw(5) << stats.total_mismatches
                  << std::setw(7) << zero_pct << "%\n";
    }
    SUCCEED("sweep produced output above");
}
