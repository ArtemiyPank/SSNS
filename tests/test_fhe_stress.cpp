// multi-step parity, determinism, and convergence for the FHE training step
// complements the 1-step parity check in test_training_fhe.cpp
//
// coverage:
//   - 5-step accumulated parity: plain vs FHE drift stays bounded under multiple Adam updates
//   - determinism: same seeds -> bit-exact FHE state (CKKS noise is deterministic in (sk, pk, evk, encrypt rng))
//   - independence: different backend seeds -> different ciphertexts, roughly same decrypted grads
//   - convergence monotonicity over 30 epochs at multiple checkpoints
//   - plain vs FHE final-loss race: FHE tracks plain within 2x
//
// tag: [protocol][training][fhe][slow]
#include <catch.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <ssns/ckks/backend.hpp>
#include <ssns/ckks/ciphertext.hpp>
#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/client.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/server.hpp>
#include <ssns/protocol/training.hpp>

using ssns::ckks::Backend;
using ssns::linalg::Matrix;
using ssns::nn::CleanClient;
using ssns::nn::CleanClientConfig;
using ssns::nn::CleanServer;
using ssns::nn::Rng;
using ssns::protocol::clean_train_step;
using ssns::protocol::clean_train_step_fhe;
using ssns::protocol::client_encrypt;
using ssns::protocol::ClientPayload;
using ssns::protocol::make_client_keys;
using ssns::protocol::StepResult;

namespace {

CleanClientConfig stress_cfg() {
    CleanClientConfig c{};
    c.input_dim          = 2;
    c.hidden_dim         = 4;
    c.output_dim         = 4;
    c.lr_max             = 0.01;
    c.lr_total_steps     = 200;
    c.lr_warmup_frac     = 0.05;
    c.seed               = 2024;
    c.grad_clip_max_norm = 1.0;
    return c;
}

double l_inf_diff(const Matrix& a, const Matrix& b) {
    REQUIRE(a.rows() == b.rows());
    REQUIRE(a.cols() == b.cols());
    double m = 0.0;
    const std::size_t n = a.size();
    const double* da = a.data();
    const double* db = b.data();
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(da[i] - db[i]);
        if (d > m) m = d;
    }
    return m;
}

}  // namespace

TEST_CASE("clean_train_step_fhe: 5-step parity stays within 5e-2",
          "[protocol][training][fhe][slow]") {
    // FHE step consumes one extra uint64 from main Rng (seeds the per-encrypt mt19937)
    // so naive lockstep with same seed would diverge X across steps
    //
    // instead drive both paths in one loop: sample X locally, plain grads, FHE pipeline on same X,
    // then bump an extra uint64 to mirror clean_train_step_fhe
    // pre-FHE state matches every step; only divergence is CKKS noise on decrypted grads
    auto cfg = stress_cfg();

    CleanClient client_p(cfg);
    CleanServer server_p(2, 4, 2, 4, 42, 43);
    CleanClient client_f(cfg);
    CleanServer server_f(2, 4, 2, 4, 42, 43);
    Backend backend = Backend::create(0xC0FFEEULL);

    // one shared Rng; both paths read from it identically
    Rng rng(2025);
    std::normal_distribution<double> dist(0.0, 1.0);

    constexpr int STEPS = 5;
    constexpr std::size_t batch_size = 8;
    constexpr std::size_t input_dim  = 2;

    auto ck = make_client_keys(backend);
    auto sk = ssns::protocol::make_server_keys(backend);

    for (int step = 1; step <= STEPS; ++step) {
        // sample X using the shared Rng
        Matrix X(batch_size, input_dim);
        for (std::size_t i = 0; i < X.size(); ++i)
            X.data()[i] = dist(rng.engine());

        // plaintext path on student_p
        Matrix Y_true_p = server_p.teacher_forward(X);
        auto fwd_p = client_p.forward(X);
        auto grads_p = server_p.compute_gradients(fwd_p.H, fwd_p.Y_pred, Y_true_p);
        client_p.update(grads_p.grad_W2, grads_p.error_hidden);

        // FHE path on student_f, same X
        Matrix Y_true_f = server_f.teacher_forward(X);
        auto fwd_f = client_f.forward(X);
        std::mt19937_64 fhe_rng(rng.engine()());  // same uint64 the composed step would use
        ClientPayload payload =
            client_encrypt(X, fwd_f.H, fwd_f.Y_pred, ck, fhe_rng);
        auto response = ssns::protocol::server_compute_gradients(payload, server_f, sk);
        auto grads_f = ssns::protocol::client_decrypt(response, ck);
        client_f.update(grads_f.grad_W2, grads_f.error_hidden);

        // X and Y_true are weight-independent -> exact match every step
        // H and Y_pred depend on weights; those drift from step 2 (CKKS noise differs grads)
        // drift bounded by the same 5e-2 grad budget
        REQUIRE(l_inf_diff(Y_true_p, Y_true_f) == 0.0);
        REQUIRE(l_inf_diff(fwd_p.H,      fwd_f.H)      < 5e-2);
        REQUIRE(l_inf_diff(fwd_p.Y_pred, fwd_f.Y_pred) < 5e-2);

        const double gw2 = l_inf_diff(grads_p.grad_W2,      grads_f.grad_W2);
        const double eh  = l_inf_diff(grads_p.error_hidden, grads_f.error_hidden);
        const double w1  = l_inf_diff(client_p.W1(),        client_f.W1());
        const double w2  = l_inf_diff(client_p.W2(),        client_f.W2());
        INFO("step="  << step
             << " gw2=" << gw2
             << " eh="  << eh
             << " w1="  << w1
             << " w2="  << w2);
        REQUIRE(gw2 < 5e-2);
        REQUIRE(eh  < 5e-2);
        REQUIRE(w1  < 5e-2);
        REQUIRE(w2  < 5e-2);
    }
}

TEST_CASE("clean_train_step_fhe: identical seeds → bit-exact outputs",
          "[protocol][training][fhe][slow]") {
    auto cfg = stress_cfg();

    CleanClient client_a(cfg);
    CleanServer server_a(2, 4, 2, 4, 42, 43);
    Backend       backend_a = Backend::create(0xDEADULL);
    Rng           rng_a(2025);
    auto a = clean_train_step_fhe(client_a, server_a, backend_a, 8, 2, rng_a);

    CleanClient client_b(cfg);
    CleanServer server_b(2, 4, 2, 4, 42, 43);
    Backend       backend_b = Backend::create(0xDEADULL);  // SAME seed
    Rng           rng_b(2025);                              // SAME seed
    auto b = clean_train_step_fhe(client_b, server_b, backend_b, 8, 2, rng_b);

    // every output must be bit-exact: pre-FHE state, decrypted grads, post-update weights
    // CKKS noise is deterministic given keygen rng + per-encrypt rng
    REQUIRE(a.loss == b.loss);
    REQUIRE(l_inf_diff(a.X,            b.X)            == 0.0);
    REQUIRE(l_inf_diff(a.Y_true,       b.Y_true)       == 0.0);
    REQUIRE(l_inf_diff(a.H,            b.H)            == 0.0);
    REQUIRE(l_inf_diff(a.Y_pred,       b.Y_pred)       == 0.0);
    REQUIRE(l_inf_diff(a.grad_W2,      b.grad_W2)      == 0.0);
    REQUIRE(l_inf_diff(a.error_hidden, b.error_hidden) == 0.0);
    REQUIRE(l_inf_diff(client_a.W1(),  client_b.W1())  == 0.0);
    REQUIRE(l_inf_diff(client_a.W2(),  client_b.W2())  == 0.0);
}

TEST_CASE("clean_train_step_fhe: different backend seeds → different ciphertexts but ~same gradients",
          "[protocol][training][fhe][slow]") {
    auto cfg = stress_cfg();

    // path A: backend seed 0xAAA1
    CleanClient client_a(cfg);
    CleanServer server_a(2, 4, 2, 4, 42, 43);
    Backend       backend_a = Backend::create(0xAAA1ULL);
    Rng           rng_a(2025);
    auto a = clean_train_step_fhe(client_a, server_a, backend_a, 8, 2, rng_a);

    // path B: backend seed 0xBBB2
    CleanClient client_b(cfg);
    CleanServer server_b(2, 4, 2, 4, 42, 43);
    Backend       backend_b = Backend::create(0xBBB2ULL);
    Rng           rng_b(2025);
    auto b = clean_train_step_fhe(client_b, server_b, backend_b, 8, 2, rng_b);

    // pre-FHE state still exact; only FHE noise differs
    REQUIRE(l_inf_diff(a.X,      b.X)      == 0.0);
    REQUIRE(l_inf_diff(a.Y_true, b.Y_true) == 0.0);
    REQUIRE(l_inf_diff(a.H,      b.H)      == 0.0);
    REQUIRE(l_inf_diff(a.Y_pred, b.Y_pred) == 0.0);

    // decrypted grads differ by at most ~CKKS noise budget (5e-2); both land in same neighbourhood
    const double gw2 = l_inf_diff(a.grad_W2,      b.grad_W2);
    const double eh  = l_inf_diff(a.error_hidden, b.error_hidden);
    INFO("inter-backend grad_W2 diff = "      << gw2);
    INFO("inter-backend error_hidden diff = " << eh);
    REQUIRE(gw2 < 5e-2);
    REQUIRE(eh  < 5e-2);

    // crucial: raw ciphertexts must differ; rebuild a payload per backend to inspect residues
    Matrix X(8, 2);
    {
        std::mt19937_64 g(0x1234ULL);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (std::size_t i = 0; i < X.size(); ++i) X.data()[i] = dist(g);
    }
    auto fwd_a = client_a.forward(X);  // re-uses post-update weights, that is fine
    auto fwd_b = client_b.forward(X);

    std::mt19937_64 fhe_rng_a(0x9999ULL);
    std::mt19937_64 fhe_rng_b(0x9999ULL);  // same encryption rng; only backend differs
    auto ck_a = make_client_keys(backend_a);
    auto ck_b = make_client_keys(backend_b);
    ClientPayload payload_a =
        client_encrypt(X, fwd_a.H, fwd_a.Y_pred, ck_a, fhe_rng_a);
    ClientPayload payload_b =
        client_encrypt(X, fwd_b.H, fwd_b.Y_pred, ck_b, fhe_rng_b);

    // different (sk, pk) chains -> very different poly residues
    // compare slot 0 of residue 0 of c0 between the two payloads
    const auto& res_a = payload_a.H_ct[0].c0.residues[0];
    const auto& res_b = payload_b.H_ct[0].c0.residues[0];
    REQUIRE(res_a.size() == res_b.size());

    bool any_differ = false;
    for (std::size_t i = 0; i < res_a.size(); ++i) {
        if (res_a[i] != res_b[i]) { any_differ = true; break; }
    }
    REQUIRE(any_differ);  // ciphertexts differ at the bit level
}

TEST_CASE("clean_train_step_fhe: 30-epoch loss trends down at multiple checkpoints",
          "[protocol][training][fhe][slow]") {
    // per-epoch loss is noisy (fresh X every step + Adam + CKKS)
    // larger config (4/8/10) smooths the curve so chunk averages strictly drop
    CleanClientConfig cfg{};
    cfg.input_dim          = 4;
    cfg.hidden_dim         = 8;
    cfg.output_dim         = 10;
    cfg.lr_max             = 0.01;
    cfg.lr_total_steps     = 30;
    cfg.lr_warmup_frac     = 0.05;
    cfg.seed               = 2024;
    cfg.grad_clip_max_norm = 1.0;

    CleanClient client(cfg);
    CleanServer server(4, 8, 4, 10, 42, 43);
    Backend backend = Backend::create(0xBEEFULL);
    Rng rng(2025);

    std::vector<double> losses;
    losses.reserve(30);
    for (long ep = 1; ep <= 30; ++ep) {
        auto r = clean_train_step_fhe(client, server, backend, 8, 4, rng);
        losses.push_back(r.loss);
    }

    auto chunk_avg = [&](std::size_t lo, std::size_t hi) {
        double s = 0.0;
        for (std::size_t i = lo; i < hi; ++i) s += losses[i];
        return s / static_cast<double>(hi - lo);
    };
    const double avg_a = chunk_avg(0,  10);   // epochs 1..10
    const double avg_b = chunk_avg(10, 20);   // epochs 11..20
    const double avg_c = chunk_avg(20, 30);   // epochs 21..30
    INFO("avg(1..10)=" << avg_a
         << " avg(11..20)=" << avg_b
         << " avg(21..30)=" << avg_c);
    REQUIRE(avg_b < avg_a);
    REQUIRE(avg_c < avg_b);
    REQUIRE(avg_c < 0.8 * avg_a);  // at least 20% drop end-to-end

    // running min strictly drops chunk over chunk
    // ensures the network finds BETTER points over time, not just lower variance
    auto chunk_min = [&](std::size_t lo, std::size_t hi) {
        double m = losses[lo];
        for (std::size_t i = lo + 1; i < hi; ++i)
            if (losses[i] < m) m = losses[i];
        return m;
    };
    const double min_a = chunk_min(0,  10);
    const double min_b = chunk_min(10, 20);
    const double min_c = chunk_min(20, 30);
    INFO("min(1..10)=" << min_a
         << " min(11..20)=" << min_b
         << " min(21..30)=" << min_c);
    REQUIRE(min_b < min_a);
    REQUIRE(min_c < min_b);
}

TEST_CASE("clean_train_step_fhe: tracks plaintext within 2x final loss",
          "[protocol][training][fhe][slow]") {
    auto cfg = stress_cfg();
    cfg.lr_total_steps = 30;

    // plaintext run
    CleanClient client_p(cfg);
    CleanServer server_p(2, 4, 2, 4, 42, 43);
    Rng rng_p(2025);
    double plain_final = 0.0;
    for (long ep = 1; ep <= 30; ++ep) {
        auto r = clean_train_step(client_p, server_p, 8, 2, rng_p);
        if (ep == 30) plain_final = r.loss;
    }

    // FHE run with identical seeds
    CleanClient client_f(cfg);
    CleanServer server_f(2, 4, 2, 4, 42, 43);
    Backend backend = Backend::create(0xBEEFULL);
    Rng rng_f(2025);
    double fhe_final = 0.0;
    for (long ep = 1; ep <= 30; ++ep) {
        auto r = clean_train_step_fhe(client_f, server_f, backend, 8, 2, rng_f);
        if (ep == 30) fhe_final = r.loss;
    }
    INFO("plain_final=" << plain_final << " fhe_final=" << fhe_final);

    REQUIRE(plain_final > 0.0);
    REQUIRE(fhe_final   > 0.0);
    REQUIRE(fhe_final < 2.0 * plain_final);
    REQUIRE(fhe_final > 0.5 * plain_final);
}
