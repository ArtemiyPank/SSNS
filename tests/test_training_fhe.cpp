// integration: clean_train_step_fhe, CKKS-encrypted variant of the SSNS step
//
// test 1: 1-step gradient parity. fixed seeds, plain vs FHE from identical init
// L_inf < 5e-2 (matches mul_cipher + rescale headline tolerance, ~10x safety margin)
//
// test 2: short convergence. CKKS noise adds per-step error but trend is preserved
//
// tiny config (input=2, S_hidden=4, output=4): 8*4=32 H ciphertexts + 32 Y_pred per step
#include <catch.hpp>

#include <cstddef>
#include <cmath>

#include <ssns/ckks/backend.hpp>
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

namespace {

// tiny config to keep per-step ciphertext count low
// parity test: input=2 hidden=4 output=4, 8*4 + 8*4 = 64 ciphertexts per step
CleanClientConfig parity_cfg() {
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

// L_inf norm of element-wise difference, matrices must have same shape
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

TEST_CASE("clean_train_step_fhe: returns step result with expected shapes",
          "[protocol][training][fhe]") {
    auto cfg = parity_cfg();
    CleanClient client(cfg);
    CleanServer server(/*input=*/2, /*student_h=*/4, /*teacher_h=*/2,
                       /*output=*/4, /*teacher_seed=*/42, /*bfa_seed=*/43);
    Backend backend = Backend::create(0xCAFE);
    Rng rng(2025);

    auto r = clean_train_step_fhe(client, server, backend,
                                  /*batch_size=*/8, /*input_dim=*/2, rng);
    REQUIRE(r.X.rows()      == 8);
    REQUIRE(r.X.cols()      == 2);
    REQUIRE(r.Y_true.rows() == 8); REQUIRE(r.Y_true.cols() == 4);
    REQUIRE(r.H.rows()      == 8); REQUIRE(r.H.cols()      == 4);
    REQUIRE(r.Y_pred.rows() == 8); REQUIRE(r.Y_pred.cols() == 4);
    REQUIRE(r.grad_W2.rows() == 4); REQUIRE(r.grad_W2.cols() == 4);
    REQUIRE(r.error_hidden.rows() == 8);
    REQUIRE(r.error_hidden.cols() == 4);
    REQUIRE(r.loss > 0.0);
}

TEST_CASE("clean_train_step_fhe: 1-step gradient parity within 5e-2",
          "[protocol][training][fhe][slow]") {
    auto cfg = parity_cfg();

    // one plaintext step
    CleanClient client_plain(cfg);
    CleanServer server_plain(2, 4, 2, 4, 42, 43);
    Rng rng_plain(2025);
    auto plain = clean_train_step(client_plain, server_plain,
                                  /*batch_size=*/8, /*input_dim=*/2, rng_plain);

    // one FHE step from identical initial state
    CleanClient client_fhe(cfg);
    CleanServer server_fhe(2, 4, 2, 4, 42, 43);
    Rng rng_fhe(2025);
    Backend backend = Backend::create(0xC0FFEEULL);
    auto fhe = clean_train_step_fhe(client_fhe, server_fhe, backend,
                                    /*batch_size=*/8, /*input_dim=*/2, rng_fhe);

    // X / Y_true / H / Y_pred computed pre-encryption from the same RNG seed, must match exactly
    REQUIRE(l_inf_diff(plain.X, fhe.X) == 0.0);
    REQUIRE(l_inf_diff(plain.Y_true, fhe.Y_true) == 0.0);
    REQUIRE(l_inf_diff(plain.H, fhe.H) == 0.0);
    REQUIRE(l_inf_diff(plain.Y_pred, fhe.Y_pred) == 0.0);

    // gradients differ only by CKKS noise, ~1e-2 absolute per scalar
    const double gw2_diff  = l_inf_diff(plain.grad_W2, fhe.grad_W2);
    const double eh_diff   = l_inf_diff(plain.error_hidden, fhe.error_hidden);
    INFO("grad_W2 L_inf diff = "       << gw2_diff);
    INFO("error_hidden L_inf diff = "  << eh_diff);
    REQUIRE(gw2_diff < 5e-2);
    REQUIRE(eh_diff  < 5e-2);

    // post-update weights agree within tolerance too
    // Adam scales gradients by ~lr/sqrt(v_hat) ~ lr, so weight diff bound is
    // gw2_diff * lr * (1 + epsilon) << 5e-2
    REQUIRE(l_inf_diff(client_plain.W1(), client_fhe.W1()) < 5e-2);
    REQUIRE(l_inf_diff(client_plain.W2(), client_fhe.W2()) < 5e-2);
}

// 20 epochs at parity-test config keeps CI cost down (~50s)
// full-config 50-epoch sweep is hidden behind [.benchmark] below, opt-in only
TEST_CASE("clean_train_step_fhe: 20-epoch loss decreases",
          "[protocol][training][fhe][slow]") {
    auto cfg = parity_cfg();
    cfg.lr_total_steps = 20;

    CleanClient client(cfg);
    CleanServer server(2, 4, 2, 4, /*teacher_seed=*/42, /*bfa_seed=*/43);
    Backend backend = Backend::create(0xBEEF);
    Rng rng(2025);

    double first_loss = 0.0;
    double last_loss  = 0.0;
    for (long ep = 1; ep <= 20; ++ep) {
        auto r = clean_train_step_fhe(client, server, backend,
                                      /*batch_size=*/8,
                                      /*input_dim=*/2, rng);
        if (ep == 1)  first_loss = r.loss;
        if (ep == 20) last_loss  = r.loss;
    }
    INFO("first_loss=" << first_loss << " last_loss=" << last_loss);
    REQUIRE(last_loss < first_loss);  // any descent counts, CKKS adds noise
}

// hidden by default, run with `./ssns_tests [benchmark]` for a longer FHE training run
TEST_CASE("clean_train_step_fhe: 50-epoch loss halves (benchmark)",
          "[.benchmark]") {
    CleanClientConfig cfg{};
    cfg.input_dim          = 4;
    cfg.hidden_dim         = 8;
    cfg.output_dim         = 10;
    cfg.lr_max             = 0.01;
    cfg.lr_total_steps     = 50;
    cfg.lr_warmup_frac     = 0.05;
    cfg.seed               = 2024;
    cfg.grad_clip_max_norm = 1.0;

    CleanClient client(cfg);
    CleanServer server(4, 8, 4, 10, 42, 43);
    Backend backend = Backend::create(0xBEEF);
    Rng rng(2025);

    double first_loss = 0.0, last_loss = 0.0;
    for (long ep = 1; ep <= 50; ++ep) {
        auto r = clean_train_step_fhe(client, server, backend, 8, 4, rng);
        if (ep == 1)  first_loss = r.loss;
        if (ep == 50) last_loss  = r.loss;
    }
    INFO("first_loss=" << first_loss << " last_loss=" << last_loss);
    REQUIRE(last_loss < first_loss * 0.5);
}
