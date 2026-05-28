// boundary + edge cases for the FHE training step
//
// extreme matrix shapes:
//   batch=1   -> no batch summation in grad_W2
//   out=1     -> degenerate Y_pred, single-output FA path
//   hid=1     -> single-neuron student layer
//   batch=16  -> long H^T * error reductions accumulate more CKKS noise (looser bound documented)
// structural: payload sizes scale linearly with batch at fixed dims
//
// tag: [protocol][training][fhe][boundary]
#include <catch.hpp>

#include <cmath>
#include <cstddef>
#include <random>

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
using ssns::protocol::client_decrypt;
using ssns::protocol::client_encrypt;
using ssns::protocol::ClientPayload;
using ssns::protocol::DecryptedGradients;
using ssns::protocol::make_client_keys;
using ssns::protocol::make_server_keys;
using ssns::protocol::server_compute_gradients;
using ssns::protocol::ServerResponse;

namespace {

CleanClientConfig cfg_for(std::size_t input_dim,
                          std::size_t hidden_dim,
                          std::size_t output_dim) {
    CleanClientConfig c{};
    c.input_dim          = input_dim;
    c.hidden_dim         = hidden_dim;
    c.output_dim         = output_dim;
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

TEST_CASE("clean_train_step_fhe: batch_size = 1",
          "[protocol][training][fhe][boundary]") {
    const std::size_t in = 2, hid = 4, out = 4, batch = 1;
    auto cfg = cfg_for(in, hid, out);

    CleanClient client_p(cfg);
    CleanServer server_p(in, hid, /*teacher_h=*/2, out, 42, 43);
    Rng rng_p(2025);
    auto plain = clean_train_step(client_p, server_p, batch, in, rng_p);

    CleanClient client_f(cfg);
    CleanServer server_f(in, hid, 2, out, 42, 43);
    Backend backend = Backend::create(0xC0FFEEULL);
    Rng rng_f(2025);
    auto fhe = clean_train_step_fhe(client_f, server_f, backend,
                                    batch, in, rng_f);

    REQUIRE(fhe.X.rows()       == batch);
    REQUIRE(fhe.grad_W2.rows() == hid);
    REQUIRE(fhe.grad_W2.cols() == out);
    REQUIRE(fhe.error_hidden.rows() == batch);
    REQUIRE(fhe.error_hidden.cols() == hid);

    REQUIRE(l_inf_diff(plain.X, fhe.X) == 0.0);
    REQUIRE(l_inf_diff(plain.H, fhe.H) == 0.0);

    const double gw2 = l_inf_diff(plain.grad_W2,      fhe.grad_W2);
    const double eh  = l_inf_diff(plain.error_hidden, fhe.error_hidden);
    INFO("batch=1 grad_W2 diff=" << gw2 << " error_hidden diff=" << eh);
    REQUIRE(gw2 < 5e-2);
    REQUIRE(eh  < 5e-2);
}

TEST_CASE("clean_train_step_fhe: output_dim = 1",
          "[protocol][training][fhe][boundary]") {
    const std::size_t in = 2, hid = 4, out = 1, batch = 8;
    auto cfg = cfg_for(in, hid, out);

    CleanClient client_p(cfg);
    CleanServer server_p(in, hid, 2, out, 42, 43);
    Rng rng_p(2025);
    auto plain = clean_train_step(client_p, server_p, batch, in, rng_p);

    CleanClient client_f(cfg);
    CleanServer server_f(in, hid, 2, out, 42, 43);
    Backend backend = Backend::create(0xC0FFEEULL);
    Rng rng_f(2025);
    auto fhe = clean_train_step_fhe(client_f, server_f, backend,
                                    batch, in, rng_f);

    REQUIRE(fhe.Y_pred.cols() == 1);
    REQUIRE(fhe.grad_W2.rows() == hid);
    REQUIRE(fhe.grad_W2.cols() == 1);
    REQUIRE(fhe.error_hidden.rows() == batch);
    REQUIRE(fhe.error_hidden.cols() == hid);

    const double gw2 = l_inf_diff(plain.grad_W2,      fhe.grad_W2);
    const double eh  = l_inf_diff(plain.error_hidden, fhe.error_hidden);
    INFO("output_dim=1 grad_W2 diff=" << gw2 << " error_hidden diff=" << eh);
    REQUIRE(gw2 < 5e-2);
    REQUIRE(eh  < 5e-2);
}

TEST_CASE("clean_train_step_fhe: hidden_dim = 1",
          "[protocol][training][fhe][boundary]") {
    const std::size_t in = 2, hid = 1, out = 4, batch = 8;
    auto cfg = cfg_for(in, hid, out);

    CleanClient client_p(cfg);
    // teacher hidden also = 1; server requires hidden >= 1 for matmul
    CleanServer server_p(in, hid, /*teacher_h=*/1, out, 42, 43);
    Rng rng_p(2025);
    auto plain = clean_train_step(client_p, server_p, batch, in, rng_p);

    CleanClient client_f(cfg);
    CleanServer server_f(in, hid, 1, out, 42, 43);
    Backend backend = Backend::create(0xC0FFEEULL);
    Rng rng_f(2025);
    auto fhe = clean_train_step_fhe(client_f, server_f, backend,
                                    batch, in, rng_f);

    REQUIRE(fhe.H.cols()             == 1);
    REQUIRE(fhe.grad_W2.rows()       == 1);
    REQUIRE(fhe.grad_W2.cols()       == out);
    REQUIRE(fhe.error_hidden.cols()  == 1);

    const double gw2 = l_inf_diff(plain.grad_W2,      fhe.grad_W2);
    const double eh  = l_inf_diff(plain.error_hidden, fhe.error_hidden);
    INFO("hidden_dim=1 grad_W2 diff=" << gw2 << " error_hidden diff=" << eh);
    REQUIRE(gw2 < 5e-2);
    REQUIRE(eh  < 5e-2);
}

TEST_CASE("clean_train_step_fhe: large batch (16) tiny dims",
          "[protocol][training][fhe][boundary]") {
    // larger batch: grad_W2 sums 16 cipher*cipher with relin + rescale residual each
    // empirically L_inf <= 1e-1 here (wider than the 5e-2 elsewhere); test pins that envelope
    const std::size_t in = 2, hid = 4, out = 4, batch = 16;
    auto cfg = cfg_for(in, hid, out);

    CleanClient client_p(cfg);
    CleanServer server_p(in, hid, 2, out, 42, 43);
    Rng rng_p(2025);
    auto plain = clean_train_step(client_p, server_p, batch, in, rng_p);

    CleanClient client_f(cfg);
    CleanServer server_f(in, hid, 2, out, 42, 43);
    Backend backend = Backend::create(0xC0FFEEULL);
    Rng rng_f(2025);
    auto fhe = clean_train_step_fhe(client_f, server_f, backend,
                                    batch, in, rng_f);

    REQUIRE(fhe.X.rows() == batch);
    const double gw2 = l_inf_diff(plain.grad_W2,      fhe.grad_W2);
    const double eh  = l_inf_diff(plain.error_hidden, fhe.error_hidden);
    INFO("batch=16 grad_W2 diff=" << gw2 << " error_hidden diff=" << eh);
    REQUIRE(gw2 < 1e-1);
    REQUIRE(eh  < 1e-1);
}

TEST_CASE("ClientPayload sizes scale linearly with batch_size",
          "[protocol][training][fhe][boundary]") {
    // structural: at fixed dims, |H_ct| = batch * H_cols and |Y_pred_ct| = batch * Y_cols
    const std::size_t in = 2, hid = 4, out = 4;
    auto cfg = cfg_for(in, hid, out);

    Backend backend = Backend::create(0xC0FFEEULL);
    auto ck = make_client_keys(backend);

    for (std::size_t batch : {std::size_t{1}, std::size_t{4}, std::size_t{8}}) {
        CleanClient client(cfg);
        std::mt19937_64 g(0x1234ULL + batch);
        std::normal_distribution<double> dist(0.0, 1.0);
        Matrix X(batch, in);
        for (std::size_t i = 0; i < X.size(); ++i) X.data()[i] = dist(g);
        auto fwd = client.forward(X);

        std::mt19937_64 fhe_rng(0xBEEFULL);
        ClientPayload payload =
            client_encrypt(X, fwd.H, fwd.Y_pred, ck, fhe_rng);

        REQUIRE(payload.H_cols       == hid);
        REQUIRE(payload.Y_cols       == out);
        REQUIRE(payload.H_ct.size()      == batch * hid);
        REQUIRE(payload.Y_pred_ct.size() == batch * out);
    }
}
