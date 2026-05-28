// integration: full plaintext training step
//   protocol::clean_train_step(client, server, batch_size, input_dim, rng)
// exercises every primitive: matmul, activations, L2 clip, Adam, FA gradients
// 200 epochs on tiny config must drive loss strictly down (modulo small bumps)
#include <catch.hpp>
#include <random>

#include <ssns/nn/client.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/server.hpp>
#include <ssns/protocol/training.hpp>

using ssns::linalg::Matrix;
using ssns::nn::CleanClient;
using ssns::nn::CleanClientConfig;
using ssns::nn::CleanServer;
using ssns::nn::Rng;
using ssns::protocol::clean_train_step;

// minimal client config used by every test in this file
static CleanClientConfig tiny_cfg() {
    CleanClientConfig c{};
    c.input_dim          = 4;
    c.hidden_dim         = 8;
    c.output_dim         = 10;
    c.lr_max             = 0.01;
    c.lr_total_steps     = 200;
    c.lr_warmup_frac     = 0.05;
    c.seed               = 2024;
    c.grad_clip_max_norm = 1.0;
    return c;
}

TEST_CASE("clean_train_step: returns step result with expected shapes",
          "[protocol][training]") {
    CleanClient client(tiny_cfg());
    CleanServer server(/*input_dim=*/4, /*student_hidden=*/8,
                       /*teacher_hidden=*/4, /*output_dim=*/10,
                       /*teacher_seed=*/42, /*bfa_seed=*/43);
    Rng rng(2025);

    auto r = clean_train_step(client, server, /*batch_size=*/16,
                              /*input_dim=*/4, rng);
    REQUIRE(r.X.rows()      == 16);
    REQUIRE(r.X.cols()      == 4);
    REQUIRE(r.Y_true.rows() == 16); REQUIRE(r.Y_true.cols() == 10);
    REQUIRE(r.H.rows()      == 16); REQUIRE(r.H.cols()      == 8);
    REQUIRE(r.Y_pred.rows() == 16); REQUIRE(r.Y_pred.cols() == 10);
    REQUIRE(r.grad_W2.rows() == 8); REQUIRE(r.grad_W2.cols() == 10);
    REQUIRE(r.error_hidden.rows() == 16);
    REQUIRE(r.error_hidden.cols() == 8);
    REQUIRE(r.loss > 0.0);   // not exactly 0 on a fresh init
}

TEST_CASE("clean_train_step: 200 epochs drive loss meaningfully down",
          "[protocol][training][slow]") {
    CleanClient client(tiny_cfg());
    CleanServer server(4, 8, 4, 10, /*teacher_seed=*/42, /*bfa_seed=*/43);
    Rng rng(2025);

    double first_loss = 0.0;
    double last_loss  = 0.0;
    for (long ep = 1; ep <= 200; ++ep) {
        auto r = clean_train_step(client, server, /*batch_size=*/16,
                                  /*input_dim=*/4, rng);
        if (ep == 1)   first_loss = r.loss;
        if (ep == 200) last_loss  = r.loss;
    }
    INFO("first_loss=" << first_loss << " last_loss=" << last_loss);
    REQUIRE(last_loss < first_loss * 0.5);  // at least 2x reduction
}

TEST_CASE("clean_train_step: deterministic when rng is seeded the same",
          "[protocol][training]") {
    auto run = []() {
        CleanClient client(tiny_cfg());
        CleanServer server(4, 8, 4, 10, 42, 43);
        Rng rng(7);
        double final_loss = 0.0;
        for (long ep = 1; ep <= 20; ++ep) {
            final_loss = clean_train_step(client, server, 8, 4, rng).loss;
        }
        return final_loss;
    };
    const double a = run();
    const double b = run();
    REQUIRE(a == b);
}
