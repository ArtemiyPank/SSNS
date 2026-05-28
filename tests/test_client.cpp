// nn::CleanClient: Student MLP, Adam + warmup-cosine LR + L2 grad clip + client-local ReLU' on FA backward
// reference: src/ssns_clean/client.py
#include <catch.hpp>

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/client.hpp>

using ssns::linalg::Matrix;
using ssns::nn::CleanClient;
using ssns::nn::CleanClientConfig;

static CleanClientConfig small_cfg() {
    CleanClientConfig c{};
    c.input_dim          = 4;
    c.hidden_dim         = 8;
    c.output_dim         = 6;
    c.lr_max             = 0.01;
    c.lr_total_steps     = 100;
    c.lr_warmup_frac     = 0.05;
    c.lr_min             = 0.0;
    c.beta1              = 0.9;
    c.beta2              = 0.999;
    c.eps                = 1e-8;
    c.grad_clip_max_norm = 1.0;
    c.seed               = 2024;
    return c;
}

TEST_CASE("CleanClient: weight shapes match input/hidden/output dims",
          "[nn][client]") {
    CleanClient c(small_cfg());
    REQUIRE(c.W1().rows() == 4);
    REQUIRE(c.W1().cols() == 8);
    REQUIRE(c.W2().rows() == 8);
    REQUIRE(c.W2().cols() == 6);
}

TEST_CASE("CleanClient: forward returns (H, Y_pred) with correct shapes",
          "[nn][client]") {
    CleanClient c(small_cfg());
    auto X = Matrix::full(3, 4, 0.5);    // batch=3
    auto [H, Y_pred] = c.forward(X);
    REQUIRE(H.rows() == 3);     REQUIRE(H.cols() == 8);
    REQUIRE(Y_pred.rows() == 3); REQUIRE(Y_pred.cols() == 6);
}

TEST_CASE("CleanClient: forward of zeros is zeros (ReLU(0) = 0)",
          "[nn][client]") {
    CleanClient c(small_cfg());
    auto X = Matrix::zeros(2, 4);
    auto [H, Y_pred] = c.forward(X);
    for (std::size_t i = 0; i < H.size();      ++i) REQUIRE(H.data()[i] == 0.0);
    for (std::size_t i = 0; i < Y_pred.size(); ++i) REQUIRE(Y_pred.data()[i] == 0.0);
}

TEST_CASE("CleanClient: update advances step_count", "[nn][client]") {
    CleanClient c(small_cfg());
    auto X = Matrix::full(2, 4, 1.0);
    (void)c.forward(X);
    REQUIRE(c.step_count() == 0);

    auto grad_W2     = Matrix::zeros(8, 6);
    auto error_hidden = Matrix::zeros(2, 8);
    c.update(grad_W2, error_hidden);
    REQUIRE(c.step_count() == 1);

    (void)c.forward(X);
    c.update(grad_W2, error_hidden);
    REQUIRE(c.step_count() == 2);
}

TEST_CASE("CleanClient: update with zero gradients leaves weights unchanged",
          "[nn][client]") {
    CleanClient c(small_cfg());
    Matrix W1_before = c.W1();   // copy
    Matrix W2_before = c.W2();

    auto X = Matrix::full(2, 4, 1.0);
    (void)c.forward(X);
    auto grad_W2     = Matrix::zeros(8, 6);
    auto error_hidden = Matrix::zeros(2, 8);
    c.update(grad_W2, error_hidden);

    for (std::size_t i = 0; i < c.W1().size(); ++i)
        REQUIRE(c.W1().data()[i] == Approx(W1_before.data()[i]));
    for (std::size_t i = 0; i < c.W2().size(); ++i)
        REQUIRE(c.W2().data()[i] == Approx(W2_before.data()[i]));
}

TEST_CASE("CleanClient: update with non-zero gradient changes weights",
          "[nn][client]") {
    CleanClient c(small_cfg());
    Matrix W2_before = c.W2();

    auto X = Matrix::full(2, 4, 1.0);
    (void)c.forward(X);
    auto grad_W2     = Matrix::full(8, 6, 0.1);
    auto error_hidden = Matrix::zeros(2, 8);   // grad_W1 stays zero
    c.update(grad_W2, error_hidden);

    bool any_diff = false;
    for (std::size_t i = 0; i < c.W2().size() && !any_diff; ++i)
        if (c.W2().data()[i] != W2_before.data()[i]) any_diff = true;
    REQUIRE(any_diff);
}

TEST_CASE("CleanClient: forward without X cache, then update throws",
          "[nn][client]") {
    CleanClient c(small_cfg());
    auto grad_W2     = Matrix::zeros(8, 6);
    auto error_hidden = Matrix::zeros(2, 8);
    REQUIRE_THROWS(c.update(grad_W2, error_hidden));
}
