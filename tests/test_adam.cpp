// nn::adam_step: Kingma-Ba Adam with bias correction
// reference: src/ssns_clean/client.py:198-219
//   m_t = beta1 * m_{t-1} + (1 - beta1) * grad
//   v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
//   m_hat = m_t / (1 - beta1^t)
//   v_hat = v_t / (1 - beta2^t)
//   W -= lr * m_hat / (sqrt(v_hat) + eps)
#include <catch.hpp>
#include <cmath>

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/adam.hpp>

using ssns::linalg::Matrix;
using namespace ssns::nn;

TEST_CASE("Adam: first step values match closed form", "[nn][adam]") {
    Matrix W = Matrix::full(1, 1, 1.0);
    Matrix g = Matrix::full(1, 1, 0.5);
    AdamState st(W.rows(), W.cols());
    const double lr = 0.01;
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;

    adam_step(W, g, st, lr, b1, b2, eps);

    // closed form at t=1:
    //   m = (1-b1)*g = 0.05
    //   v = (1-b2)*g^2 = 0.00025
    //   m_hat = 0.5, v_hat = 0.25
    //   delta = lr * m_hat / (sqrt(v_hat)+eps) ~ 0.01
    //   W' ~ 0.99
    REQUIRE(W(0, 0) == Approx(0.99).margin(1e-7));
    REQUIRE(st.t == 1);
}

TEST_CASE("Adam: zero gradient is a no-op", "[nn][adam]") {
    Matrix W = Matrix::full(2, 2, 5.0);
    Matrix g = Matrix::zeros(2, 2);
    AdamState st(W.rows(), W.cols());

    adam_step(W, g, st, 0.01);
    for (std::size_t i = 0; i < W.rows(); ++i)
        for (std::size_t j = 0; j < W.cols(); ++j)
            REQUIRE(W(i, j) == Approx(5.0).margin(1e-12));
    REQUIRE(st.t == 1);   // step counter still advances
}

TEST_CASE("Adam: state advances across multiple steps", "[nn][adam]") {
    Matrix W = Matrix::full(1, 1, 0.0);
    Matrix g = Matrix::full(1, 1, 1.0);
    AdamState st(W.rows(), W.cols());

    adam_step(W, g, st, 0.001);   // t = 1
    adam_step(W, g, st, 0.001);   // t = 2
    adam_step(W, g, st, 0.001);   // t = 3
    REQUIRE(st.t == 3);

    // with constant gradient, update magnitude -> lr; after 3 steps within ~50%, just check ordering
    REQUIRE(W(0, 0) < 0.0);
    REQUIRE(W(0, 0) > -0.01);
}

TEST_CASE("Adam: weight shape mismatch throws", "[nn][adam]") {
    Matrix W = Matrix::zeros(2, 3);
    Matrix g = Matrix::zeros(3, 2);
    AdamState st(W.rows(), W.cols());
    REQUIRE_THROWS(adam_step(W, g, st, 0.01));
}
