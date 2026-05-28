// nn::Teacher: fixed random reference MLP, ref impl src/ssns_teacher.py
// forward: Y = ReLU(X @ W1_T) @ W2_T (no sigmoid in training)
// weights init via private RNG seeded by `teacher_seed`, ctor never touches global state
#include <catch.hpp>

#include <ssns/nn/teacher.hpp>

using ssns::linalg::Matrix;
using ssns::nn::Teacher;

TEST_CASE("Teacher: weights have expected shapes", "[nn][teacher]") {
    Teacher t(/*input_dim=*/64, /*hidden_dim=*/128, /*output_dim=*/640,
              /*seed=*/42);
    REQUIRE(t.W1().rows() == 64);
    REQUIRE(t.W1().cols() == 128);
    REQUIRE(t.W2().rows() == 128);
    REQUIRE(t.W2().cols() == 640);
}

TEST_CASE("Teacher: same seed -> same weights", "[nn][teacher]") {
    Teacher a(8, 4, 6, /*seed=*/123);
    Teacher b(8, 4, 6, /*seed=*/123);
    REQUIRE(a.W1().rows() == b.W1().rows());
    for (std::size_t i = 0; i < a.W1().rows(); ++i)
        for (std::size_t j = 0; j < a.W1().cols(); ++j)
            REQUIRE(a.W1()(i, j) == b.W1()(i, j));
    for (std::size_t i = 0; i < a.W2().rows(); ++i)
        for (std::size_t j = 0; j < a.W2().cols(); ++j)
            REQUIRE(a.W2()(i, j) == b.W2()(i, j));
}

TEST_CASE("Teacher: different seeds -> different weights", "[nn][teacher]") {
    Teacher a(8, 4, 6, /*seed=*/1);
    Teacher b(8, 4, 6, /*seed=*/2);
    bool any_diff = false;
    for (std::size_t i = 0; i < a.W1().size() && !any_diff; ++i) {
        if (a.W1().data()[i] != b.W1().data()[i]) any_diff = true;
    }
    REQUIRE(any_diff);
}

TEST_CASE("Teacher: forward shape [batch, input] -> [batch, output]",
          "[nn][teacher]") {
    Teacher t(/*input_dim=*/8, /*hidden_dim=*/4, /*output_dim=*/16, /*seed=*/0);
    auto X = Matrix::full(3, 8, 0.5);   // batch=3
    auto Y = t.forward(X);
    REQUIRE(Y.rows() == 3);
    REQUIRE(Y.cols() == 16);
}

TEST_CASE("Teacher: forward of all-zeros X -> all zeros Y (ReLU on zero hidden)",
          "[nn][teacher]") {
    // X = 0 -> H_pre = 0 -> H = ReLU(0) = 0 -> Y = 0 @ W2 = 0
    Teacher t(8, 4, 6, /*seed=*/7);
    auto X = Matrix::zeros(2, 8);
    auto Y = t.forward(X);
    for (std::size_t i = 0; i < Y.rows(); ++i)
        for (std::size_t j = 0; j < Y.cols(); ++j)
            REQUIRE(Y(i, j) == 0.0);
}

TEST_CASE("Teacher: forward is deterministic (same X -> same Y)", "[nn][teacher]") {
    Teacher t(8, 4, 6, /*seed=*/99);
    auto X = Matrix::full(2, 8, 1.0);
    auto Y1 = t.forward(X);
    auto Y2 = t.forward(X);
    for (std::size_t i = 0; i < Y1.size(); ++i)
        REQUIRE(Y1.data()[i] == Y2.data()[i]);
}
