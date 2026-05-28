// nn::activations: element-wise ReLU, ReLU', sigmoid
// plain loops, auto-vectorised by -O2 -march=native (no SIMD intrinsics)
#include <catch.hpp>

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/activations.hpp>

using ssns::linalg::Matrix;
using namespace ssns::nn;

TEST_CASE("activations: relu zeroes negatives, keeps positives", "[nn][activations]") {
    auto A = Matrix::from_rows({{-1.0, 0.0, 1.0},
                                {-3.5, 2.0, -7.0}});
    auto R = relu(A);
    REQUIRE(R(0, 0) == 0.0);
    REQUIRE(R(0, 1) == 0.0);
    REQUIRE(R(0, 2) == 1.0);
    REQUIRE(R(1, 0) == 0.0);
    REQUIRE(R(1, 1) == 2.0);
    REQUIRE(R(1, 2) == 0.0);
}

TEST_CASE("activations: relu_deriv 1 for positive, 0 for non-positive", "[nn][activations]") {
    // matches PyTorch / Python reference: x>0 -> 1.0, x<=0 -> 0.0 (zero treated as 0)
    auto A = Matrix::from_rows({{-1.0, 0.0, 0.5}});
    auto D = relu_deriv(A);
    REQUIRE(D(0, 0) == 0.0);
    REQUIRE(D(0, 1) == 0.0);
    REQUIRE(D(0, 2) == 1.0);
}

TEST_CASE("activations: sigmoid known values", "[nn][activations]") {
    auto A = Matrix::from_rows({{0.0, 1.0, -1.0, 100.0, -100.0}});
    auto S = sigmoid(A);
    REQUIRE(S(0, 0) == Approx(0.5));
    REQUIRE(S(0, 1) == Approx(0.7310585786).margin(1e-9));
    REQUIRE(S(0, 2) == Approx(0.2689414214).margin(1e-9));
    // numerical stability at large magnitudes
    REQUIRE(S(0, 3) == Approx(1.0));
    REQUIRE(S(0, 4) == Approx(0.0).margin(1e-30));
}

TEST_CASE("activations: relu does NOT mutate input", "[nn][activations]") {
    auto A = Matrix::from_rows({{-1.0, 2.0}});
    auto R = relu(A);
    REQUIRE(A(0, 0) == -1.0);   // original untouched
    REQUIRE(R(0, 0) == 0.0);
}
