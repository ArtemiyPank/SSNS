// L2 (Frobenius) grad clip; mirrors CleanClient._clip_l2 in Python ref
// if ||grad|| > max_norm, scale by max_norm/||grad||; else leave alone
// in-place to skip an alloc in the hot training loop
#include <catch.hpp>

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/clip.hpp>

using ssns::linalg::Matrix;
using ssns::nn::clip_l2_inplace;

TEST_CASE("clip_l2: leaves a small-norm matrix untouched", "[nn][clip]") {
    auto A = Matrix::from_rows({{0.3, 0.4}});         // norm = 0.5
    clip_l2_inplace(A, /*max_norm=*/1.0);
    REQUIRE(A(0, 0) == Approx(0.3));
    REQUIRE(A(0, 1) == Approx(0.4));
}

TEST_CASE("clip_l2: scales a too-large matrix to exactly max_norm", "[nn][clip]") {
    auto A = Matrix::from_rows({{3.0, 4.0}});         // norm = 5
    clip_l2_inplace(A, /*max_norm=*/1.0);             // scales by 1/5
    REQUIRE(A(0, 0) == Approx(0.6));
    REQUIRE(A(0, 1) == Approx(0.8));
    REQUIRE(A.frobenius_norm() == Approx(1.0).margin(1e-12));
}

TEST_CASE("clip_l2: matrix at exactly max_norm is unchanged", "[nn][clip]") {
    auto A = Matrix::from_rows({{3.0, 4.0}});         // norm = 5
    clip_l2_inplace(A, /*max_norm=*/5.0);
    REQUIRE(A(0, 0) == Approx(3.0));
    REQUIRE(A(0, 1) == Approx(4.0));
}

TEST_CASE("clip_l2: zero matrix is a no-op", "[nn][clip]") {
    auto A = Matrix::zeros(2, 3);
    clip_l2_inplace(A, /*max_norm=*/1.0);
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            REQUIRE(A(i, j) == 0.0);
}
