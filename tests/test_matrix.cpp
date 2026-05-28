// linalg::Matrix: row-major dense f64 with BLAS-backed matmul, transpose,
// element-wise add/sub, scalar scale, frobenius norm
// all ops deterministic, exact within f64 eps on small fixtures
#include <catch.hpp>

#include <ssns/linalg/matrix.hpp>

#include <cmath>
#include <random>

using ssns::linalg::Matrix;

namespace {

// independent triple-loop matmul, never compares kernel against itself
// must agree bit-for-bit on small inputs, within 1e-12 on random inputs
Matrix reference_matmul(const Matrix& A, const Matrix& B) {
    Matrix C = Matrix::zeros(A.rows(), B.cols());
    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < B.cols(); ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < A.cols(); ++k) {
                acc += A(i, k) * B(k, j);
            }
            C(i, j) = acc;
        }
    }
    return C;
}

// r x c matrix from U(-1, 1) with mt19937(seed)
Matrix random_matrix(std::size_t r, std::size_t c, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix m(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            m(i, j) = dist(gen);
    return m;
}

// max |A[i,j] - B[i,j]| across all entries
double max_abs_diff(const Matrix& A, const Matrix& B) {
    double m = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            m = std::max(m, std::fabs(A(i, j) - B(i, j)));
    return m;
}

}  // namespace

TEST_CASE("Matrix: construction and shape", "[linalg]") {
    Matrix m(3, 4);
    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 4);
    REQUIRE(m.size() == 12);
    REQUIRE(m.data() != nullptr);
}

TEST_CASE("Matrix: zeros / full constructors", "[linalg]") {
    auto z = Matrix::zeros(2, 3);
    for (std::size_t i = 0; i < z.rows(); ++i)
        for (std::size_t j = 0; j < z.cols(); ++j)
            REQUIRE(z(i, j) == 0.0);

    auto f = Matrix::full(2, 2, 7.5);
    REQUIRE(f(0, 0) == 7.5);
    REQUIRE(f(1, 1) == 7.5);
}

TEST_CASE("Matrix: element access (read/write)", "[linalg]") {
    Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;
    REQUIRE(m(0, 0) == 1.0);
    REQUIRE(m(0, 2) == 3.0);
    REQUIRE(m(1, 0) == 4.0);
    REQUIRE(m(1, 2) == 6.0);

    // row-major contiguous: data()[row*cols + col]
    REQUIRE(m.data()[0 * 3 + 1] == 2.0);
    REQUIRE(m.data()[1 * 3 + 2] == 6.0);
}

TEST_CASE("Matrix: from_rows initializer-list constructor", "[linalg]") {
    auto m = Matrix::from_rows({{1.0, 2.0, 3.0},
                                {4.0, 5.0, 6.0}});
    REQUIRE(m.rows() == 2);
    REQUIRE(m.cols() == 3);
    REQUIRE(m(0, 0) == 1.0);
    REQUIRE(m(1, 2) == 6.0);
}

TEST_CASE("Matrix: matmul against identity is identity", "[linalg][matmul]") {
    auto I = Matrix::zeros(3, 3);
    I(0, 0) = I(1, 1) = I(2, 2) = 1.0;
    auto A = Matrix::from_rows({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    auto C = ssns::linalg::matmul(I, A);
    REQUIRE(C.rows() == 3);
    REQUIRE(C.cols() == 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(C(i, j) == A(i, j));
}

TEST_CASE("Matrix: matmul known small fixture", "[linalg][matmul]") {
    // [1 2] [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
    // [3 4] [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
    auto A = Matrix::from_rows({{1, 2}, {3, 4}});
    auto B = Matrix::from_rows({{5, 6}, {7, 8}});
    auto C = ssns::linalg::matmul(A, B);
    REQUIRE(C.rows() == 2);
    REQUIRE(C.cols() == 2);
    REQUIRE(C(0, 0) == 19.0);
    REQUIRE(C(0, 1) == 22.0);
    REQUIRE(C(1, 0) == 43.0);
    REQUIRE(C(1, 1) == 50.0);
}

TEST_CASE("Matrix: matmul with non-square shapes", "[linalg][matmul]") {
    // [3, 2] @ [2, 4] -> [3, 4]
    auto A = Matrix::from_rows({{1, 2}, {3, 4}, {5, 6}});
    auto B = Matrix::from_rows({{1, 0, 1, 0}, {0, 1, 0, 1}});
    auto C = ssns::linalg::matmul(A, B);
    REQUIRE(C.rows() == 3);
    REQUIRE(C.cols() == 4);
    REQUIRE(C(0, 0) == 1.0); REQUIRE(C(0, 1) == 2.0); REQUIRE(C(0, 2) == 1.0); REQUIRE(C(0, 3) == 2.0);
    REQUIRE(C(2, 0) == 5.0); REQUIRE(C(2, 3) == 6.0);
}

TEST_CASE("Matrix: matmul rejects shape mismatch", "[linalg][matmul]") {
    auto A = Matrix::zeros(3, 2);
    auto B = Matrix::zeros(3, 4);   // A.cols=2 != B.rows=3, mismatch
    REQUIRE_THROWS(ssns::linalg::matmul(A, B));
}

TEST_CASE("Matrix: transpose", "[linalg]") {
    auto A = Matrix::from_rows({{1, 2, 3}, {4, 5, 6}});
    auto T = ssns::linalg::transpose(A);
    REQUIRE(T.rows() == 3);
    REQUIRE(T.cols() == 2);
    REQUIRE(T(0, 0) == 1.0);
    REQUIRE(T(1, 0) == 2.0);
    REQUIRE(T(2, 0) == 3.0);
    REQUIRE(T(0, 1) == 4.0);
    REQUIRE(T(2, 1) == 6.0);
}

TEST_CASE("Matrix: element-wise add / sub", "[linalg]") {
    auto A = Matrix::from_rows({{1, 2}, {3, 4}});
    auto B = Matrix::from_rows({{10, 20}, {30, 40}});
    auto S = ssns::linalg::add(A, B);
    REQUIRE(S(0, 0) == 11.0);
    REQUIRE(S(1, 1) == 44.0);

    auto D = ssns::linalg::sub(B, A);
    REQUIRE(D(0, 0) == 9.0);
    REQUIRE(D(1, 1) == 36.0);
}

TEST_CASE("Matrix: scale_in_place", "[linalg]") {
    auto A = Matrix::from_rows({{1, 2}, {3, 4}});
    A.scale_in_place(2.5);
    REQUIRE(A(0, 0) == 2.5);
    REQUIRE(A(1, 0) == 7.5);
    REQUIRE(A(1, 1) == 10.0);
}

TEST_CASE("Matrix: Frobenius norm (sum of squares, sqrt)", "[linalg]") {
    auto A = Matrix::from_rows({{3, 4}});       // |[3,4]| = 5
    REQUIRE(A.frobenius_norm() == Approx(5.0));

    auto B = Matrix::from_rows({{0, 0}, {0, 0}});  // zero matrix has norm 0
    REQUIRE(B.frobenius_norm() == Approx(0.0));
}

TEST_CASE("Matrix: copy and move semantics", "[linalg]") {
    auto A = Matrix::from_rows({{1, 2}, {3, 4}});
    Matrix B = A;                  // copy
    REQUIRE(B(0, 0) == 1.0);
    A(0, 0) = 99.0;                // mutate original
    REQUIRE(B(0, 0) == 1.0);       // copy unaffected

    Matrix C = std::move(A);       // move
    REQUIRE(C(0, 0) == 99.0);
    REQUIRE(C(1, 1) == 4.0);
}

// numerical parity tests, compare production matmul (in-tree AVX2 kernel)
// against the independent triple-loop reference above
// edge cases: tall-skinny, fat, non-power-of-2, near microkernel boundaries

TEST_CASE("Matrix: matmul parity vs reference (1x1)", "[linalg][matmul][parity]") {
    auto A = Matrix::from_rows({{2.5}});
    auto B = Matrix::from_rows({{4.0}});
    auto C = ssns::linalg::matmul(A, B);
    auto Cref = reference_matmul(A, B);
    REQUIRE(max_abs_diff(C, Cref) == 0.0);
    REQUIRE(C(0, 0) == 10.0);
}

TEST_CASE("Matrix: matmul parity vs reference (small odd shapes)",
          "[linalg][matmul][parity]") {
    for (auto [m, k, n] : std::initializer_list<std::tuple<int, int, int>>{
             {7, 3, 5}, {3, 11, 2}, {1, 17, 9}, {9, 1, 17}, {8, 8, 8},
             {15, 7, 31}}) {
        auto A = random_matrix(m, k, 100u + m * 31u + k);
        auto B = random_matrix(k, n, 200u + k * 31u + n);
        auto C = ssns::linalg::matmul(A, B);
        auto Cref = reference_matmul(A, B);
        INFO("shape " << m << "x" << k << " * " << k << "x" << n);
        REQUIRE(max_abs_diff(C, Cref) < 1e-12);
    }
}

TEST_CASE("Matrix: matmul parity vs reference (medium random)",
          "[linalg][matmul][parity]") {
    auto A = random_matrix(64, 48, 7u);
    auto B = random_matrix(48, 96, 13u);
    auto C = ssns::linalg::matmul(A, B);
    auto Cref = reference_matmul(A, B);
    REQUIRE(max_abs_diff(C, Cref) < 1e-12);
}

TEST_CASE("Matrix: matmul parity vs reference (tall x skinny / fat)",
          "[linalg][matmul][parity]") {
    {
        auto A = random_matrix(1, 256, 21u);
        auto B = random_matrix(256, 1, 22u);
        auto C = ssns::linalg::matmul(A, B);
        auto Cref = reference_matmul(A, B);
        REQUIRE(max_abs_diff(C, Cref) < 1e-12);
    }
    {
        auto A = random_matrix(256, 1, 23u);
        auto B = random_matrix(1, 256, 24u);
        auto C = ssns::linalg::matmul(A, B);
        auto Cref = reference_matmul(A, B);
        REQUIRE(max_abs_diff(C, Cref) < 1e-12);
    }
}

TEST_CASE("Matrix: matmul parity at microkernel-boundary shapes",
          "[linalg][matmul][parity][microkernel]") {
    // 6x8 microkernel wants M%6==0 and N%8==0, partial tiles take the scalar path
    // cover both regimes explicitly
    for (auto [m, k, n] : std::initializer_list<std::tuple<int, int, int>>{
             {1, 1, 1},        // degenerate
             {7, 3, 5},        // edge tail in M, N, K
             {8, 64, 8},       // exact NR multiple
             {6, 64, 8},       // exact MR x NR
             {12, 64, 16},     // 2x2 tile sweep
             {64, 64, 64},     // square, hits L1 path
             {128, 128, 128},  // square, hits L2 tile
             {256, 256, 256},  // square, multi-tile sweep
             {1, 1024, 1},     // dot-product geometry
             {1024, 1, 1024},  // outer-product geometry
             {97, 31, 137},    // all primes, exercises every edge
             {6, 7, 8},        // mr full / kc tail / nr full
             {5, 8, 7},        // mr tail / kc full / nr tail
         }) {
        auto A = random_matrix(m, k, 1000u + m * 7u + k * 13u + n * 19u);
        auto B = random_matrix(k, n, 2000u + m * 7u + k * 13u + n * 19u);
        auto C = ssns::linalg::matmul(A, B);
        auto Cref = reference_matmul(A, B);
        const double diff = max_abs_diff(C, Cref);
        // tolerance scales with k (sum length), 1e-12 is comfortable up to
        // k=1024 with double precision and inputs in [-1, 1]
        INFO("shape " << m << "x" << k << " * " << k << "x" << n
                      << " max_abs_diff=" << diff);
        REQUIRE(diff < 1e-12);
    }
}
