#include <ssns/linalg/matrix.hpp>

#ifndef SSNS_NO_BLAS
#include <cblas_minimal.h>
#endif

#include <cmath>
#include <stdexcept>

namespace ssns::linalg {

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), storage_(rows * cols) {}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols);          // std::vector<double> default = 0.0
}

Matrix Matrix::full(std::size_t rows, std::size_t cols, double value) {
    Matrix m(rows, cols);
    for (auto& v : m.storage_) v = value;
    return m;
}

Matrix Matrix::from_rows(std::initializer_list<std::initializer_list<double>> rows) {
    if (rows.size() == 0) return Matrix(0, 0);
    const std::size_t r = rows.size();
    const std::size_t c = rows.begin()->size();
    Matrix m(r, c);
    std::size_t i = 0;
    for (const auto& row : rows) {
        if (row.size() != c) {
            throw std::invalid_argument("Matrix::from_rows: ragged rows");
        }
        std::size_t j = 0;
        for (const auto& v : row) m(i, j++) = v;
        ++i;
    }
    return m;
}

void Matrix::scale_in_place(double k) {
    for (auto& v : storage_) v *= k;
}

double Matrix::frobenius_norm() const {
    double acc = 0.0;
    for (auto v : storage_) acc += v * v;
    return std::sqrt(acc);
}

// ---------------------------------------------------------------------------
// Free functions.
// ---------------------------------------------------------------------------

#ifdef SSNS_NO_BLAS
// Forward declaration of the in-tree native kernel.  Implementation lives in
// src/linalg/matmul_native.cpp and is selected when the build is configured
// with -DSSNS_USE_BLAS=OFF.  Computes C = A * B (row-major, M*K x K*N -> M*N).
void matmul_native(const double* A, const double* B, double* C,
                   std::size_t M, std::size_t N, std::size_t K);
#endif

Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("matmul: A.cols != B.rows");
    }
    Matrix C = Matrix::zeros(A.rows(), B.cols());
#ifdef SSNS_NO_BLAS
    matmul_native(A.data(), B.data(), C.data(),
                  A.rows(), B.cols(), A.cols());
#else
    const int M = static_cast<int>(A.rows());
    const int N = static_cast<int>(B.cols());
    const int K = static_cast<int>(A.cols());
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.0,
        A.data(), K,
        B.data(), N,
        0.0,
        C.data(), N);
#endif
    return C;
}

Matrix transpose(const Matrix& A) {
    Matrix T(A.cols(), A.rows());
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            T(j, i) = A(i, j);
    return T;
}

Matrix add(const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("add: shape mismatch");
    }
    Matrix R(A.rows(), A.cols());
    const double* a = A.data();
    const double* b = B.data();
    double* r = R.data();
    const std::size_t n = A.size();
    for (std::size_t i = 0; i < n; ++i) r[i] = a[i] + b[i];
    return R;
}

Matrix sub(const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("sub: shape mismatch");
    }
    Matrix R(A.rows(), A.cols());
    const double* a = A.data();
    const double* b = B.data();
    double* r = R.data();
    const std::size_t n = A.size();
    for (std::size_t i = 0; i < n; ++i) r[i] = a[i] - b[i];
    return R;
}

}  // namespace ssns::linalg
