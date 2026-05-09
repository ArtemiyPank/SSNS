#include <ssns/linalg/matrix.hpp>

#include <cmath>
#include <stdexcept>

namespace ssns::linalg {

// alloc rows*cols doubles
// vector zero inits so this is also zeros
Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), storage_(rows * cols) {}

// rows x cols of zeros
Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols);
}

// rows x cols filled with value
Matrix Matrix::full(std::size_t rows, std::size_t cols, double value) {
    Matrix m(rows, cols);
    for (auto& v : m.storage_) v = value;
    return m;
}

// build from {{1,2},{3,4}}
// throws on ragged rows
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

// *= k in place
void Matrix::scale_in_place(double k) {
    for (auto& v : storage_) v *= k;
}

// sqrt of sum of squares
// just naive single pass
// осторожно: одинарный аккум может переполниться на больших матрицах
double Matrix::frobenius_norm() const {
    double acc = 0.0;
    for (auto v : storage_) acc += v * v;
    return std::sqrt(acc);
}

// free fns

// fwd decl of native kernel
// impl in matmul_native cpp
// raw ptr api so its easy to test
void matmul_native(const double* A, const double* B, double* C,
                   std::size_t M, std::size_t N, std::size_t K);

// A*B throws on shape mismatch
// A is MxK B is KxN C is MxN
// C starts as zeros so kernel can += into it
Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("matmul: A.cols != B.rows");
    }
    Matrix C = Matrix::zeros(A.rows(), B.cols());
    matmul_native(A.data(), B.data(), C.data(),
                  A.rows(), B.cols(), A.cols());
    return C;
}

// transpose of A
// just naive loop not on hot path
// помнить: write по T(j,i) идёт со страйдом A.rows() кэш-промахи неизбежны
Matrix transpose(const Matrix& A) {
    Matrix T(A.cols(), A.rows());
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            T(j, i) = A(i, j);
    return T;
}

// A+B element wise
// flat loop compiler vectorises this
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

// A-B element wise
// same as add
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
