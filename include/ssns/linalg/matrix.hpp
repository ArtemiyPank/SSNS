// Row-major dense double-precision matrix.  Forms the substrate for every
// neural-net tensor in this project.  Storage is std::vector<double>; size
// is fixed at construction (no resize).  matmul and transpose are free
// functions so the call sites read like math.
//
// Performance notes:
//   * matmul calls cblas_dgemm — replaceable with a hand-written SIMD kernel
//     in a later phase by switching the implementation file.
//   * Other ops are simple element-wise loops; we let -O2 -march=native
//     auto-vectorise.
#ifndef SSNS_LINALG_MATRIX_HPP
#define SSNS_LINALG_MATRIX_HPP

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace ssns::linalg {

class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols);

    static Matrix zeros(std::size_t rows, std::size_t cols);
    static Matrix full(std::size_t rows, std::size_t cols, double value);
    static Matrix from_rows(std::initializer_list<std::initializer_list<double>> rows);

    // Element access.  Bounds checking is debug-only (relies on std::vector's
    // operator[] in release; .at would force an extra branch in hot loops).
    double& operator()(std::size_t r, std::size_t c)       { return storage_[r * cols_ + c]; }
    double  operator()(std::size_t r, std::size_t c) const { return storage_[r * cols_ + c]; }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }
    [[nodiscard]] double*       data()       noexcept { return storage_.data(); }
    [[nodiscard]] const double* data() const noexcept { return storage_.data(); }

    void scale_in_place(double k);
    [[nodiscard]] double frobenius_norm() const;

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> storage_;
};

// Free-function operations.  Throws std::invalid_argument on dimension
// mismatch; this mirrors how torch raises RuntimeError on shape mismatches.
[[nodiscard]] Matrix matmul(const Matrix& A, const Matrix& B);
[[nodiscard]] Matrix transpose(const Matrix& A);
[[nodiscard]] Matrix add(const Matrix& A, const Matrix& B);
[[nodiscard]] Matrix sub(const Matrix& A, const Matrix& B);

}  // namespace ssns::linalg

#endif  // SSNS_LINALG_MATRIX_HPP
