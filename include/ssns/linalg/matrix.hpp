// row major dense f64 matrix
// base type for nn tensors
// storage is std vector double shape fixed at ctor
// matmul and transpose are free fns
// matmul uses avx2 + threads no blas
// other ops are plain loops compiler vectorises them
#ifndef SSNS_LINALG_MATRIX_HPP
#define SSNS_LINALG_MATRIX_HPP

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace ssns::linalg {
    class Matrix {
    public:
        // rows x cols storage uninit
        Matrix(std::size_t rows, std::size_t cols);

        // rows x cols of zeros
        static Matrix zeros(std::size_t rows, std::size_t cols);

        // rows x cols filled with value
        static Matrix full(std::size_t rows, std::size_t cols, double value);

        // build from {{1,2},{3,4}}
        static Matrix from_rows(std::initializer_list<std::initializer_list<double> > rows);

        // element access no bounds check in release (M(e, c) = r * cols + c)
        // row major idx = r*cols + c
        // строка непрерывна в памяти столбец идёт со страйдом cols_
        double &operator()(std::size_t r, std::size_t c) { return storage_[r * cols_ + c]; }
        double operator()(std::size_t r, std::size_t c) const { return storage_[r * cols_ + c]; }

        [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
        [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
        [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }
        [[nodiscard]] double *data() noexcept { return storage_.data(); }
        [[nodiscard]] const double *data() const noexcept { return storage_.data(); }

        // *= k in place
        void scale_in_place(double k);

        // sqrt of sum of squares
        [[nodiscard]] double frobenius_norm() const;

    private:
        std::size_t rows_;
        std::size_t cols_;
        std::vector<double> storage_;
    };

    // free fn ops throw on shape mismatch

    // A*B goes through avx2 kernel
    [[nodiscard]] Matrix matmul(const Matrix &A, const Matrix &B);

    // transpose of A out of place
    [[nodiscard]] Matrix transpose(const Matrix &A);

    // A+B element wise
    [[nodiscard]] Matrix add(const Matrix &A, const Matrix &B);

    // A-B element wise
    [[nodiscard]] Matrix sub(const Matrix &A, const Matrix &B);
} // namespace ssns::linalg

#endif  // SSNS_LINALG_MATRIX_HPP
