// Adam optimiser state + step.  Mirrors src/ssns_clean/client.py:198-219
// of the Python reference exactly: bias-corrected on every step (no
// "skip on first step" hack), constants default to Kingma-Ba's
// (beta1=0.9, beta2=0.999, eps=1e-8).
#ifndef SSNS_NN_ADAM_HPP
#define SSNS_NN_ADAM_HPP

#include <cstdint>

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

// First and second moments + global step counter.  One AdamState per weight
// matrix (W1 and W2 each have their own).
struct AdamState {
    AdamState(std::size_t rows, std::size_t cols)
        : m(linalg::Matrix::zeros(rows, cols)),
          v(linalg::Matrix::zeros(rows, cols)),
          t(0) {}

    linalg::Matrix m;
    linalg::Matrix v;
    std::int64_t   t;
};

// Updates W in place using `grad` and the current state.  Throws
// std::invalid_argument if grad shape doesn't match W shape.
void adam_step(linalg::Matrix& W, const linalg::Matrix& grad,
               AdamState& state, double lr,
               double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);

}  // namespace ssns::nn

#endif  // SSNS_NN_ADAM_HPP
