// adam optimizer state and one step
//
// per-param lr that adapts to local grad scale
//   m = beta1*m + (1-beta1)*g          // 1st moment mean of g
//   v = beta2*v + (1-beta2)*g^2        // 2nd moment uncentered var
//   m_hat = m / (1 - beta1^t)          // bias correct since m v start at 0
//   v_hat = v / (1 - beta2^t)
//   W -= lr * m_hat / (sqrt(v_hat) + eps)
//
// bias correction matters for first ~50 steps else early steps too small
// defaults beta1=0.9 beta2=0.999 eps=1e-8
#ifndef SSNS_NN_ADAM_HPP
#define SSNS_NN_ADAM_HPP

#include <cstdint>

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

// m v matrices + step counter
// one AdamState per weight matrix (student has two for W1 W2)
struct AdamState {
    // zero init m and v t = 0
    // важно: m и v нули иначе bias correction не сработает на старте
    AdamState(std::size_t rows, std::size_t cols)
        : m(linalg::Matrix::zeros(rows, cols)),
          v(linalg::Matrix::zeros(rows, cols)),
          t(0) {}

    linalg::Matrix m;  // 1st moment running mean of g
    linalg::Matrix v;  // 2nd moment running mean of g^2
    std::int64_t   t;  // step counter for bias correction
};

// one adam step on W using grad and state
// throws if grad shape != W shape
void adam_step(linalg::Matrix& W, const linalg::Matrix& grad,
               AdamState& state, double lr,
               double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);

}  // namespace ssns::nn

#endif  // SSNS_NN_ADAM_HPP
