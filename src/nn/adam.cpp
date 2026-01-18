#include <ssns/nn/adam.hpp>

#include <cmath>
#include <stdexcept>

namespace ssns::nn {

void adam_step(linalg::Matrix& W, const linalg::Matrix& grad,
               AdamState& state, double lr,
               double beta1, double beta2, double eps) {
    if (W.rows() != grad.rows() || W.cols() != grad.cols()) {
        throw std::invalid_argument("adam_step: grad shape mismatch with W");
    }
    if (state.m.rows() != W.rows() || state.m.cols() != W.cols()) {
        throw std::invalid_argument("adam_step: state shape mismatch with W");
    }

    state.t += 1;
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(state.t));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(state.t));

    const std::size_t n = W.size();
    double*       w   = W.data();
    const double* g   = grad.data();
    double*       m   = state.m.data();
    double*       v   = state.v.data();

    for (std::size_t i = 0; i < n; ++i) {
        const double gi = g[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        const double m_hat = m[i] / bc1;
        const double v_hat = v[i] / bc2;
        w[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

}  // namespace ssns::nn
