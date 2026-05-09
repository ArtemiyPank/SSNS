#include <ssns/nn/adam.hpp>

#include <cmath>
#include <stdexcept>

namespace ssns::nn {

// one adam step on W using grad and state
//
// per element:
//   m = beta1*m + (1-beta1)*g
//   v = beta2*v + (1-beta2)*g^2
//   m_hat = m / (1 - beta1^t)         // bias correct
//   v_hat = v / (1 - beta2^t)
//   W -= lr * m_hat / (sqrt(v_hat) + eps)
//
// bias correction every step no first-step shortcut
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
    // m и v стартуют с нуля поэтому первые шаги смещены
    // деление на (1 - beta^t) убирает смещение важно в первые ~50 шагов
    const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(state.t));
    const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(state.t));

    const std::size_t n = W.size();
    double*       w   = W.data();
    const double* g   = grad.data();
    double*       m   = state.m.data();
    double*       v   = state.v.data();

    for (std::size_t i = 0; i < n; ++i) {
        const double gi = g[i];
        // 1st moment ema of g
        // beta1=0.9 значит окно ~10 шагов сглаживания градиента
        m[i] = beta1 * m[i] + (1.0 - beta1) * gi;
        // 2nd moment ema of g^2 (uncentered var)
        // beta2=0.999 окно ~1000 шагов почти статичная оценка дисперсии
        v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        const double m_hat = m[i] / bc1;
        const double v_hat = v[i] / bc2;
        // per-param step div by sqrt(v_hat) makes effective lr scale-invariant
        // i.e. doubling all grads leaves step size roughly unchanged
        // eps в знаменателе чтобы не делить на ноль когда v_hat около нуля
        w[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

}  // namespace ssns::nn
