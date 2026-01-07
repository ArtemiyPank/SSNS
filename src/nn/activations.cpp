#include <ssns/nn/activations.hpp>

#include <cmath>

namespace ssns::nn {

linalg::Matrix relu(const linalg::Matrix& M) {
    linalg::Matrix R(M.rows(), M.cols());
    const double* in  = M.data();
    double*       out = R.data();
    const std::size_t n = M.size();
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (in[i] > 0.0) ? in[i] : 0.0;
    }
    return R;
}

linalg::Matrix relu_deriv(const linalg::Matrix& M) {
    linalg::Matrix D(M.rows(), M.cols());
    const double* in  = M.data();
    double*       out = D.data();
    const std::size_t n = M.size();
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (in[i] > 0.0) ? 1.0 : 0.0;
    }
    return D;
}

linalg::Matrix sigmoid(const linalg::Matrix& M) {
    // Numerically stable: avoid std::exp overflow at large negative magnitudes.
    linalg::Matrix S(M.rows(), M.cols());
    const double* in  = M.data();
    double*       out = S.data();
    const std::size_t n = M.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double x = in[i];
        if (x >= 0.0) {
            const double z = std::exp(-x);
            out[i] = 1.0 / (1.0 + z);
        } else {
            const double z = std::exp(x);
            out[i] = z / (1.0 + z);
        }
    }
    return S;
}

}  // namespace ssns::nn
