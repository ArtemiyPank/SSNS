#include <ssns/nn/activations.hpp>

#include <cmath>

namespace ssns::nn {

// elementwise max(M, 0)
// cheap no exp
linalg::Matrix relu(const linalg::Matrix& M) {
    linalg::Matrix R(M.rows(), M.cols());
    const double* in  = M.data();
    double*       out = R.data();
    const std::size_t n = M.size();
    for (std::size_t i = 0; i < n; ++i) {
        // ternary lets compiler use cmov instead of branch
        out[i] = (in[i] > 0.0) ? in[i] : 0.0;
    }
    return R;
}

// 1 if M > 0 else 0
// relu' factor client mults into fa error path
// at x = 0 we return 0 same as pytorch
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

// elementwise sigmoid 1/(1+exp(-x))
// two branch form to avoid overflow:
//   x >= 0  use 1/(1+exp(-x))     exp arg non-positive
//   x <  0  use exp(x)/(1+exp(x))  exp arg non-positive
// naive form overflows to inf at x ~ -750 and gives nan
linalg::Matrix sigmoid(const linalg::Matrix& M) {
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
