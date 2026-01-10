// Frobenius-norm gradient clipping.  In-place to avoid allocating a copy
// inside the inner training loop.
#ifndef SSNS_NN_CLIP_HPP
#define SSNS_NN_CLIP_HPP

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

inline void clip_l2_inplace(linalg::Matrix& M, double max_norm) {
    const double n = M.frobenius_norm();
    if (n > max_norm && n > 0.0) {
        M.scale_in_place(max_norm / n);
    }
}

}  // namespace ssns::nn

#endif  // SSNS_NN_CLIP_HPP
