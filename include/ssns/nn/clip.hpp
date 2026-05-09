// frobenius norm grad clipping
//
// keeps training stable when occasional batch gives huge grad
// fa error variance can spike when B_FA aligns badly with current err direction
//
// if ||g||_F > max_norm scale by max_norm/||g||_F else leave alone
// preserves direction only changes magnitude
//
// in-place because inner loop runs many times no point allocating
#ifndef SSNS_NN_CLIP_HPP
#define SSNS_NN_CLIP_HPP

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

// scale M down so frobenius norm <= max_norm
// no-op if already ok or matrix is zero (avoid div by zero)
inline void clip_l2_inplace(linalg::Matrix& M, double max_norm) {
    const double n = M.frobenius_norm();
    if (n > max_norm && n > 0.0) {
        M.scale_in_place(max_norm / n);
    }
}

}  // namespace ssns::nn

#endif  // SSNS_NN_CLIP_HPP
