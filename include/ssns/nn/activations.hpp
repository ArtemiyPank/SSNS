// Element-wise activation functions used by the Student / Teacher MLPs.
// All take a Matrix and return a NEW Matrix; do not mutate the input.
// (For inner-loop performance an in-place variant could be added, but the
// callers always need both H_pre and H separately for ReLU' correction.)
#ifndef SSNS_NN_ACTIVATIONS_HPP
#define SSNS_NN_ACTIVATIONS_HPP

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

[[nodiscard]] linalg::Matrix relu(const linalg::Matrix& M);
[[nodiscard]] linalg::Matrix relu_deriv(const linalg::Matrix& M);
[[nodiscard]] linalg::Matrix sigmoid(const linalg::Matrix& M);

}  // namespace ssns::nn

#endif  // SSNS_NN_ACTIVATIONS_HPP
