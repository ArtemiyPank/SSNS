// element-wise activations for student and teacher mlps
// take Matrix return new one input never mutated
//
// not in-place because fa backward needs both H_pre and H = relu(H_pre) at same time
//
// relu inside training sigmoid only at keygen
// sigmoid inside fa would kill grad ~0.25x per layer
#ifndef SSNS_NN_ACTIVATIONS_HPP
#define SSNS_NN_ACTIVATIONS_HPP

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

// max(M, 0) elementwise
[[nodiscard]] linalg::Matrix relu(const linalg::Matrix& M);

// 1 if M > 0 else 0
// used as relu' on fa error in CleanClient::update
// at exactly 0 we return 0 same as pytorch
[[nodiscard]] linalg::Matrix relu_deriv(const linalg::Matrix& M);

// 1 / (1 + exp(-M))
// two branches to avoid exp overflow on big negs
[[nodiscard]] linalg::Matrix sigmoid(const linalg::Matrix& M);

}  // namespace ssns::nn

#endif  // SSNS_NN_ACTIVATIONS_HPP
