#include <ssns/nn/teacher.hpp>

#include <ssns/nn/activations.hpp>
#include <ssns/nn/init.hpp>

namespace ssns::nn {

Teacher::Teacher(std::size_t input_dim, std::size_t hidden_dim,
                 std::size_t output_dim, std::uint64_t seed)
    : W1_(linalg::Matrix::zeros(input_dim, hidden_dim)),
      W2_(linalg::Matrix::zeros(hidden_dim, output_dim))
{
    Rng rng(seed);
    W1_ = he_init(input_dim,  hidden_dim, input_dim,  rng);
    W2_ = he_init(hidden_dim, output_dim, hidden_dim, rng);
}

linalg::Matrix Teacher::forward(const linalg::Matrix& X) const {
    auto H_pre = linalg::matmul(X, W1_);     // [batch, hidden]
    auto H     = relu(H_pre);
    auto Y     = linalg::matmul(H, W2_);     // [batch, output]
    return Y;
}

}  // namespace ssns::nn
