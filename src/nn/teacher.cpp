#include <ssns/nn/teacher.hpp>

#include <ssns/nn/activations.hpp>
#include <ssns/nn/init.hpp>

namespace ssns::nn {
    // build teacher he-init both weights from seed then optionally rescale W2
    // w2_scale > 1 sharpens output dist (more saturated sigmoid clusters at keygen)
    // w2_scale < 1 keeps it gentler default 1 leaves he init untouched
    Teacher::Teacher(std::size_t input_dim, std::size_t hidden_dim,
                     std::size_t output_dim, std::uint64_t seed,
                     double w2_scale)
        : W1_(linalg::Matrix::zeros(input_dim, hidden_dim)),
          W2_(linalg::Matrix::zeros(hidden_dim, output_dim)) {
        Rng rng(seed);
        W1_ = he_init(input_dim, hidden_dim, input_dim, rng);
        W2_ = he_init(hidden_dim, output_dim, hidden_dim, rng);
        // rescale once at ctor so forward stays a pure matmul
        if (w2_scale != 1.0) W2_.scale_in_place(w2_scale);
    }

    // linear forward Y = relu(X @ W1) @ W2
    // no sigmoid here keygen pipeline applies sigmoid on Y at extraction
    linalg::Matrix Teacher::forward(const linalg::Matrix &X) const {
        auto H_pre = linalg::matmul(X, W1_); // [batch hidden]
        auto H = relu(H_pre); // relu net was he-init for relu
        auto Y = linalg::matmul(H, W2_); // [batch output] linear
        return Y;
    }
} // namespace ssns::nn
