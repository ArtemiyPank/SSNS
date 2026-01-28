#include <ssns/nn/client.hpp>

#include <ssns/nn/activations.hpp>
#include <ssns/nn/clip.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/lr_schedule.hpp>

#include <stdexcept>

namespace ssns::nn {

namespace {
// Element-wise multiply (Hadamard product).  Throws on shape mismatch.
linalg::Matrix hadamard(const linalg::Matrix& A, const linalg::Matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("hadamard: shape mismatch");
    }
    linalg::Matrix R(A.rows(), A.cols());
    const double* a = A.data();
    const double* b = B.data();
    double*       r = R.data();
    const std::size_t n = A.size();
    for (std::size_t i = 0; i < n; ++i) r[i] = a[i] * b[i];
    return R;
}
}  // namespace

CleanClient::CleanClient(CleanClientConfig cfg)
    : cfg_(cfg),
      W1_(linalg::Matrix::zeros(cfg.input_dim, cfg.hidden_dim)),
      W2_(linalg::Matrix::zeros(cfg.hidden_dim, cfg.output_dim)),
      st_W1_(cfg.input_dim,  cfg.hidden_dim),
      st_W2_(cfg.hidden_dim, cfg.output_dim),
      step_count_(0)
{
    // He init (separate Rng so seed is reproducible across construction).
    Rng rng(cfg.seed);
    W1_ = he_init(cfg.input_dim,  cfg.hidden_dim, cfg.input_dim,  rng);
    W2_ = he_init(cfg.hidden_dim, cfg.output_dim, cfg.hidden_dim, rng);
}

ForwardResult CleanClient::forward(const linalg::Matrix& X) {
    auto H_pre = linalg::matmul(X, W1_);
    auto H     = relu(H_pre);
    auto Y     = linalg::matmul(H, W2_);

    saved_X_.emplace(X);
    saved_H_pre_.emplace(std::move(H_pre));
    return ForwardResult{ std::move(H), std::move(Y) };
}

void CleanClient::update(const linalg::Matrix& grad_W2_in,
                         const linalg::Matrix& error_hidden) {
    if (!saved_X_ || !saved_H_pre_) {
        throw std::runtime_error("CleanClient::update called without prior forward()");
    }
    const auto& X     = *saved_X_;
    const auto& H_pre = *saved_H_pre_;

    // ReLU' correction client-side.
    auto relu_d        = relu_deriv(H_pre);
    auto err_corrected = hadamard(error_hidden, relu_d);

    // grad_W1 = X^T @ err_corrected   shape [input, hidden]
    auto grad_W1 = linalg::matmul(linalg::transpose(X), err_corrected);

    // Mutable copies to apply L2 clip in place.
    linalg::Matrix grad_W2 = grad_W2_in;
    clip_l2_inplace(grad_W1, cfg_.grad_clip_max_norm);
    clip_l2_inplace(grad_W2, cfg_.grad_clip_max_norm);

    const double lr_t = warmup_cosine_lr(
        step_count_, cfg_.lr_total_steps,
        cfg_.lr_warmup_frac, cfg_.lr_max, cfg_.lr_min);

    adam_step(W1_, grad_W1, st_W1_, lr_t, cfg_.beta1, cfg_.beta2, cfg_.eps);
    adam_step(W2_, grad_W2, st_W2_, lr_t, cfg_.beta1, cfg_.beta2, cfg_.eps);

    ++step_count_;
}

}  // namespace ssns::nn
