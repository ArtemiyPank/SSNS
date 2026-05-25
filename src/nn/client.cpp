#include <ssns/nn/client.hpp>

#include <ssns/nn/activations.hpp>
#include <ssns/nn/clip.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/lr_schedule.hpp>

#include <stdexcept>

namespace ssns::nn {
    namespace {
        // elementwise mult (hadamard) throws on shape mismatch
        // used to apply relu' (0/1 mask) onto fa error
        linalg::Matrix hadamard(const linalg::Matrix &A, const linalg::Matrix &B) {
            if (A.rows() != B.rows() || A.cols() != B.cols()) {
                throw std::invalid_argument("hadamard: shape mismatch");
            }
            linalg::Matrix R(A.rows(), A.cols());
            const double *a = A.data();
            const double *b = B.data();
            double *r = R.data();
            const std::size_t n = A.size();
            for (std::size_t i = 0; i < n; ++i) r[i] = a[i] * b[i];
            return R;
        }
    } // namespace

    // build student
    // alloc W1 W2 set up adam state for both he-init from cfg.seed
    CleanClient::CleanClient(CleanClientConfig cfg)
        : cfg_(cfg),
          W1_(linalg::Matrix::zeros(cfg.input_dim, cfg.hidden_dim)),
          W2_(linalg::Matrix::zeros(cfg.hidden_dim, cfg.output_dim)),
          st_W1_(cfg.input_dim, cfg.hidden_dim),
          st_W2_(cfg.hidden_dim, cfg.output_dim),
          step_count_(0) {
        // local rng so seed is reproducible per construction
        // shared global rng would leak state between teacher/student/B_FA inits
        Rng rng(cfg.seed);
        W1_ = he_init(cfg.input_dim, cfg.hidden_dim, cfg.input_dim, rng);
        W2_ = he_init(cfg.hidden_dim, cfg.output_dim, cfg.hidden_dim, rng);
    }

    // forward + cache X and pre-act H_pre
    // cache consumed by next update for relu' correction
    ForwardResult CleanClient::forward(const linalg::Matrix &X) {
        auto H_pre = linalg::matmul(X, W1_); // [batch hidden] pre-act
        auto H = relu(H_pre); // [batch hidden] post-act
        auto Y = linalg::matmul(H, W2_); // [batch output] linear

        saved_X_.emplace(X);
        saved_H_pre_.emplace(std::move(H_pre));
        return ForwardResult{std::move(H), std::move(Y)};
    }

    // one adam step using server grads and cached forward state
    //
    // fa path:
    //   server gives error_hidden = error @ B_FA  (no relu' factor server cant)
    //   we apply relu'(H_pre) here on client
    //   then grad_W1 = X^T @ err_corrected
    //
    // клиент применяет relu' локально потому что нужен W1
    // если бы это делал сервер ему пришлось бы расшифровать H_pre что ломает fhe
    void CleanClient::update(const linalg::Matrix &grad_W2_in, const linalg::Matrix &error_hidden) {
        if (!saved_X_ || !saved_H_pre_) {
            throw std::runtime_error("CleanClient::update called without prior forward()");
        }
        const auto &X = *saved_X_;
        const auto &H_pre = *saved_H_pre_;

        // relu' correction client side server doesnt know W1
        // relu' тут это маска 0/1 на пред-активациях zero-out мёртвых нейронов
        auto relu_d = relu_deriv(H_pre);
        auto err_corrected = hadamard(error_hidden, relu_d);

        // grad_W1 = X^T @ err_corrected shape [input hidden]
        // обычное правило цепочки backprop X^T умножается на error в скрытом слое
        auto grad_W1 = linalg::matmul(linalg::transpose(X), err_corrected);

        // mutable copy of grad_W2 so we can clip in place without touching servers matrix
        linalg::Matrix grad_W2 = grad_W2_in;
        // clip both grads independently runaway grad_W2 mustnt poison grad_W1 lr
        // L2 clip max_norm=1.0 защита от FA error spikes сохраняет направление меняет только норму
        clip_l2_inplace(grad_W1, cfg_.grad_clip_max_norm);
        clip_l2_inplace(grad_W2, cfg_.grad_clip_max_norm);

        // current lr from warmup-cosine
        const double lr_t = warmup_cosine_lr(
            step_count_, cfg_.lr_total_steps,
            cfg_.lr_warmup_frac, cfg_.lr_max, cfg_.lr_min);

        // both adam states share lr_t and adam hp
        // separate AdamState so each weight matrix has its own m/v history
        adam_step(W1_, grad_W1, st_W1_, lr_t, cfg_.beta1, cfg_.beta2, cfg_.eps);
        adam_step(W2_, grad_W2, st_W2_, lr_t, cfg_.beta1, cfg_.beta2, cfg_.eps);

        ++step_count_;
    }
} // namespace ssns::nn
