// client side (Student) of the protocol
//
// 2-layer mlp learns teacher fn without seeing teacher weights
//   forward:  H = relu(X @ W1)  Y_pred = H @ W2
//   backward: server sends grad_W2 and fa error
//             fa replaces W2^T with random fixed B_FA
//             client mults by relu' locally (server cant know W1)
//             then grad_W1 = X^T @ err_corrected
//
// state we keep
//   W1 W2                  trainable weights
//   AdamState for each     m v + step counter
//   step_count_            global step for lr sched
//   saved_X_ saved_H_pre_  cache from last forward used by next update
#ifndef SSNS_NN_CLIENT_HPP
#define SSNS_NN_CLIENT_HPP

#include <cstdint>
#include <optional>
#include <utility>

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/adam.hpp>

namespace ssns::nn {

struct CleanClientConfig {
    std::size_t   input_dim;
    std::size_t   hidden_dim;
    std::size_t   output_dim;
    // warmup-cosine schedule
    double        lr_max             = 0.01;
    long          lr_total_steps     = 1000;
    double        lr_warmup_frac     = 0.05;
    double        lr_min             = 0.0;
    // adam hp defaults
    double        beta1              = 0.9;
    double        beta2              = 0.999;
    double        eps                = 1e-8;
    // l2 grad clip stops fa error spikes blowing up training
    double        grad_clip_max_norm = 1.0;
    std::uint64_t seed               = 2024;
};

// out of one forward H and Y_pred = H @ W2
struct ForwardResult {
    linalg::Matrix H;       // relu(X @ W1) [batch hidden]
    linalg::Matrix Y_pred;  // H @ W2       [batch output]
};

class CleanClient {
public:
    // build student he-init W1 W2 from cfg.seed
    explicit CleanClient(CleanClientConfig cfg);

    // forward + cache X H_pre so next update can apply relu'
    [[nodiscard]] ForwardResult forward(const linalg::Matrix& X);

    // one adam step
    // server gives (grad_W2 error_hidden) we apply relu' to error_hidden
    // then grad_W1 = saved_X^T @ err_corrected clip both then step W1 W2
    //
    // тут именно relu' клиент-сайд сервер не знает W1
    // если сервер делал бы это сам ему пришлось бы расшифровать H_pre что ломает fhe
    void update(const linalg::Matrix& grad_W2,
                const linalg::Matrix& error_hidden);

    [[nodiscard]] const linalg::Matrix& W1() const noexcept { return W1_; }
    [[nodiscard]] const linalg::Matrix& W2() const noexcept { return W2_; }
    [[nodiscard]] long step_count()  const noexcept { return step_count_; }

private:
    CleanClientConfig cfg_;
    linalg::Matrix    W1_;
    linalg::Matrix    W2_;
    AdamState         st_W1_;
    AdamState         st_W2_;
    long              step_count_;

    // last forward cache written by forward read once by update
    // optional so first update with no prior forward throws cleanly
    std::optional<linalg::Matrix> saved_X_;
    std::optional<linalg::Matrix> saved_H_pre_;
};

}  // namespace ssns::nn

#endif  // SSNS_NN_CLIENT_HPP
