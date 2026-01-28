// Client side (Student) of the Teacher-Student protocol.  Owns its W1, W2,
// the Adam optimiser state for both, a step counter for the LR scheduler,
// and a forward-pass cache (X, H_pre) used by update() for the client-local
// ReLU' correction on the Feedback-Alignment backward path.
//
// Mirrors src/ssns_clean/client.py.
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
    double        lr_max             = 0.01;
    long          lr_total_steps     = 1000;
    double        lr_warmup_frac     = 0.05;
    double        lr_min             = 0.0;
    double        beta1              = 0.9;
    double        beta2              = 0.999;
    double        eps                = 1e-8;
    double        grad_clip_max_norm = 1.0;
    std::uint64_t seed               = 2024;
};

struct ForwardResult {
    linalg::Matrix H;       // ReLU(X @ W1)
    linalg::Matrix Y_pred;  // H @ W2
};

class CleanClient {
public:
    explicit CleanClient(CleanClientConfig cfg);

    // Caches X and H_pre for the next update() call.
    [[nodiscard]] ForwardResult forward(const linalg::Matrix& X);

    // Applies one Adam step using server-returned (grad_W2, error_hidden).
    // Performs client-local ReLU' correction on error_hidden, computes
    // grad_W1 = saved_X.T @ err_corrected, L2-clips both grads, then steps.
    void update(const linalg::Matrix& grad_W2,
                const linalg::Matrix& error_hidden);

    [[nodiscard]] const linalg::Matrix& W1() const noexcept { return W1_; }
    [[nodiscard]] const linalg::Matrix& W2() const noexcept { return W2_; }
    [[nodiscard]] long step_count()  const noexcept { return step_count_; }
    [[nodiscard]] const CleanClientConfig& config() const noexcept { return cfg_; }

private:
    CleanClientConfig cfg_;
    linalg::Matrix    W1_;
    linalg::Matrix    W2_;
    AdamState         st_W1_;
    AdamState         st_W2_;
    long              step_count_;

    // Forward-pass cache.  Filled by forward(), consumed by update().
    std::optional<linalg::Matrix> saved_X_;
    std::optional<linalg::Matrix> saved_H_pre_;
};

}  // namespace ssns::nn

#endif  // SSNS_NN_CLIENT_HPP
