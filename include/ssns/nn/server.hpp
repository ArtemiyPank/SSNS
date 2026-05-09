// server side of protocol
//
// holds frozen Teacher and fa matrix B_FA
//
// computes grads from student H Y_pred and teacher Y_true
// fa so server never needs student W2:
//   exact:  error_hidden = error @ W2_S^T   needs W2_S server doesnt have
//   fa:     error_hidden = error @ B_FA      B_FA fixed random known to server
//
// fa: B_FA фиксируется один раз иначе grad диверг
//
// student mults error_hidden by relu' on its own side because it needs W1_S
#ifndef SSNS_NN_SERVER_HPP
#define SSNS_NN_SERVER_HPP

#include <cstdint>
#include <utility>

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/teacher.hpp>

namespace ssns::nn {

// one server step returns W2 grad and fa error client turns into grad_W1
struct GradientPair {
    linalg::Matrix grad_W2;        // [hidden_S output_dim]
    linalg::Matrix error_hidden;   // [batch hidden_S]
};

class CleanServer {
public:
    // teacher and B_FA seeded independently so same teacher reused across runs varying only B_FA
    // useful for ablation does B_FA choice matter (mostly no)
    CleanServer(std::size_t input_dim,
                std::size_t student_hidden_dim,
                std::size_t teacher_hidden_dim,
                std::size_t output_dim,
                std::uint64_t teacher_seed,
                std::uint64_t bfa_seed,
                double teacher_w2_scale = 1.0);

    // run frozen teacher on a batch
    [[nodiscard]] linalg::Matrix teacher_forward(const linalg::Matrix& X) const {
        return teacher_.forward(X);
    }

    // plaintext fa grad compute
    //   error        = (Y_pred - Y_true) / batch
    //   grad_W2      = H^T @ error           (plain backprop no fa here)
    //   error_hidden = error @ B_FA          (fa replaces error @ W2^T)
    //
    // bimodality_alpha > 0 amps Y_true on already-confident teacher clusters:
    //   target_k = Y_true_k * (1 + alpha * conf_k)
    //   conf_k = 2 * |sigma(Y_true_k) - 0.5|     in [0, 1]
    // if teacher already pushes cluster firm to 0/1 lean harder so student lands deep in safe zone
    // default 0 disables
    [[nodiscard]] GradientPair compute_gradients(
        const linalg::Matrix& H,
        const linalg::Matrix& Y_pred,
        const linalg::Matrix& Y_true,
        double bimodality_alpha = 0.0) const;

    [[nodiscard]] const Teacher&        teacher() const noexcept { return teacher_; }
    [[nodiscard]] const linalg::Matrix& b_fa()    const noexcept { return B_FA_; }

private:
    Teacher        teacher_;
    linalg::Matrix B_FA_;
};

}  // namespace ssns::nn

#endif  // SSNS_NN_SERVER_HPP
