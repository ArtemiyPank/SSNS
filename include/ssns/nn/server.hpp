// Server side of the Teacher-Student protocol: holds the frozen Teacher MLP
// and the Feedback-Alignment matrix B_FA.  Computes gradients server-side
// from the Student's encrypted/plaintext H, Y_pred and the Teacher's Y_true,
// using FA (B_FA in place of W2_S^T) so it never needs Student's W2.
//
// Mirrors src/ssns_clean/server.py.
#ifndef SSNS_NN_SERVER_HPP
#define SSNS_NN_SERVER_HPP

#include <cstdint>
#include <utility>

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/teacher.hpp>

namespace ssns::nn {

struct GradientPair {
    linalg::Matrix grad_W2;        // [hidden_S, output_dim]
    linalg::Matrix error_hidden;   // [batch, hidden_S]
};

class CleanServer {
public:
    CleanServer(std::size_t input_dim,
                std::size_t student_hidden_dim,
                std::size_t teacher_hidden_dim,
                std::size_t output_dim,
                std::uint64_t teacher_seed,
                std::uint64_t bfa_seed);

    [[nodiscard]] linalg::Matrix teacher_forward(const linalg::Matrix& X) const {
        return teacher_.forward(X);
    }

    // Plaintext FA gradient compute.
    //   error        = (Y_pred - Y_true) / batch
    //   grad_W2      = H^T @ error
    //   error_hidden = error @ B_FA
    [[nodiscard]] GradientPair compute_gradients(
        const linalg::Matrix& H,
        const linalg::Matrix& Y_pred,
        const linalg::Matrix& Y_true) const;

    [[nodiscard]] const Teacher&        teacher() const noexcept { return teacher_; }
    [[nodiscard]] const linalg::Matrix& b_fa()    const noexcept { return B_FA_; }

private:
    Teacher        teacher_;
    linalg::Matrix B_FA_;
};

}  // namespace ssns::nn

#endif  // SSNS_NN_SERVER_HPP
