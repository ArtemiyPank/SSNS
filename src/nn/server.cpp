#include <ssns/nn/server.hpp>

#include <ssns/nn/init.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace ssns::nn {

CleanServer::CleanServer(std::size_t input_dim,
                         std::size_t student_hidden_dim,
                         std::size_t teacher_hidden_dim,
                         std::size_t output_dim,
                         std::uint64_t teacher_seed,
                         std::uint64_t bfa_seed)
    : teacher_(input_dim, teacher_hidden_dim, output_dim, teacher_seed),
      B_FA_(linalg::Matrix::zeros(output_dim, student_hidden_dim))
{
    // B_FA is sized to the *Student's* hidden — gradients flow into W1_S,
    // not W1_T.  1/sqrt(output_dim) scaling keeps Var(error_hidden) constant
    // w.r.t. output_dim (mirror the Python reference).
    Rng rng(bfa_seed);
    const double sigma = 1.0 / std::sqrt(static_cast<double>(output_dim));
    std::normal_distribution<double> dist(0.0, sigma);
    auto& g = rng.engine();
    double* d = B_FA_.data();
    const std::size_t n = B_FA_.size();
    for (std::size_t i = 0; i < n; ++i) d[i] = dist(g);
}

GradientPair CleanServer::compute_gradients(
    const linalg::Matrix& H,
    const linalg::Matrix& Y_pred,
    const linalg::Matrix& Y_true) const
{
    if (Y_pred.rows() != Y_true.rows() || Y_pred.cols() != Y_true.cols()) {
        throw std::invalid_argument(
            "compute_gradients: Y_pred and Y_true shapes differ");
    }
    if (H.rows() != Y_pred.rows()) {
        throw std::invalid_argument(
            "compute_gradients: H batch dim differs from Y_pred batch");
    }

    const auto batch = static_cast<double>(Y_pred.rows());

    // error = (Y_pred - Y_true) / batch
    auto error = linalg::sub(Y_pred, Y_true);
    error.scale_in_place(1.0 / batch);

    // grad_W2      = H^T @ error                shape [hidden_S, output]
    auto grad_W2 = linalg::matmul(linalg::transpose(H), error);
    // error_hidden = error @ B_FA               shape [batch, hidden_S]
    auto error_hidden = linalg::matmul(error, B_FA_);

    return GradientPair{ std::move(grad_W2), std::move(error_hidden) };
}

}  // namespace ssns::nn
