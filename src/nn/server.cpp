#include <ssns/nn/server.hpp>

#include <ssns/nn/init.hpp>

#include <cmath>
#include <cstdlib>
#include <random>
#include <stdexcept>

namespace ssns::nn {
    // build server: build frozen teacher and sample B_FA
    //
    // fa replaces unknown W2_S^T in backward with fixed random B_FA
    // net still learns because grads align over time
    //
    // важно: B_FA фиксируется один раз никогда не пересэмплируется
    // иначе каждый шаг был бы случайным и сеть бы не сходилась
    CleanServer::CleanServer(std::size_t input_dim, std::size_t student_hidden_dim,
                             std::size_t teacher_hidden_dim, std::size_t output_dim,
                             std::uint64_t teacher_seed, std::uint64_t bfa_seed,
                             double teacher_w2_scale)
        : teacher_(input_dim, teacher_hidden_dim, output_dim, teacher_seed, teacher_w2_scale),
          B_FA_(linalg::Matrix::zeros(output_dim, student_hidden_dim)) {
        // B_FA sized to student hidden not teacher (grads flow into W1_S not W1_T)
        // common bug in early fa was tying B_FA to teacher hidden dim
        //
        // 1/sqrt(output_dim) scaling keeps Var(error_hidden) independent of output_dim:
        // error_hidden = error @ B_FA so Var ~ output_dim * Var(error) * Var(B_FA)
        // pick Var(B_FA) = 1/output_dim and output_dim cancels
        Rng rng(bfa_seed);
        const double sigma = 1.0 / std::sqrt(static_cast<double>(output_dim));
        std::normal_distribution<double> dist(0.0, sigma);
        auto &g = rng.engine();
        double *d = B_FA_.data();
        const std::size_t n = B_FA_.size();
        for (std::size_t i = 0; i < n; ++i) d[i] = dist(g);
    }

    // plaintext fa grad compute
    // returns grad_W2 and error_hidden so client can finish update locally
    //
    // pure form (no bimodality):
    //   error        = (Y_pred - Y_true) / batch       mse loss grad
    //   grad_W2      = H^T @ error                     exact server has H from client
    //   error_hidden = error @ B_FA                    fa approx of error @ W2^T
    GradientPair CleanServer::compute_gradients( const linalg::Matrix &H,
        const linalg::Matrix &Y_pred,
        const linalg::Matrix &Y_true,
        double bimodality_alpha) const {
        if (Y_pred.rows() != Y_true.rows() || Y_pred.cols() != Y_true.cols()) {
            throw std::invalid_argument(
                "compute_gradients: Y_pred and Y_true shapes differ");
        }
        if (H.rows() != Y_pred.rows()) {
            throw std::invalid_argument(
                "compute_gradients: H batch dim differs from Y_pred batch");
        }

        const auto batch = static_cast<double>(Y_pred.rows());

        // error = (Y_pred - Y_true_effective) / batch
        //
        // bimodality boost (alpha > 0):
        //   Y_true_effective_k = Y_true_k * (1 + alpha * conf_k)
        //   conf_k = 2 * |sigma(Y_true_k) - 0.5|     in [0, 1] how far teacher is from 0.5
        // net effect on error: error_k -= Y_true_k * alpha * conf_k
        // if teacher already firm on cluster k push student even harder so it lands deep in safe zone
        auto error = linalg::sub(Y_pred, Y_true);
        if (bimodality_alpha != 0.0) {
            const double *yt = Y_true.data();
            double *e = error.data();
            const std::size_t n = error.size();
            for (std::size_t i = 0; i < n; ++i) {
                // sigmoid here matches keygen pipeline (sigmoid applied at extraction time)
                // conf in [0 1] метрика того насколько teacher уверен далеко от 0.5
                const double sig = 1.0 / (1.0 + std::exp(-yt[i]));
                const double conf = 2.0 * std::abs(sig - 0.5);
                // вычитаем yt*alpha*conf эффективный target становится yt*(1+alpha*conf)
                // student целится дальше 0/1 dead zone будет легче пройти
                e[i] -= yt[i] * bimodality_alpha * conf;
            }
        }
        // делим на batch не на размер тензора потому что MSE среднее по батчу
        error.scale_in_place(1.0 / batch);

        // grad_W2 = H^T @ error shape [hidden_S output]
        // exact backprop no fa needed (H and error both server known)
        // тут именно H^T а не W2^T градиент по весам последнего слоя без FA
        auto grad_W2 = linalg::matmul(linalg::transpose(H), error);
        // error_hidden = error @ B_FA shape [batch hidden_S]
        // fa replaces W2^T with B_FA client mults by relu' before grad_W1
        // B_FA фиксирован при ctor выровнен по student hidden а не teacher
        auto error_hidden = linalg::matmul(error, B_FA_);

        return GradientPair{std::move(grad_W2), std::move(error_hidden)};
    }
} // namespace ssns::nn
