#include <ssns/protocol/training.hpp>

#include <random>
#include <utility>

namespace ssns::protocol {

namespace {
linalg::Matrix sample_normal_batch(std::size_t batch, std::size_t cols,
                                   nn::Rng& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    auto& g = rng.engine();
    linalg::Matrix X(batch, cols);
    double* d = X.data();
    const std::size_t n = X.size();
    for (std::size_t i = 0; i < n; ++i) d[i] = dist(g);
    return X;
}

double mse_loss(const linalg::Matrix& A, const linalg::Matrix& B) {
    double acc = 0.0;
    const double* a = A.data();
    const double* b = B.data();
    const std::size_t n = A.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double d = a[i] - b[i];
        acc += d * d;
    }
    return acc / static_cast<double>(n);
}
}  // namespace

StepResult clean_train_step(
    nn::CleanClient&  client,
    const nn::CleanServer& server,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng)
{
    linalg::Matrix X      = sample_normal_batch(batch_size, input_dim, rng);
    linalg::Matrix Y_true = server.teacher_forward(X);

    auto fwd = client.forward(X);
    auto& H      = fwd.H;
    auto& Y_pred = fwd.Y_pred;

    auto grads = server.compute_gradients(H, Y_pred, Y_true);
    auto& grad_W2     = grads.grad_W2;
    auto& error_hidden = grads.error_hidden;

    const double loss = mse_loss(Y_pred, Y_true);

    client.update(grad_W2, error_hidden);

    return StepResult{
        std::move(X),
        std::move(Y_true),
        std::move(H),
        std::move(Y_pred),
        std::move(grad_W2),
        std::move(error_hidden),
        loss,
    };
}

}  // namespace ssns::protocol
