// Plaintext SSNS training step.  Mirrors src/ssns_clean/training.py
// exactly: sample X, teacher forward, student forward, server-side FA
// gradient compute, client-side update.  Returns every intermediate
// tensor for logging / inspection.
#ifndef SSNS_PROTOCOL_TRAINING_HPP
#define SSNS_PROTOCOL_TRAINING_HPP

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/client.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/server.hpp>

namespace ssns::protocol {

struct StepResult {
    linalg::Matrix X;
    linalg::Matrix Y_true;
    linalg::Matrix H;
    linalg::Matrix Y_pred;
    linalg::Matrix grad_W2;
    linalg::Matrix error_hidden;
    double         loss;
};

// One full epoch of plaintext synchronisation: fresh X each call, teacher
// target, student forward, server gradient, client update.  `loss` is the
// pre-update MSE between Y_pred and Y_true.
[[nodiscard]] StepResult clean_train_step(
    nn::CleanClient&  client,
    const nn::CleanServer& server,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng);

}  // namespace ssns::protocol

#endif  // SSNS_PROTOCOL_TRAINING_HPP
