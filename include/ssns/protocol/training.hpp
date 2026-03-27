// Plaintext SSNS training step.  Mirrors src/ssns_clean/training.py
// exactly: sample X, teacher forward, student forward, server-side FA
// gradient compute, client-side update.  Returns every intermediate
// tensor for logging / inspection.
//
// `clean_train_step_fhe` is the FHE-encrypted variant: H and Y_pred are
// encrypted (per-element broadcast-CKKS-ciphertexts; one ciphertext per
// scalar with the value broadcast across all slots), the server computes
// gradients homomorphically, and the client decrypts before applying
// Adam.  Math identical to clean_train_step modulo CKKS noise (~1e-2
// per slot after one mul_cipher + rescale).
#ifndef SSNS_PROTOCOL_TRAINING_HPP
#define SSNS_PROTOCOL_TRAINING_HPP

#include <ssns/ckks/backend.hpp>
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

// FHE-encrypted variant.  Same math as `clean_train_step`, but H and
// Y_pred cross the (notional) wire encrypted under the CKKS keys held
// in `backend`.  The server's gradient compute runs homomorphically,
// and the client decrypts grad_W2 + error_hidden before applying Adam.
//
// Encoding scheme: each scalar of H / Y_pred is encrypted into its own
// CKKS ciphertext with the value broadcast across all POLY_DEGREE/2
// slots.  This trades slot packing for simpler matrix-arithmetic (no
// rotations needed) — fine for the parity / smoke tests where matrix
// dims stay small.  Slowdown vs plaintext is ~hundreds-x at this size.
//
// Result mirrors `clean_train_step` shape-for-shape; `loss` is computed
// on the plaintext Y_pred (same as the plaintext path) — Y_true and
// Y_pred are local to client-side compute, only their ciphertexts cross
// the wire.
[[nodiscard]] StepResult clean_train_step_fhe(
    nn::CleanClient&  client,
    const nn::CleanServer& server,
    ckks::Backend& backend,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng);

}  // namespace ssns::protocol

#endif  // SSNS_PROTOCOL_TRAINING_HPP
