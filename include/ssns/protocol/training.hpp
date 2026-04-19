// SSNS Teacher-Student synchronisation step — plaintext + FHE variants.
//
// Mirrors src/ssns_clean/training.py: sample X, teacher target, student
// forward, server-side Feedback-Alignment gradient compute, client-side
// Adam update.  The FHE variant additionally encrypts H + Y_pred before
// the server compute and decrypts the gradients afterwards.
//
//   ┌───────────────── Wire-protocol invariants (FHE path) ─────────────────┐
//   │  Client → Server   :  X (plaintext)                                   │
//   │                       H_ct, Y_pred_ct (CKKS ciphertexts)              │
//   │  Server → Client   :  grad_W2_ct, error_hidden_ct (CKKS ciphertexts)  │
//   │                                                                       │
//   │  The server function takes `const ServerKeys&` which has NO           │
//   │  SecretKey field — it is structurally impossible for server code to   │
//   │  decrypt a ciphertext.  Tests under [protocol][training][fhe][wire]   │
//   │  exercise this end-to-end.                                            │
//   └───────────────────────────────────────────────────────────────────────┘
#ifndef SSNS_PROTOCOL_TRAINING_HPP
#define SSNS_PROTOCOL_TRAINING_HPP

#include <ssns/ckks/backend.hpp>
#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/encoder.hpp>
#include <ssns/ckks/eval_key.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/public_key.hpp>
#include <ssns/ckks/secret_key.hpp>
#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/client.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/server.hpp>

#include <cstddef>
#include <random>
#include <vector>

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

// =========================================================================
// FHE wire-protocol types — every cipher-bound message between client and
// server is shaped by these structs.  Construction of either struct is
// always done by the side that owns the relevant keys; the other side can
// only consume them.
// =========================================================================

// Client→Server payload: input batch in plaintext (X is not secret — it's
// fresh Gaussian noise per epoch, sampled by the client) plus per-scalar
// CKKS ciphertexts of the student's hidden activations and outputs.
struct ClientPayload {
    linalg::Matrix X;                            // [batch x input_dim]
    std::vector<ckks::Ciphertext> H_ct;          // length = batch * H_cols
    std::vector<ckks::Ciphertext> Y_pred_ct;     // length = batch * Y_cols
    std::size_t H_cols;
    std::size_t Y_cols;
};

// Server→Client response: ONLY ciphertexts.  No plaintext side-channel.
struct ServerResponse {
    std::vector<ckks::Ciphertext> grad_W2_ct;       // length = H_cols * Y_cols
    std::vector<ckks::Ciphertext> error_hidden_ct;  // length = batch * H_cols
    std::size_t H_cols;
    std::size_t Y_cols;
    std::size_t batch_size;
};

// Plaintext gradients after client-side decryption.
struct DecryptedGradients {
    linalg::Matrix grad_W2;       // [H_cols x Y_cols]
    linalg::Matrix error_hidden;  // [batch x H_cols]
};

// =========================================================================
// Type-level key separation.
//
//   ClientKeys carries the SecretKey — only client code may hold it.
//   ServerKeys explicitly OMITS the SecretKey — server code cannot decrypt.
//
// Both structs borrow the NTT cache + Encoder by const-pointer to avoid
// expensive copies; the lifetime is tied to a long-lived ckks::Backend.
// =========================================================================

struct ClientKeys {
    const ckks::SecretKey* sk;                                 // borrow
    const ckks::PublicKey* pk;                                 // borrow
    const std::array<ckks::NTT, ckks::NUM_PRIMES>* ntts;       // borrow
    const ckks::Encoder* encoder;                              // borrow
    const ckks::Polynomial* s_ntt;                             // borrow — cached NTT-form sk
    double scale;
};

struct ServerKeys {
    const ckks::PublicKey* pk;                                 // borrow
    const ckks::EvalKey* evk;                                  // borrow
    const std::array<ckks::NTT, ckks::NUM_PRIMES>* ntts;       // borrow
    const ckks::Encoder* encoder;                              // borrow (encodes Y_true / B_FA — public)
    double scale;
    // Intentionally NO SecretKey — server code cannot construct one.
};

// Convenience: split a fully populated Backend into the two views.  The
// borrowing pointers stay valid as long as `backend` outlives the views.
[[nodiscard]] ClientKeys make_client_keys(const ckks::Backend& backend);
[[nodiscard]] ServerKeys make_server_keys(const ckks::Backend& backend);

// =========================================================================
// Three-phase FHE training step.
// =========================================================================

// Phase 1 (client side, has SecretKey).
//
// Encrypts H and Y_pred per-scalar (broadcast across all POLY_DEGREE/2
// slots).  The plaintext X is forwarded as-is.  Random noise for the
// encryption is drawn from `fhe_rng` so the same seed yields identical
// payloads — required for parity tests.
[[nodiscard]] ClientPayload client_encrypt(
    const linalg::Matrix& X,
    const linalg::Matrix& H,
    const linalg::Matrix& Y_pred,
    const ClientKeys& keys,
    std::mt19937_64& fhe_rng);

// Phase 2 (server side — NO SecretKey).
//
// Computes the FA gradients homomorphically.  The server takes:
//   - the client payload (plaintext X + ciphertexts of H, Y_pred)
//   - its private nn::CleanServer (Teacher weights + B_FA)
//   - the public/eval keys + NTTs/encoder via ServerKeys
// and returns ONLY ciphertexts.
//
// Internally: error = (Y_pred − Y_true) * (1/batch); grad_W2 = H^T·error;
// error_hidden = error·B_FA.  Y_true is computed locally from X +
// Teacher; B_FA is encoded plaintext-side.  The function never decrypts
// any input or intermediate value.
[[nodiscard]] ServerResponse server_compute_gradients(
    const ClientPayload& payload,
    const nn::CleanServer& server,
    const ServerKeys& keys);

// Phase 3 (client side, has SecretKey).
//
// Decrypts the server's response into plaintext matrices.  No mutation
// of any nn::CleanClient state happens here — the caller chooses how
// to consume the gradients (typically apply Adam).
[[nodiscard]] DecryptedGradients client_decrypt(
    const ServerResponse& response,
    const ClientKeys& keys);

// =========================================================================
// Composition: one full FHE training step.
// =========================================================================
//
// Same observable behaviour as Python's `clean_train_step(use_fhe=True)`.
// Internally it runs the three phases above plus the Adam update.  loss
// is computed plaintext-side (Y_pred — Y_true MSE).
[[nodiscard]] StepResult clean_train_step_fhe(
    nn::CleanClient&  client,
    const nn::CleanServer& server,
    ckks::Backend& backend,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng);

}  // namespace ssns::protocol

#endif  // SSNS_PROTOCOL_TRAINING_HPP
