// SSNS teacher student sync step plain and FHE variants
//
// per call sample X get teacher target run student forward server FA grads client update
// FHE variant encrypts H and Y_pred before server step then decrypts grads back
//
// wire protocol on FHE path:
//   client -> server: X plain plus H_ct Y_pred_ct ciphertexts
//   server -> client: grad_W2_ct error_hidden_ct ciphertexts
//
// ServerKeys has no SecretKey field server cannot decrypt by construction
// tests under [protocol][training][fhe][wire] cover this
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
    double loss;
};

// one plain sync step fresh X teacher target student forward server grad client update
// loss is pre update MSE
[[nodiscard]] StepResult clean_train_step(
    nn::CleanClient& client,
    const nn::CleanServer& server,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng,
    double bimodality_alpha = 0.0);

// plain step plus gaussian noise on grads
// fakes CKKS error ~100x faster than real FHE
// quick prescreen for configs noise_std=0 same as plain
[[nodiscard]] StepResult clean_train_step_noisy(
    nn::CleanClient& client,
    const nn::CleanServer& server,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng,
    double bimodality_alpha,
    double noise_std);

// FHE wire types
// every cipher message between client and server is one of these
// always built by the side with the keys other side only reads

// client -> server payload
// X is plain (not secret fresh gaussian per epoch)
// H_ct and Y_pred_ct per scalar ciphertexts
struct ClientPayload {
    linalg::Matrix X;                            // [batch x input_dim]
    std::vector<ckks::Ciphertext> H_ct;          // batch * H_cols
    std::vector<ckks::Ciphertext> Y_pred_ct;     // batch * Y_cols
    std::size_t H_cols;
    std::size_t Y_cols;
};

// server -> client response ciphertexts only
struct ServerResponse {
    std::vector<ckks::Ciphertext> grad_W2_ct;       // H_cols * Y_cols
    std::vector<ckks::Ciphertext> error_hidden_ct;  // batch * H_cols
    std::size_t H_cols;
    std::size_t Y_cols;
    std::size_t batch_size;
};

// plain grads after client decrypt
struct DecryptedGradients {
    linalg::Matrix grad_W2;       // [H_cols x Y_cols]
    linalg::Matrix error_hidden;  // [batch x H_cols]
};

// type level key separation
//
// ClientKeys carries SecretKey only client may hold it
// ServerKeys leaves SecretKey out server cannot decrypt
//
// both borrow NTT cache and Encoder by const ptr
// pointers valid as long as backing ckks::Backend lives

struct ClientKeys {
    const ckks::SecretKey* sk;                                 // borrow
    const ckks::PublicKey* pk;                                 // borrow
    const std::array<ckks::NTT, ckks::NUM_PRIMES>* ntts;       // borrow
    const ckks::Encoder* encoder;                              // borrow
    const ckks::Polynomial* s_ntt;                             // borrow cached NTT sk
    double scale;
};

struct ServerKeys {
    const ckks::PublicKey* pk;                                 // borrow
    const ckks::EvalKey* evk;                                  // borrow
    const std::array<ckks::NTT, ckks::NUM_PRIMES>* ntts;       // borrow
    const ckks::Encoder* encoder;                              // borrow
    double scale;
    // SecretKey absent on purpose
};

// split a populated Backend into the two views
// borrowed pointers stay valid while backend outlives the views
[[nodiscard]] ClientKeys make_client_keys(const ckks::Backend& backend);
[[nodiscard]] ServerKeys make_server_keys(const ckks::Backend& backend);

// three phase FHE training step

// phase 1 client side has SecretKey
// encrypts H and Y_pred per scalar value broadcast to all slots
// X forwarded as plain encryption noise from fhe_rng so same seed gives same payload
[[nodiscard]] ClientPayload client_encrypt(
    const linalg::Matrix& X,
    const linalg::Matrix& H,
    const linalg::Matrix& Y_pred,
    const ClientKeys& keys,
    std::mt19937_64& fhe_rng);

// phase 2 server side no SecretKey
// computes FA grads homomorphically
// inputs: client payload plus server's CleanServer plus public + eval keys via ServerKeys
// output: ciphertexts only
//
// compute: error = (Y_pred - Y_true) * (1/batch)
//          grad_W2 = H^T * error
//          error_hidden = error * B_FA
// Y_true computed locally from X and teacher B_FA encoded plain
// never decrypts anything
[[nodiscard]] ServerResponse server_compute_gradients(
    const ClientPayload& payload,
    const nn::CleanServer& server,
    const ServerKeys& keys,
    double bimodality_alpha = 0.0);

// phase 3 client side has SecretKey
// decrypts response into plain matrices does not touch CleanClient state
// caller decides what to do with the grads usually one update step
[[nodiscard]] DecryptedGradients client_decrypt(
    const ServerResponse& response,
    const ClientKeys& keys);

// one full FHE training step
//
// runs three phases plus update
// loss computed plain side as MSE between Y_pred and Y_true
[[nodiscard]] StepResult clean_train_step_fhe(
    nn::CleanClient& client,
    const nn::CleanServer& server,
    ckks::Backend& backend,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng,
    double bimodality_alpha = 0.0);

}  // namespace ssns::protocol

#endif  // SSNS_PROTOCOL_TRAINING_HPP
