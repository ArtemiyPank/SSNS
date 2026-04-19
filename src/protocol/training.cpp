#include <ssns/protocol/training.hpp>

#include <complex>
#include <random>
#include <utility>
#include <vector>

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/decrypt.hpp>
#include <ssns/ckks/encrypt.hpp>
#include <ssns/ckks/linear_ops.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/poly.hpp>

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

// ---------------------------------------------------------------------------
// FHE training step — three-phase implementation
// ---------------------------------------------------------------------------
//
// Encoding scheme: each scalar of H / Y_pred is encrypted into its own CKKS
// ciphertext with the value broadcast across all POLY_DEGREE/2 slots.  This
// trades slot packing for simpler matrix-arithmetic (no rotations needed).
//
// Pipeline (all server-side ops on ciphertexts):
//   1. error = (Y_pred - Y_true) / batch       [cipher - plain] then [scalar]
//      then `rescale` to bring scale back to ~2^40 at level=3.
//   2. H_l3  = rescale(mul_scalar(H, 1.0))     dummy scalar to drop H's level
//      to 3 with the matching scale chain — needed so mul_cipher accepts both.
//   3. grad_W2[h,o]    = sum_i mul_cipher(H_l3[i,h], error[i,o])  then rescale
//      → level 2, scale = (~2^40)^2 / q_drop ≈ 2^20.
//   4. error_hidden[i,h] = sum_o mul_plain(error[i,o], B_FA[o,h])
//      → level 3, scale = (~2^40) * 2^40 ≈ 2^80.
namespace {

using ssns::ckks::Ciphertext;
using ssns::ckks::Encoder;
using ssns::ckks::NTT;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::Plaintext;
using ssns::ckks::Polynomial;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::PublicKey;
using ssns::ckks::SecretKey;
using ssns::ckks::add;
using ssns::ckks::decrypt;
using ssns::ckks::encrypt;
using ssns::ckks::mul_cipher;
using ssns::ckks::mul_plain;
using ssns::ckks::mul_scalar;
using ssns::ckks::rescale;
using ssns::ckks::sub_plain;

constexpr std::size_t SLOT_COUNT = POLY_DEGREE / 2;

// Encode a single scalar into a Plaintext with the value broadcast across
// every slot.  Same plaintext can be reused for many ciphertext-plaintext
// ops sharing the same `scale` and `level`.
Plaintext encode_scalar(const Encoder& encoder,
                        const std::array<NTT, NUM_PRIMES>& ntts,
                        double value, double scale, std::size_t level) {
    std::vector<std::complex<double>> slots(SLOT_COUNT, std::complex<double>(value, 0.0));
    Polynomial coeff = encoder.encode(slots, scale);
    return Plaintext::from_polynomial(std::move(coeff), scale, ntts, level);
}

// Encrypt a single scalar broadcast across all slots.  Uses the supplied
// RNG so determinism follows the caller's seeding.  Initial level is the
// full RNS depth (NUM_PRIMES).
Ciphertext encrypt_scalar(const Encoder& encoder,
                          const std::array<NTT, NUM_PRIMES>& ntts,
                          const PublicKey& pk,
                          double value, double scale,
                          std::mt19937_64& rng) {
    Plaintext pt = encode_scalar(encoder, ntts, value, scale, NUM_PRIMES);
    return encrypt(pt, pk, ntts, rng);
}

// Decrypt a broadcast-scalar ciphertext and read slot 0's real part.
// Uses pre-computed NTT-form sk to skip 4 forward NTTs per call —
// crucial for the hot decrypt loop at preset config (153k decrypts).
double decrypt_scalar(const Encoder& encoder,
                      const std::array<NTT, NUM_PRIMES>& ntts,
                      const Polynomial& s_ntt,
                      const Ciphertext& ct) {
    Plaintext pt = decrypt_with_ntt_sk(ct, s_ntt);
    Polynomial coeff = pt.poly;
    for (std::size_t i = 0; i < pt.level; ++i) {
        ntts[i].inverse(coeff.residues[i].data());
    }
    auto slots = encoder.decode(coeff, pt.scale, pt.level);
    return slots[0].real();
}

}  // namespace

// ---------------------------------------------------------------------------
// Backend → ClientKeys / ServerKeys views
// ---------------------------------------------------------------------------

ClientKeys make_client_keys(const ckks::Backend& backend) {
    return ClientKeys{
        /*sk=*/      &backend.sk,
        /*pk=*/      &backend.pk,
        /*ntts=*/    &backend.ntts,
        /*encoder=*/ &backend.encoder,
        /*s_ntt=*/   &backend.s_ntt,
        /*scale=*/   backend.scale,
    };
}

ServerKeys make_server_keys(const ckks::Backend& backend) {
    // NOTE: SecretKey is intentionally absent from ServerKeys.  This is
    // the load-bearing invariant of the wire protocol — server code can
    // only consume ciphertexts, never plaintext.
    return ServerKeys{
        /*pk=*/      &backend.pk,
        /*evk=*/     &backend.evk,
        /*ntts=*/    &backend.ntts,
        /*encoder=*/ &backend.encoder,
        /*scale=*/   backend.scale,
    };
}

// ---------------------------------------------------------------------------
// Phase 1: client-side encryption
// ---------------------------------------------------------------------------

ClientPayload client_encrypt(
    const linalg::Matrix& X,
    const linalg::Matrix& H,
    const linalg::Matrix& Y_pred,
    const ClientKeys& keys,
    std::mt19937_64& fhe_rng)
{
    const std::size_t batch_size = H.rows();
    const std::size_t H_cols     = H.cols();
    const std::size_t Y_cols     = Y_pred.cols();

    // Pre-derive per-task RNG seeds from the caller's fhe_rng so the
    // encryption noise is deterministic regardless of thread schedule.
    // Each scalar's encryption gets its own mt19937_64 instance.
    const std::size_t H_count = batch_size * H_cols;
    const std::size_t Y_count = batch_size * Y_cols;
    std::vector<std::uint64_t> H_seeds(H_count);
    std::vector<std::uint64_t> Y_seeds(Y_count);
    for (auto& s : H_seeds) s = fhe_rng();
    for (auto& s : Y_seeds) s = fhe_rng();

    std::vector<Ciphertext> H_ct(H_count);
    std::vector<Ciphertext> Y_pred_ct(Y_count);

    #pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < H_count; ++k) {
        const std::size_t i = k / H_cols;
        const std::size_t h = k % H_cols;
        std::mt19937_64 rk(H_seeds[k]);
        H_ct[k] = encrypt_scalar(*keys.encoder, *keys.ntts, *keys.pk,
                                  H(i, h), keys.scale, rk);
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < Y_count; ++k) {
        const std::size_t i = k / Y_cols;
        const std::size_t o = k % Y_cols;
        std::mt19937_64 rk(Y_seeds[k]);
        Y_pred_ct[k] = encrypt_scalar(*keys.encoder, *keys.ntts, *keys.pk,
                                       Y_pred(i, o), keys.scale, rk);
    }

    return ClientPayload{
        /*X=*/         X,
        /*H_ct=*/      std::move(H_ct),
        /*Y_pred_ct=*/ std::move(Y_pred_ct),
        /*H_cols=*/    H_cols,
        /*Y_cols=*/    Y_cols,
    };
}

// ---------------------------------------------------------------------------
// Phase 2: server-side homomorphic compute (NO SecretKey access)
// ---------------------------------------------------------------------------

ServerResponse server_compute_gradients(
    const ClientPayload& payload,
    const nn::CleanServer& server,
    const ServerKeys& keys)
{
    const std::size_t batch_size = payload.X.rows();
    const std::size_t H_cols     = payload.H_cols;
    const std::size_t Y_cols     = payload.Y_cols;

    // Server computes Y_true locally — Teacher weights never leave the server.
    linalg::Matrix Y_true = server.teacher_forward(payload.X);
    const linalg::Matrix& B_FA = server.b_fa();

    // ---- error = (Y_pred − Y_true) * (1/batch) -------------------------
    const double inv_batch = 1.0 / static_cast<double>(batch_size);
    const std::size_t E_count = batch_size * Y_cols;
    std::vector<Ciphertext> error_ct(E_count);
    #pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < E_count; ++k) {
        const std::size_t i = k / Y_cols;
        const std::size_t o = k % Y_cols;
        Plaintext yt_pt = encode_scalar(*keys.encoder, *keys.ntts,
                                         Y_true(i, o), keys.scale, NUM_PRIMES);
        Ciphertext diff   = sub_plain(payload.Y_pred_ct[k], yt_pt);
        Ciphertext scaled = mul_scalar(diff, inv_batch);
        error_ct[k]       = rescale(scaled, *keys.ntts);
    }

    // ---- Bring H to level 3 (matching error's scale chain) -------------
    const std::size_t H_count = payload.H_ct.size();
    std::vector<Ciphertext> H_l3(H_count);
    #pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < H_count; ++k) {
        Ciphertext bumped = mul_scalar(payload.H_ct[k], 1.0);
        H_l3[k] = rescale(bumped, *keys.ntts);
    }

    // ---- grad_W2[h,o] = Σ_i H[i,h] * error[i,o] (cipher × cipher) -----
    // H_cols × Y_cols independent reductions — perfect for thread fanout.
    // mul_cipher / rescale read shared evk/ntts but never mutate them and
    // allocate their outputs on the stack/heap, so no locking is needed.
    std::vector<Ciphertext> grad_W2_ct(H_cols * Y_cols);
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t h = 0; h < H_cols; ++h) {
        for (std::size_t o = 0; o < Y_cols; ++o) {
            Ciphertext acc = mul_cipher(H_l3[0 * H_cols + h],
                                         error_ct[0 * Y_cols + o],
                                         *keys.evk, *keys.ntts);
            for (std::size_t i = 1; i < batch_size; ++i) {
                Ciphertext term = mul_cipher(H_l3[i * H_cols + h],
                                              error_ct[i * Y_cols + o],
                                              *keys.evk, *keys.ntts);
                acc = add(acc, term);
            }
            grad_W2_ct[h * Y_cols + o] = rescale(acc, *keys.ntts);
        }
    }

    // ---- error_hidden[i,h] = Σ_o error[i,o] * B_FA[o,h] (cipher × plain)
    std::vector<Plaintext> bfa_pt;
    bfa_pt.reserve(Y_cols * H_cols);
    for (std::size_t o = 0; o < Y_cols; ++o) {
        for (std::size_t h = 0; h < H_cols; ++h) {
            bfa_pt.push_back(encode_scalar(*keys.encoder, *keys.ntts,
                                           B_FA(o, h), keys.scale, /*level=*/3));
        }
    }
    std::vector<Ciphertext> error_hidden_ct(batch_size * H_cols);
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t i = 0; i < batch_size; ++i) {
        for (std::size_t h = 0; h < H_cols; ++h) {
            Ciphertext acc = mul_plain(error_ct[i * Y_cols + 0],
                                        bfa_pt[0 * H_cols + h]);
            for (std::size_t o = 1; o < Y_cols; ++o) {
                Ciphertext term = mul_plain(error_ct[i * Y_cols + o],
                                             bfa_pt[o * H_cols + h]);
                acc = add(acc, term);
            }
            error_hidden_ct[i * H_cols + h] = std::move(acc);
        }
    }

    return ServerResponse{
        /*grad_W2_ct=*/      std::move(grad_W2_ct),
        /*error_hidden_ct=*/ std::move(error_hidden_ct),
        /*H_cols=*/          H_cols,
        /*Y_cols=*/          Y_cols,
        /*batch_size=*/      batch_size,
    };
}

// ---------------------------------------------------------------------------
// Phase 3: client-side decryption
// ---------------------------------------------------------------------------

DecryptedGradients client_decrypt(
    const ServerResponse& response,
    const ClientKeys& keys)
{
    DecryptedGradients out{
        linalg::Matrix(response.H_cols, response.Y_cols),
        linalg::Matrix(response.batch_size, response.H_cols),
    };
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t h = 0; h < response.H_cols; ++h) {
        for (std::size_t o = 0; o < response.Y_cols; ++o) {
            out.grad_W2(h, o) = decrypt_scalar(*keys.encoder, *keys.ntts, *keys.s_ntt,
                                                response.grad_W2_ct[h * response.Y_cols + o]);
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t i = 0; i < response.batch_size; ++i) {
        for (std::size_t h = 0; h < response.H_cols; ++h) {
            out.error_hidden(i, h) = decrypt_scalar(*keys.encoder, *keys.ntts, *keys.s_ntt,
                                                     response.error_hidden_ct[i * response.H_cols + h]);
        }
    }
    return out;
}

StepResult clean_train_step_fhe(
    nn::CleanClient& client,
    const nn::CleanServer& server,
    ckks::Backend& backend,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng)
{
    // Sample X, run forward — all client-side, plaintext.
    linalg::Matrix X       = sample_normal_batch(batch_size, input_dim, rng);
    auto fwd               = client.forward(X);
    linalg::Matrix& H      = fwd.H;
    linalg::Matrix& Y_pred = fwd.Y_pred;

    // Pre-compute Y_true for the plaintext loss return value.  In a real
    // deployment this happens server-side; computing it here for the
    // client's MSE log is just an observability concession (it adds no
    // information that wouldn't be in the gradients anyway).
    linalg::Matrix Y_true = server.teacher_forward(X);
    const double loss = mse_loss(Y_pred, Y_true);

    // Phase 1 — client encrypts.
    std::mt19937_64 fhe_rng(rng.engine()());
    auto client_keys = make_client_keys(backend);
    ClientPayload payload = client_encrypt(X, H, Y_pred, client_keys, fhe_rng);

    // Phase 2 — server compute.  ServerKeys has no SecretKey by construction.
    auto server_keys = make_server_keys(backend);
    ServerResponse response = server_compute_gradients(payload, server, server_keys);

    // Phase 3 — client decrypt + Adam update.
    DecryptedGradients grads = client_decrypt(response, client_keys);
    client.update(grads.grad_W2, grads.error_hidden);

    return StepResult{
        std::move(X),
        std::move(Y_true),
        std::move(H),
        std::move(Y_pred),
        std::move(grads.grad_W2),
        std::move(grads.error_hidden),
        loss,
    };
}

}  // namespace ssns::protocol
