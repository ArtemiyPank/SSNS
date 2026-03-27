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
// FHE training step
// ---------------------------------------------------------------------------
//
// Encoding scheme: each scalar of H / Y_pred is encrypted into its own CKKS
// ciphertext with the value broadcast across all POLY_DEGREE/2 slots.  This
// trades slot packing for simpler matrix-arithmetic (no rotations needed).
//
// Pipeline:
//   1. error = (Y_pred - Y_true) / batch       [cipher - plain] then [scalar]
//      then `rescale` to bring scale back to ~2^40 at level=3.
//   2. H_l3  = rescale(mul_scalar(H, 1.0))     dummy scalar to drop H's level
//      to 3 with the matching scale chain — needed so mul_cipher accepts both.
//   3. grad_W2[h,o]    = sum_i mul_cipher(H_l3[i,h], error[i,o])  then rescale
//      → level 2, scale = (~2^40)^2 / q_drop ≈ 2^20.
//   4. error_hidden[i,h] = sum_o mul_plain(error[i,o], B_FA[o,h])
//      → level 3, scale = (~2^40) * 2^40 ≈ 2^80.
namespace {

using ssns::ckks::Backend;
using ssns::ckks::Ciphertext;
using ssns::ckks::Plaintext;
using ssns::ckks::Polynomial;
using ssns::ckks::POLY_DEGREE;
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
Plaintext encode_scalar(const Backend& backend, double value,
                        double scale, std::size_t level) {
    std::vector<std::complex<double>> slots(SLOT_COUNT, std::complex<double>(value, 0.0));
    Polynomial coeff = backend.encoder.encode(slots, scale);
    return Plaintext::from_polynomial(std::move(coeff), scale, backend.ntts, level);
}

// Encrypt a single scalar broadcast across all slots.  Uses the supplied
// RNG so determinism follows the caller's seeding.  Initial level is the
// full RNS depth (NUM_PRIMES).
Ciphertext encrypt_scalar(const Backend& backend, double value,
                          std::mt19937_64& rng) {
    Plaintext pt = encode_scalar(backend, value, backend.scale, ssns::ckks::NUM_PRIMES);
    return encrypt(pt, backend.pk, backend.ntts, rng);
}

// Decrypt a broadcast-scalar ciphertext and read slot 0's real part.
double decrypt_scalar(const Backend& backend, const Ciphertext& ct) {
    Plaintext pt = decrypt(ct, backend.sk, backend.ntts);
    Polynomial coeff = pt.poly;
    for (std::size_t i = 0; i < pt.level; ++i) {
        backend.ntts[i].inverse(coeff.residues[i].data());
    }
    auto slots = backend.encoder.decode(coeff, pt.scale, pt.level);
    return slots[0].real();
}

}  // namespace

StepResult clean_train_step_fhe(
    nn::CleanClient& client,
    const nn::CleanServer& server,
    ckks::Backend& backend,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng)
{
    // Steps 1-3 — same plaintext ops the plaintext path uses.  X / Y_true /
    // H / Y_pred are returned in plaintext for diagnostic logging; only the
    // wire-bound H and Y_pred get encrypted before crossing to the server.
    linalg::Matrix X      = sample_normal_batch(batch_size, input_dim, rng);
    linalg::Matrix Y_true = server.teacher_forward(X);
    auto fwd              = client.forward(X);
    linalg::Matrix& H     = fwd.H;
    linalg::Matrix& Y_pred = fwd.Y_pred;

    const std::size_t H_cols = H.cols();
    const std::size_t Y_cols = Y_pred.cols();
    const linalg::Matrix& B_FA = server.b_fa();

    // Use a separate RNG seeded from the training-step rng for the CKKS
    // randomness so encryption noise is deterministic across runs that
    // share `rng`.
    std::mt19937_64 fhe_rng(rng.engine()());

    // ---- Encrypt H and Y_pred element-by-element ---------------------
    // Layout: H_ct[i * H_cols + h], Y_pred_ct[i * Y_cols + o].
    std::vector<Ciphertext> H_ct;
    H_ct.reserve(batch_size * H_cols);
    for (std::size_t i = 0; i < batch_size; ++i) {
        for (std::size_t h = 0; h < H_cols; ++h) {
            H_ct.push_back(encrypt_scalar(backend, H(i, h), fhe_rng));
        }
    }
    std::vector<Ciphertext> Y_pred_ct;
    Y_pred_ct.reserve(batch_size * Y_cols);
    for (std::size_t i = 0; i < batch_size; ++i) {
        for (std::size_t o = 0; o < Y_cols; ++o) {
            Y_pred_ct.push_back(encrypt_scalar(backend, Y_pred(i, o), fhe_rng));
        }
    }

    // ---- Server: error = (Y_pred − Y_true) * (1/batch) ----------------
    // sub_plain expects matching (scale, level): encode each Y_true scalar
    // at (backend.scale, NUM_PRIMES).  After mul_scalar + rescale, error
    // sits at level 3, scale = backend.scale * 2^60 / q3 (≈ 2^40 numerically,
    // not exactly).  Cache this `s3` for the H-level-down step below.
    const double inv_batch = 1.0 / static_cast<double>(batch_size);
    std::vector<Ciphertext> error_ct;
    error_ct.reserve(batch_size * Y_cols);
    double s3 = 0.0;
    for (std::size_t i = 0; i < batch_size; ++i) {
        for (std::size_t o = 0; o < Y_cols; ++o) {
            Plaintext yt_pt = encode_scalar(backend, Y_true(i, o),
                                            backend.scale, ssns::ckks::NUM_PRIMES);
            Ciphertext diff = sub_plain(Y_pred_ct[i * Y_cols + o], yt_pt);
            Ciphertext scaled = mul_scalar(diff, inv_batch);
            Ciphertext err = rescale(scaled, backend.ntts);
            s3 = err.scale;
            error_ct.push_back(std::move(err));
        }
    }

    // ---- Bring H to level 3 via dummy mul_scalar(1.0) + rescale -----
    // Same chain as `error`, so scales match exactly (mul_cipher tolerates
    // ~1e-6 relative drift; identical chain is well within tolerance).
    std::vector<Ciphertext> H_ct_l3;
    H_ct_l3.reserve(batch_size * H_cols);
    for (auto& ct : H_ct) {
        Ciphertext bumped = mul_scalar(ct, 1.0);
        H_ct_l3.push_back(rescale(bumped, backend.ntts));
    }

    // ---- grad_W2[h,o] = Σ_i H[i,h] * error[i,o] ---------------------
    // Each mul_cipher pushes the scale to s3*s3, level=3.  Sum first, then
    // rescale once → level=2, scale = s3*s3 / q2 ≈ 2^40.  Decoding at that
    // scale recovers the gradient.
    linalg::Matrix grad_W2(H_cols, Y_cols);
    for (std::size_t h = 0; h < H_cols; ++h) {
        for (std::size_t o = 0; o < Y_cols; ++o) {
            Ciphertext acc = mul_cipher(H_ct_l3[0 * H_cols + h],
                                        error_ct[0 * Y_cols + o],
                                        backend.evk, backend.ntts);
            for (std::size_t i = 1; i < batch_size; ++i) {
                Ciphertext term = mul_cipher(H_ct_l3[i * H_cols + h],
                                             error_ct[i * Y_cols + o],
                                             backend.evk, backend.ntts);
                acc = add(acc, term);
            }
            Ciphertext rs = rescale(acc, backend.ntts);
            grad_W2(h, o) = decrypt_scalar(backend, rs);
        }
    }

    // ---- error_hidden[i,h] = Σ_o error[i,o] * B_FA[o,h] -------------
    // mul_plain needs ct.level == pt.level; encode B_FA at (level=3,
    // scale=backend.scale).  Scales need not match; mul_plain just
    // multiplies them (s3 * backend.scale).  Sum is closed under add
    // because every term shares the same scale chain.
    linalg::Matrix error_hidden(batch_size, H_cols);
    // Pre-encode B_FA plaintext column-by-column (same plaintext is reused
    // across all batch rows).
    std::vector<Plaintext> bfa_pt;
    bfa_pt.reserve(Y_cols * H_cols);
    for (std::size_t o = 0; o < Y_cols; ++o) {
        for (std::size_t h = 0; h < H_cols; ++h) {
            bfa_pt.push_back(encode_scalar(backend, B_FA(o, h),
                                           backend.scale, /*level=*/3));
        }
    }
    for (std::size_t i = 0; i < batch_size; ++i) {
        for (std::size_t h = 0; h < H_cols; ++h) {
            Ciphertext acc = mul_plain(error_ct[i * Y_cols + 0],
                                       bfa_pt[0 * H_cols + h]);
            for (std::size_t o = 1; o < Y_cols; ++o) {
                Ciphertext term = mul_plain(error_ct[i * Y_cols + o],
                                            bfa_pt[o * H_cols + h]);
                acc = add(acc, term);
            }
            error_hidden(i, h) = decrypt_scalar(backend, acc);
        }
    }

    const double loss = mse_loss(Y_pred, Y_true);
    client.update(grad_W2, error_hidden);

    // Silence unused-variable warning when the trace is disabled.
    (void)s3;

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
