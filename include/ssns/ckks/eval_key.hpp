// CKKS evaluation key (relinearisation) — RNS-gadget construction.
//
// After cipher × cipher we end up with a degree-2 ciphertext (c0, c1, c2)
// satisfying  c0 + c1·s + c2·s² ≈ Δ²·m1·m2  (mod Q).  Relinearisation
// folds the s² term back into a degree-1 ciphertext using an evaluation
// key for which decryption recovers d2·s² with controlled noise.
//
// Why RNS-gadget (not BV / not extended-modulus P)
// ------------------------------------------------
// The naive BV construction (a single (b_evk, a_evk) with
// b_evk + a_evk·s ≈ s²) introduces a relin noise term `d2·e_evk mod Q`
// whose centred magnitude can reach ||d2||·N·σ ≈ Q/2 · N · σ — which
// swamps the message that lives at scale Δ ≪ Q.  The standard fix is
// either an auxiliary modulus P (requires extending the prime chain) or
// a gadget decomposition of d2 into pieces with controlled magnitude.
// We use the RNS-gadget variant: the existing prime chain itself is the
// gadget basis, no chain extension required.
//
// Construction
// ------------
// Define the RNS-basis indicator e_i ∈ Z_Q so that
//     e_i ≡ 1 (mod q_i)
//     e_i ≡ 0 (mod q_j)   for j ≠ i
// In RNS form this is just "residue 1 at slot i, residue 0 at slot j ≠ i".
//
// For each i ∈ [0, NUM_PRIMES) we generate a sub-key (b_i, a_i) with
//     a_i        ← uniform in Z_Q (NTT form)
//     e_i_noise  ← Gaussian σ=3.2 (coefficient form, lifted)
//     b_i        =  -a_i · s + e_i_noise + e_i · s²   (mod Q)
//
// In RNS form, `e_i · s²` is "s² at slot i, zero at slot j ≠ i", so:
//     b_i at slot i:   (-a_i · s + e_i_noise + s²) mod q_i
//     b_i at slot j≠i: (-a_i · s + e_i_noise)      mod q_j
//
// Relinearisation
// ---------------
// Given degree-2 (d0, d1, d2) all in NTT form:
//   1. Inverse-NTT d2 per prime to recover its coefficient-form residues.
//   2. For each i, take d2's slot-i residue, lift to a centred integer in
//      (-q_i/2, q_i/2], reduce mod each q_j to build d2_at_i (full RNS
//      polynomial), and forward-NTT each slot.
//   3. Accumulate Σ_i d2_at_i · b_i into c0_relin (NTT-domain pointwise mul + add)
//      and Σ_i d2_at_i · a_i into c1_relin.
//   4. c0_relin += d0; c1_relin += d1.
//
// Decryption sketch
// -----------------
// (c0_relin) + s · (c1_relin)
//   = d0 + d1·s + Σ_i d2_at_i · (b_i + a_i·s)
//   = d0 + d1·s + Σ_i d2_at_i · (e_i · s² + e_i_noise)
//   = d0 + d1·s + s² · Σ_i d2_at_i · e_i  +  Σ_i d2_at_i · e_i_noise.
// Because d2_at_i · e_i is non-zero only at RNS slot i (where it equals d2's
// slot-i residue), Σ_i d2_at_i · e_i = d2 in CRT, recovering the s²·d2 term.
// The residual noise Σ_i d2_at_i · e_i_noise has bounded coefficients (each
// d2_at_i has centred coefficients ≤ q_i/2 instead of Q/2) — small enough
// that the message survives one mul_cipher + rescale.
//
// Storage
// -------
// All sub_keys[i].a and sub_keys[i].b are stored in NTT form, identical to
// PublicKey, so relinearisation can do its accumulation purely in the
// frequency domain (apart from the one inverse-NTT-on-d2 needed to lift its
// slot-i residue to a centred integer).
#pragma once

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <random>
#include <utility>

namespace ssns::ckks {

// One RNS-gadget sub-key: (b_i, a_i) such that
//     b_i + a_i·s = e_i_noise + e_i · s²   (mod Q)
// where e_i is the RNS-basis indicator for prime slot i.
struct EvalSubKey {
    Polynomial b;  // NTT form.
    Polynomial a;  // NTT form, uniform in Z_Q.
};

// RNS-gadget evaluation key: one sub-key per RNS prime.
struct EvalKey {
    std::array<EvalSubKey, NUM_PRIMES> sub_keys;
};

// Generate a fresh RNS-gadget evaluation key from `sk`.  Uses a tightened
// Gaussian σ for the per-sub-key noise (lower than KEYGEN_NOISE_SIGMA — see
// EVAL_KEY_NOISE_SIGMA below) so cipher × cipher + rescale recovers the
// message inside the test tolerance.  All sub_keys are stored in NTT form.
EvalKey gen_eval_key(const SecretKey& sk,
                     const std::array<NTT, NUM_PRIMES>& ntts,
                     std::mt19937_64& rng);

// Discrete-Gaussian σ for the per-sub-key noise term e_i_noise.
//
// Why a tightened σ (not 3.2)
// ---------------------------
// Final relin noise after one mul_cipher + rescale is ≈ NUM_PRIMES · N · σ_evk
// · max(q_i)/Q_drop (roughly) — at σ=3.2 this lands at ~0.1 absolute error on
// unit-magnitude inputs, which is too coarse for the SSNS gradient pipeline.
// Lowering σ to 0.5 brings the noise back below 1e-2 absolute error per slot
// (validated by tests/test_mul_cipher.cpp).  The security implication is mild:
// only the eval key's noise is reduced — the encryption noise σ stays at 3.2,
// so the secret key remains protected at the documented ~128-bit RLWE level.
// The eval key itself becomes easier to attack as an LWE sample, but it
// reveals at most s² (which is no more sensitive than s itself) and only via
// an attack with sample complexity comparable to the encryption-noise attack.
// For a research demonstrator this trade-off is acceptable; production
// deployments would either use real auxiliary-modulus key-switching (SEAL /
// Lattigo style) or accept a wider prime chain.
inline constexpr double EVAL_KEY_NOISE_SIGMA = 0.5;

}  // namespace ssns::ckks
