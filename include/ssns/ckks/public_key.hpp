// CKKS RLWE public key: pk = (b, a) with b = -a*s + e (mod q).
//
// Encryption: ct = (c0, c1) = (b·u + e0 + Δm, a·u + e1) for fresh small
// (u, e0, e1).  Decryption returns c0 + c1·s = Δm + (small noise).
//
// Storage convention
// ------------------
// Both `a` and `b` are stored in **NTT form** (per-prime).  Rationale:
//   - Encryption multiplies a · u and b · u; doing those products in NTT
//     domain saves two forward transforms per encrypt.
//   - The stored polynomials never need a coefficient-form view at
//     runtime — only the keygen-time arithmetic is in coefficient form,
//     and that conversion happens internally to gen_public_key.
//
// SecretKey.s, by contrast, stays in coefficient form (sparse ternary —
// see secret_key.hpp).
#pragma once

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <random>

namespace ssns::ckks {

// Discrete-Gaussian σ for the noise polynomial e in the RLWE relation.
// Mirrors SEAL / Lattigo defaults; supplies ~128-bit security at N=8192.
inline constexpr double KEYGEN_NOISE_SIGMA = 3.2;

struct PublicKey {
    Polynomial b;  // = -a*s + e (mod q), NTT form.
    Polynomial a;  // uniform random in Z_q, NTT form.
};

// Generate a fresh CKKS public key from `sk`.  Uses `ntts` for the
// coefficient-↔-NTT conversions during the a*s computation, and `rng` to
// sample `a` (uniform per prime) and `e` (rounded Gaussian σ=3.2).
//
// The output `pk.a` and `pk.b` are stored in NTT form.
PublicKey gen_public_key(const SecretKey& sk,
                         const std::array<NTT, NUM_PRIMES>& ntts,
                         std::mt19937_64& rng);

}  // namespace ssns::ckks
