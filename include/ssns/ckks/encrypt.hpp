// CKKS RLWE encryption — public-key form.
//
// Given a plaintext `pt` (NTT form, scale, level) and a public key
// `pk = (b, a)` satisfying b ≈ -a·s + e (mod q), encrypt produces
// ciphertext (c0, c1) with
//
//     c0 =  b·v + e0 + pt    (mod q)
//     c1 =  a·v + e1         (mod q)
//
// where:
//     v  ← sparse ternary in {-1, 0, +1}, density 1/2 (each ±1 w.p. 1/4)
//     e0, e1 ← rounded discrete Gaussian σ = KEYGEN_NOISE_SIGMA = 3.2
//
// Decryption recovers
//     c0 + c1·s = (b + a·s)·v + e0 + e1·s + pt  =  pt + small noise.
//
// Storage convention follows PublicKey: both `c0` and `c1` are stored
// in NTT form so cipher arithmetic operates pointwise without any
// further transforms.
//
// Determinism: output depends only on the RNG state at call time
// (same seed → same ciphertext) — useful for round-trip tests.
#pragma once

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/public_key.hpp>

#include <array>
#include <random>

namespace ssns::ckks {

Ciphertext encrypt(const Plaintext& pt,
                   const PublicKey& pk,
                   const std::array<NTT, NUM_PRIMES>& ntts,
                   std::mt19937_64& rng);

}  // namespace ssns::ckks
