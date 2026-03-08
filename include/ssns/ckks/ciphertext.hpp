// CKKS ciphertext — a degree-1 RLWE pair encrypting a plaintext.
//
// Storage convention
// ------------------
// `c0` and `c1` are stored in **NTT form** (per-prime).  Decryption
// computes  m̃ ≈ c0 + c1·s  (mod q), and every CKKS arithmetic op
// (cipher-cipher add/mul, cipher-plain add/mul) operates pointwise in
// the frequency domain — so keeping ciphertexts pre-transformed avoids
// repeating the NTT for every operation.
//
// Tracking fields
// ---------------
//   `scale` — the float-to-integer scaling carried by the message.  Two
//             ciphertexts must agree on `scale` before they can be added;
//             cipher×cipher squares it; rescale (Phase 6.5) divides it
//             by one prime.
//   `level` — number of active RNS primes.  Fresh ciphertexts use all
//             `NUM_PRIMES`.  Each rescale lowers `level` by one; downstream
//             ops should ignore residue slots `[level..NUM_PRIMES)`.  We
//             do NOT enforce truncation inside `Polynomial` — that's a
//             concern for the rescale / arithmetic implementations.
#pragma once

#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>

#include <cstddef>

namespace ssns::ckks {

struct Ciphertext {
    Polynomial c0;       // NTT form.
    Polynomial c1;       // NTT form.
    double scale{0.0};   // Scaling factor of the encrypted message.
    std::size_t level{NUM_PRIMES};  // Active RNS primes.
};

}  // namespace ssns::ckks
