// CKKS plaintext — encoded message ready to be added to / multiplied by
// ciphertexts.
//
// Storage convention
// ------------------
// `poly` is stored in **NTT form** (per-prime).  Rationale: cipher-plain
// add/mul both operate elementwise in the frequency domain, so keeping
// the plaintext pre-transformed avoids an NTT per arithmetic step.
//
// Encoder currently produces a coefficient-form polynomial — the helper
// `Plaintext::from_polynomial` converts in place via the supplied NTT
// instances.  We deliberately do not change `Encoder::encode`: keeping
// the encoder pure (no NTT dependency in its public signature) means
// callers that just want a polynomial form (tests, debug dumps) still
// get one.
//
// Tracking fields
// ---------------
//   `scale` — the float-to-integer scaling that was used when encoding;
//             needed by rescale (Phase 6.5) so the ciphertext can match
//             the plaintext's scale.
//   `level` — number of active RNS primes for this plaintext.  Fresh
//             plaintexts use all `NUM_PRIMES`; rescale will drop one at
//             a time.  We do not enforce truncation in `Polynomial` —
//             higher-numbered residues are simply ignored by ops once
//             `level` shrinks.
#pragma once

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>

#include <array>
#include <cstddef>

namespace ssns::ckks {

struct Plaintext {
    Polynomial poly;     // NTT form.
    double scale;        // Encoder scale used to produce `poly`.
    std::size_t level;   // Active RNS primes; NUM_PRIMES on fresh plaintexts.

    // Wrap a coefficient-form polynomial (e.g. straight from
    // `Encoder::encode`) into a Plaintext: applies forward NTT per prime,
    // captures the supplied scale, and tags level (defaulting to the
    // full RNS depth NUM_PRIMES).
    static Plaintext from_polynomial(Polynomial coeff_form,
                                     double scale,
                                     const std::array<NTT, NUM_PRIMES>& ntts,
                                     std::size_t level = NUM_PRIMES);
};

}  // namespace ssns::ckks
