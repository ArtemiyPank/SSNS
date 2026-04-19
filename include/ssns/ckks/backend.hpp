// CKKS backend bundle — packages everything needed for end-to-end FHE
// (NTT cache, encoder, secret/public/eval keys, scale) into a single object
// so the protocol layer can pass one parameter instead of five.
//
// Determinism: `Backend::create(seed)` builds a single `std::mt19937_64`
// from the provided seed and consumes it sequentially for SecretKey,
// PublicKey, and EvalKey generation.  Same seed → bit-identical bundle,
// which the SSNS protocol relies on for reproducible FHE training runs.
//
// The NTT cache is shared across all CKKS ops in the bundle (one per prime
// in COEFF_MODULI).  The encoder is stateless apart from its FFT twiddle
// tables, so a single instance is reused across the entire pipeline.
//
// Default scale is 2^SCALE_BITS (the CKKS global scale used throughout the
// project — see params.hpp).
#pragma once

#include <ssns/ckks/encoder.hpp>
#include <ssns/ckks/eval_key.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/public_key.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <cstdint>

namespace ssns::ckks {

struct Backend {
    std::array<NTT, NUM_PRIMES> ntts;
    Encoder    encoder;
    SecretKey  sk;
    PublicKey  pk;
    EvalKey    evk;
    double     scale;

    // Pre-computed NTT-form of sk.s — populated once at Backend::create.
    // The decrypt hot path needs sk in NTT form for pointwise mul; doing
    // 4 forward NTTs per decrypt call wastes ~20 % of FHE training wall
    // time at preset config.  Cached here, used via `decrypt_cached`.
    Polynomial s_ntt;

    // Deterministic factory: same seed → identical (sk, pk, evk).
    // Uses a single mt19937_64 stream so the order of key sampling is
    // fixed (sk, pk, evk).  Default scale is 2^SCALE_BITS.
    static Backend create(std::uint64_t seed);
};

}  // namespace ssns::ckks
