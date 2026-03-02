// CKKS RLWE secret key — a sparse ternary polynomial in Z[X]/(X^N+1).
//
// We use a fixed Hamming weight H = 64 (mirrors the SEAL / Lattigo
// "h-secret-key" convention): the polynomial has exactly 64 nonzero
// coefficients, each independently ±1.  This is more compact than a full
// uniform-ternary secret and gives marginally cheaper noise growth.
//
// Representation
// --------------
// `s` is stored in **coefficient form** (NOT NTT form).  Rationale:
//   - Sparse representation is natural in coefficient form (only 64 of
//     N=8192 entries are nonzero).
//   - PublicKey / EvalKey generation needs to multiply a · s; that path
//     converts s to NTT form on the fly via Polynomial::multiply, so the
//     stored form does not need to be NTT.
//
// Lifting into RNS:
//   centered value +1  →  residue 1   for every prime q_i
//   centered value -1  →  residue q_i - 1
//   centered value 0   →  residue 0
//
// All four RNS slots therefore agree on the support set (the index set of
// nonzero coefficients), which the tests rely on.
#pragma once

#include <ssns/ckks/poly.hpp>

#include <cstddef>
#include <cstdint>
#include <random>

namespace ssns::ckks {

// Hamming-weight target for the sparse ternary secret.  64 nonzero
// positions out of N=8192.
inline constexpr std::size_t SECRET_HAMMING_WEIGHT = 64;

struct SecretKey {
    // Ternary secret in **coefficient form** (NOT NTT form).
    Polynomial s;

    // Sample a fresh secret key using the supplied RNG.
    //
    // Algorithm:
    //   1. Pick H=64 distinct positions out of N=8192 via partial
    //      Fisher-Yates on an index array.
    //   2. For each chosen position, draw a sign ∈ {-1, +1} uniformly.
    //   3. Lift into RNS form per CKKS prime.
    //
    // Determinism: the output is a pure function of the RNG state at
    // call time.
    static SecretKey sample(std::mt19937_64& rng);
};

}  // namespace ssns::ckks
