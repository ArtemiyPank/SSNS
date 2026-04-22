// Compile-time CKKS configuration for the SSNS protocol.
//
// CKKS (Cheon-Kim-Kim-Song) operates over the ring Z_q[X]/(X^N + 1) for a
// power-of-two N.  The coefficient modulus q is a product of small primes
// (RNS / Chinese-Remainder representation), each chosen so a primitive
// 2N-th root of unity exists in F_{q_i} — which makes the negacyclic
// Number-Theoretic Transform applicable.
//
// Parameters here mirror the Python reference (`src/ssns_tenseal.py`):
//     poly_modulus_degree = 8192
//     coeff_mod_bit_sizes = [60, 40, 40, 60]
//     global_scale        = 2^40
//
// Prime selection strategy
// ------------------------
// Every q_i must satisfy:
//   (1) q_i is prime
//   (2) q_i ≡ 1 (mod 2N), i.e. q_i ≡ 1 (mod 16384)  — guarantees a primitive
//       2N-th root of unity exists in F_{q_i}
//   (3) q_i fits in the requested bit-size (60 or 40)
//   (4) all q_i pairwise coprime (automatic for distinct primes)
//
// Search procedure (one-time, executed offline; results baked in below):
//   - Start from (2^60 − 1) and (2^40 − 1).
//   - Descend by steps of 2N = 16384, preserving the ≡ 1 (mod 16384) residue.
//   - For each candidate run deterministic Miller-Rabin with witnesses
//     {2,3,5,7,11,13,17,19,23,29,31,37} (provably correct for n < 2^64).
//   - Take the first two passing primes for each bit-size.
//
// The four chosen primes are validated at startup via `tests/test_modarith.cpp`.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ssns::ckks {

// Polynomial ring degree.  Must be a power of 2.
//
// We use N=4096 (vs Python ref's 8192) because the SSNS protocol broadcasts
// each scalar across all slots — so the slot count (N/2) is irrelevant — and
// halving N halves NTT cost and pointwise op cost.  The existing CKKS primes
// satisfy q_i ≡ 1 mod 16384 ⇒ q_i ≡ 1 mod 8192, so they remain valid.
// Security at N=4096 with the same modulus chain (~200 bits) is ~80-bit
// classical (LWE), acceptable for a research demo; production would bump
// back to N=8192 + slot-packing.
constexpr std::size_t POLY_DEGREE = 4096;

// 2N — used as the order of the primitive root of unity for the negacyclic
// NTT.  Each q_i must satisfy q_i ≡ 1 (mod TWO_N) so such a root exists.
constexpr std::size_t TWO_N = 8192;

// log2(POLY_DEGREE) — used by the radix-2 NTT loop bookkeeping.
constexpr std::size_t LOG_N = 12;

// CKKS encoding scale.  The Python reference uses `global_scale = 2^40`.
constexpr int SCALE_BITS = 40;

// Number of RNS primes in the coefficient modulus chain.  With four primes
// (60+40+40+60 = 200 bits total), CKKS gets ~3 multiplicative levels — enough
// for the SSNS gradient computation as documented in the Python reference.
constexpr std::size_t NUM_PRIMES = 4;

// The four hard-coded primes.  Each was found by descending from
// (2^bits − 1) in steps of 2N = 16384 and Miller-Rabin testing each
// candidate.  Verified by `tests/test_modarith.cpp` to be prime and
// ≡ 1 (mod 16384).
//
// Placement order matches `coeff_mod_bit_sizes`: 60, 40, 40, 60.
constexpr std::array<std::uint64_t, NUM_PRIMES> COEFF_MODULI = {
    // q0 — 60-bit, ≡ 1 mod 16384.
    1152921504606830593ULL,
    // q1 — 40-bit, ≡ 1 mod 16384.
    1099511480321ULL,
    // q2 — 40-bit, ≡ 1 mod 16384 (distinct from q1 for CRT).
    1099510890497ULL,
    // q3 — 60-bit, ≡ 1 mod 16384 (distinct from q0).
    1152921504606748673ULL,
};

// Bit-size of each prime — kept in lockstep with COEFF_MODULI for tests
// that want to assert "this prime fits in N bits".
constexpr std::array<int, NUM_PRIMES> COEFF_MOD_BIT_SIZES = {60, 40, 40, 60};

// Pseudo-Mersenne offsets: q_i = 2^K_i - PSM_C[i] where K_i = bit_size.
// All four primes are 2^K minus a small constant (< 2^21), enabling fast
// modular reduction via mul_mod_psm{40,60} in modarith.hpp.
constexpr std::array<std::uint64_t, NUM_PRIMES> PSM_C = {
    (std::uint64_t{1} << 60) - COEFF_MODULI[0],   // 16383
    (std::uint64_t{1} << 40) - COEFF_MODULI[1],   // 147455
    (std::uint64_t{1} << 40) - COEFF_MODULI[2],   // 737279
    (std::uint64_t{1} << 60) - COEFF_MODULI[3],   // 98303
};

}  // namespace ssns::ckks
