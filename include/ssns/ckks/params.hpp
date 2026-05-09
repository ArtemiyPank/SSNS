// compile time ckks config for ssns
//
// ckks operates over Z_q[X]/(X^N + 1) for power of two N
// q is a product of small primes (rns) each chosen so a primitive 2N-th root of unity exists in F_{q_i}
// makes negacyclic ntt applicable
//
// помнить: q = product of small primes каждый prime это слот в rns
// операции независимо на каждом слоте потом crt восстановление
//
// matches python ref
//     poly_modulus_degree = 8192
//     coeff_mod_bit_sizes = [60, 40, 40, 60]
//     global_scale        = 2^40
//
// prime selection
// every q_i must satisfy
//   1 q_i is prime
//   2 q_i == 1 (mod 2N) i.e q_i == 1 (mod 16384) so 2N-th root exists
//   3 q_i fits in the requested bit size (60 or 40)
//   4 all q_i pairwise coprime (auto for distinct primes)
//
// search done offline
//   start from (2^60 - 1) and (2^40 - 1)
//   step down by 2N = 16384 keeping == 1 (mod 16384) residue
//   miller rabin each candidate
//   take first two passing primes for each bit size
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ssns::ckks {

// polynomial ring degree must be power of 2
//
// using N=4096 (vs python ref 8192) because ssns broadcasts each scalar across all slots so slot count is irrelevant
// halving N halves ntt cost
// existing primes satisfy q_i == 1 mod 16384 which implies q_i == 1 mod 8192 still valid
// security at N=4096 with same modulus chain is around 80 bit classical lwe acceptable for a research demo
// production would bump back to N=8192 + slot packing
constexpr std::size_t POLY_DEGREE = 4096;

// 2N order of primitive root for negacyclic ntt
//
// q_i == 1 mod 16384 чтоб primitive 2N-й корень в F_q существовал
constexpr std::size_t TWO_N = 8192;

// ckks scale python ref uses 2^40
constexpr int SCALE_BITS = 40;

// number of rns primes
// 60+40+40+60 = 200 bits gives ~3 multiplicative levels
constexpr std::size_t NUM_PRIMES = 4;

// hard coded primes
// each found by descending from (2^bits - 1) in steps of 2N = 16384
// verified by tests/test_modarith.cpp to be prime and == 1 (mod 16384)
//
// order matches coeff_mod_bit_sizes 60 40 40 60
constexpr std::array<std::uint64_t, NUM_PRIMES> COEFF_MODULI = {
    // q0 60 bit
    1152921504606830593ULL,
    // q1 40 bit
    1099511480321ULL,
    // q2 40 bit distinct from q1
    1099510890497ULL,
    // q3 60 bit distinct from q0
    1152921504606748673ULL,
};

// bit size of each prime kept in lockstep with COEFF_MODULI
constexpr std::array<int, NUM_PRIMES> COEFF_MOD_BIT_SIZES = {60, 40, 40, 60};

// pseudo mersenne offsets q_i = 2^K_i - PSM_C[i]
// all four primes are 2^K minus a small constant (< 2^21)
// enables fast modular reduction via mul_mod_psm{40,60}
constexpr std::array<std::uint64_t, NUM_PRIMES> PSM_C = {
    (std::uint64_t{1} << 60) - COEFF_MODULI[0],   // 16383
    (std::uint64_t{1} << 40) - COEFF_MODULI[1],   // 147455
    (std::uint64_t{1} << 40) - COEFF_MODULI[2],   // 737279
    (std::uint64_t{1} << 60) - COEFF_MODULI[3],   // 98303
};

}  // namespace ssns::ckks
