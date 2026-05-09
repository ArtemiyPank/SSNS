// ckks rlwe secret key sparse ternary polynomial in Z[X]/(X^N+1)
//
// rlwe assumption secret is small attacker recovers it iff they can solve approx svp on a structured ideal lattice
//
// fixed hamming weight H = 64 (mirrors seal lattigo h-secret-key)
// poly has exactly 64 nonzero coefs each independently +/-1
// more compact than full uniform ternary and gives slightly cheaper noise growth
//
// representation
// s stored in coef form NOT ntt
//   sparse repr is natural in coef form (only 64 of N entries nonzero)
//   pk and evk gen need a*s which converts s to ntt on the fly via Polynomial::multiply
//
// lifting into rns
//   centered +1 -> residue 1 for every q_i
//   centered -1 -> residue q_i - 1
//   centered  0 -> residue 0
//
// all rns slots agree on the support set
#pragma once

#include <ssns/ckks/poly.hpp>

#include <cstddef>
#include <cstdint>
#include <random>

namespace ssns::ckks {

// hamming weight target 64 nonzero positions out of N=8192
inline constexpr std::size_t SECRET_HAMMING_WEIGHT = 64;

struct SecretKey {
    // ternary secret in coef form NOT ntt
    Polynomial s;

    // sample a fresh secret key
    //
    // 1 pick H=64 distinct positions out of N via partial fisher yates
    // 2 for each chosen position draw sign in {-1, +1} uniformly
    // 3 lift into rns per prime
    //
    // determinism output is pure function of rng state at call time
    static SecretKey sample(std::mt19937_64& rng);
};

}  // namespace ssns::ckks
