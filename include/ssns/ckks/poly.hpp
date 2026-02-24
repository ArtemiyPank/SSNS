// Polynomial element of Z[X]/(X^N+1) in RNS / Chinese-Remainder form.
//
// Conceptually a polynomial coefficient lives mod Q where Q = ∏ q_i, but
// the CKKS RNS representation stores one residue vector per prime q_i —
// so the in-memory shape is `NUM_PRIMES × N` uint64_t values.  All
// operations (add, sub, multiply via NTT) are performed independently on
// each residue vector, and the result is exactly the polynomial reduced
// mod Q (CRT).
//
// `from_coeffs` lifts a small-integer polynomial into RNS form by
// reducing each integer coefficient mod q_i (for negative values: add
// q_i first to map them into [0, q_i)).  The "small" requirement is
// only that |c_i| < min(q_i) — well satisfied by Phase-5 test inputs.
//
// `multiply` performs polynomial multiplication via the NTT pipeline:
// forward-transform both operands per prime, pointwise multiply mod
// q_i, inverse-transform.  The caller supplies pre-built NTT instances
// (one per prime) so they can be cached across many multiplications —
// the per-prime twiddle tables cost ~64 KB at N=8192.
#pragma once

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace ssns::ckks {

class Polynomial {
public:
    // All-zero polynomial — every residue vector is N zeros.
    Polynomial();

    // Lift a vector of small integer coefficients into RNS form.
    // `coeffs.size()` must be ≤ POLY_DEGREE; missing high coefficients
    // are taken as zero.
    static Polynomial from_coeffs(const std::vector<std::int64_t>& coeffs);

    // In-place element-wise polynomial addition mod q_i, per prime.
    // Both operands must have the canonical NUM_PRIMES × POLY_DEGREE
    // shape (which `Polynomial()` and `from_coeffs` always produce).
    Polynomial& add_inplace(const Polynomial& other);
    Polynomial& sub_inplace(const Polynomial& other);

    // Polynomial multiplication in Z_q[X]/(X^N+1).  Performed per prime
    // via three steps: forward NTT, pointwise multiply, inverse NTT.
    // The caller-supplied `ntts` array is expected to satisfy
    // `ntts[i].prime() == COEFF_MODULI[i]` and `ntts[i].degree() == POLY_DEGREE`.
    static Polynomial multiply(
        const Polynomial& a,
        const Polynomial& b,
        const std::array<NTT, NUM_PRIMES>& ntts);

    // Equality is bitwise on every residue.
    bool operator==(const Polynomial& other) const noexcept;
    bool operator!=(const Polynomial& other) const noexcept { return !(*this == other); }

    // Public storage: one residue vector per prime, each of length
    // POLY_DEGREE.  Exposed directly because tests and Phase-6 ops will
    // need to read / write residues in place.
    std::array<std::vector<std::uint64_t>, NUM_PRIMES> residues;
};

}  // namespace ssns::ckks
