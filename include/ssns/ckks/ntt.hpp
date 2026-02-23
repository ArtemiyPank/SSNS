// Negacyclic Number-Theoretic Transform for CKKS polynomial multiplication.
//
// The NTT is the discrete-Fourier-transform analogue over a finite field
// F_p, where the role of the complex 2N-th root of unity is played by
// ψ ∈ F_p satisfying ψ^(2N) ≡ 1 and ψ^N ≢ 1.  Such ψ exists iff p ≡ 1
// (mod 2N), which the four CKKS primes in `params.hpp` satisfy by
// construction.
//
// "Negacyclic" — multiplication in the NTT domain corresponds to
// multiplication in Z[X]/(X^N+1) rather than Z[X]/(X^N-1).  The standard
// trick is to pre-multiply input coefficients by ψ^i (i = 0..N-1) before
// the forward transform, and post-multiply by ψ^-i after the inverse.
// Concretely: define â_k = NTT(ψ^i · a_i), and the inverse multiplies by
// ψ^-i and divides by N.  See e.g. Longa-Naehrig "Speeding up the
// Number Theoretic Transform" or Nussbaumer's textbook.
//
// API contract:
//   - NTT::forward(a) consumes the time-domain coefficient array and
//     produces the frequency-domain values *in bit-reversed order*.
//     This is fine because pointwise multiplication is permutation-
//     invariant, and NTT::inverse is the matched bit-reversed-input
//     inverse — round-trip is identity.
//   - Both transforms are in-place; caller owns the buffer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ssns::ckks {

class NTT {
public:
    // Pre-compute twiddle tables for prime `p` and degree `N`.  Throws
    // `std::invalid_argument` if N is not a power of two or 2N does not
    // divide p-1 (i.e. no primitive 2N-th root of unity exists in F_p).
    NTT(std::uint64_t p, std::size_t N);

    // In-place forward NTT.  `a` must point to N uint64_t values, each
    // already reduced mod p.  Output is in bit-reversed index order.
    void forward(std::uint64_t* a) const;

    // In-place inverse NTT.  Consumes input in bit-reversed order (the
    // shape produced by `forward`), returns the original time-domain
    // coefficients in natural order — including the 1/N scaling.
    void inverse(std::uint64_t* a) const;

    std::uint64_t prime() const noexcept { return p_; }
    std::size_t   degree() const noexcept { return N_; }

private:
    std::uint64_t p_;
    std::size_t   N_;

    // Forward butterflies use ψ^k in bit-reversed order at the
    // appropriate stage; inverse butterflies use ψ^-k similarly.  We
    // store them flat (length N) and index by `bitrev(k, log2_N)` at
    // each level, exactly as in Longa-Naehrig.
    std::vector<std::uint64_t> psi_powers_;       // ψ^k, bit-reversed
    std::vector<std::uint64_t> inv_psi_powers_;   // ψ^-k, bit-reversed
    std::uint64_t inv_N_;                         // N^-1 mod p
};

}  // namespace ssns::ckks
