// negacyclic number theoretic transform for ckks poly mul
//
// ntt is dft over F_p
// role of the complex 2N-th root is played by psi in F_p with psi^(2N) == 1 and psi^N != 1
// such psi exists iff p == 1 (mod 2N) which our primes satisfy
//
// negacyclic means mul in ntt domain == mul in Z[X]/(X^N+1) instead of Z[X]/(X^N-1)
// trick: pre multiply input coefs by psi^i before the forward and post multiply by psi^-i after the inverse
//
// api
//   forward consumes time domain produces frequency in bit reversed order
//   pointwise mul is permutation invariant so this is fine
//   inverse takes bit reversed input and gives natural order back round trip is identity
//   both are in place
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ssns::ckks {
    class NTT {
    public:
        // pre compute twiddle tables for prime p and degree N
        // throws if N is not power of two or 2N does not divide p-1
        NTT(std::uint64_t p, std::size_t N);

        // in place forward
        // a points to N uint64 already reduced mod p
        // output is bit reversed
        void forward(std::uint64_t *a) const;

        // in place inverse
        // takes bit reversed input
        // returns natural order with 1/N scaling
        void inverse(std::uint64_t *a) const;

        std::uint64_t prime() const noexcept { return p_; }
        std::size_t degree() const noexcept { return N_; }

    private:
        std::uint64_t p_;
        std::size_t N_;

        // twiddles are stored bit reversed
        // indexed by bitrev(k, log2_N) at each butterfly stage
        std::vector<std::uint64_t> psi_powers_; // psi^k bit reversed
        std::vector<std::uint64_t> inv_psi_powers_; // psi^-k bit reversed
        std::uint64_t inv_N_; // N^-1 mod p
    };
} // namespace ssns::ckks
