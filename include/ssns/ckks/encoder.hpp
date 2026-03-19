// CKKS canonical embedding: complex<double> slot vector ↔ Polynomial.
//
// CKKS encodes a vector of N/2 complex numbers as a polynomial in
// Z[X]/(X^N + 1) by evaluating the polynomial at the primitive 2N-th
// roots of unity ζ^(2k+1), k = 0..N-1, where ζ = exp(πi/N).  Because we
// require the polynomial to have real coefficients, the N evaluations come
// in N/2 conjugate pairs, leaving N/2 complex degrees of freedom — these
// are the "slots".
//
// The encode pipeline:
//   1. Mirror length-(N/2) complex slots to a length-N complex vector
//      satisfying z_full[N-1-k] = conj(z_full[k]).
//   2. Apply the inverse special-FFT (twist by ζ^(-k), N-point IFFT,
//      twist by ζ^(-k)) to recover real coefficients m_j.
//   3. Multiply by `scale`, round to nearest integer, lift into RNS form
//      (one residue per CKKS prime).
//
// Decode is the matched inverse: lift residues to signed integers via
// CRT (centered representation), divide by scale, apply the special FFT
// to recover slot values, take the first N/2 entries.
//
// FFT is implemented from scratch (Cooley-Tukey radix-2 on N points) — we
// rely only on std::complex<double>.  Precision: N=8192, scale=2^40 leaves
// O(N log N · 2^-52) relative round-off ≪ 1e-10 in the encoded slots.
#pragma once

#include <ssns/ckks/poly.hpp>

#include <complex>
#include <cstddef>
#include <vector>

namespace ssns::ckks {

class Encoder {
public:
    // Build the encoder.  Pre-computes the FFT twiddle tables and the
    // ζ^k twist tables (ζ = primitive 2N-th root of unity in C).
    Encoder();

    // Encode a slot vector of length POLY_DEGREE/2 into a polynomial.
    // Throws std::invalid_argument if `z.size() != POLY_DEGREE/2`.
    // `scale` controls the precision: bigger scale → smaller relative
    // round-off but eats into the modulus budget faster.
    Polynomial encode(const std::vector<std::complex<double>>& z, double scale) const;

    // Decode a polynomial back into a slot vector of length POLY_DEGREE/2.
    // Lifts each RNS coefficient to a signed integer (centered around 0)
    // using Garner-style CRT, divides by scale, applies the special FFT.
    //
    // `level` controls how many RNS primes participate in the CRT lift.
    // Defaults to NUM_PRIMES so callers using fresh ciphertexts / plaintexts
    // see the existing API.  After `rescale` (Phase 6.4) the active level
    // shrinks; pass the smaller `level` so the dropped residues — which are
    // zeroed out by rescale — don't perturb the lift.  Must satisfy
    // `1 <= level <= NUM_PRIMES`.
    std::vector<std::complex<double>> decode(const Polynomial& p,
                                             double scale,
                                             std::size_t level = NUM_PRIMES) const;

    static constexpr std::size_t slot_count() { return POLY_DEGREE / 2; }

private:
    // Standard radix-2 Cooley-Tukey on a length-N vector.  In-place.
    // `inverse=true` divides by N at the end.
    void fft(std::vector<std::complex<double>>& a, bool inverse) const;

    // Bit-reversal permutation used by both fft directions.
    void bitreverse_permute(std::vector<std::complex<double>>& a) const;

    // Pre-computed twist factors for the special FFT.
    // zeta_pow_[k] = ζ^k where ζ = exp(πi/N), used in encode (decode is
    // the conjugate).  Length N.
    std::vector<std::complex<double>> zeta_pow_;
    std::vector<std::complex<double>> zeta_pow_conj_;

    // Pre-computed FFT twiddles (forward direction).  twiddle_[s+j] =
    // exp(-2πi · j / (2s)) for the size-2s butterfly stage.  We index
    // by stage during the loop.
    std::vector<std::complex<double>> twiddle_fwd_;
    std::vector<std::complex<double>> twiddle_inv_;
};

}  // namespace ssns::ckks
