// Implementation of the negacyclic NTT.  Algorithm references:
//
//   Longa & Naehrig, "Speeding up the Number Theoretic Transform for Faster
//   Ideal Lattice-Based Cryptography" (2016), Algorithms 1 (forward, CT)
//   and 2 (inverse, GS).  These pre-bake ψ^bitrev(k) into the twiddle
//   table so the butterfly does not need to track bit-reversed indices
//   explicitly.
//
// Forward (Cooley-Tukey, decimation-in-frequency, no bit-reversal at the
// end — output ends up in bit-reversed order which is fine for our
// pointwise-multiplication-then-inverse usage):
//
//   for s in N/2, N/4, ..., 1:           (m = N/(2s) doubles each round)
//       for j in 0..m:
//           ψ_j = psi_powers[m + j]      (bit-reversed twiddle)
//           for k in 2*s*j .. 2*s*j + s:
//               u = a[k]
//               v = a[k+s] * ψ_j
//               a[k]   = u + v
//               a[k+s] = u - v
//
// Inverse (Gentleman-Sande, decimation-in-time): mirror layout, swap u/v
// pre/post-multiply, then divide by N at the end.
#include <ssns/ckks/ntt.hpp>

#include <ssns/ckks/modarith.hpp>

#include <bit>
#include <stdexcept>

namespace ssns::ckks {

namespace {

// Bit-reversal of `x` using `bits` significant bits.  Used at NTT setup
// to pack ψ^bitrev(k) into psi_powers_ — at run time the inner loop just
// indexes psi_powers_[m + j] sequentially.
std::uint64_t bitreverse(std::uint64_t x, int bits) {
    std::uint64_t r = 0;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

}  // namespace

NTT::NTT(std::uint64_t p, std::size_t N) : p_(p), N_(N) {
    if (!is_power_of_two(N)) {
        throw std::invalid_argument("NTT: N must be a power of two");
    }
    const std::uint64_t two_n = static_cast<std::uint64_t>(2 * N);
    if ((p - 1) % two_n != 0) {
        throw std::invalid_argument("NTT: 2N does not divide p-1; no primitive 2N-th root in F_p");
    }
    const std::uint64_t psi = primitive_2n_root(p, two_n);
    const std::uint64_t inv_psi = inv_mod(psi, p);

    const int log_N = std::bit_width(N) - 1;
    psi_powers_.resize(N);
    inv_psi_powers_.resize(N);
    // Build the bit-reversed twiddle tables.  psi_powers_[bitrev(k)] = ψ^k.
    std::uint64_t cur = 1;
    std::uint64_t cur_inv = 1;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t br = static_cast<std::size_t>(bitreverse(k, log_N));
        psi_powers_[br] = cur;
        inv_psi_powers_[br] = cur_inv;
        cur     = mul_mod(cur,     psi,     p);
        cur_inv = mul_mod(cur_inv, inv_psi, p);
    }
    inv_N_ = inv_mod(static_cast<std::uint64_t>(N), p);
}

void NTT::forward(std::uint64_t* a) const {
    // Cooley-Tukey, top-down: s = N/2, N/4, ..., 1.  m = N/(2s) doubles each round.
    const std::uint64_t p = p_;
    std::size_t m = 1;
    for (std::size_t s = N_ / 2; s >= 1; s >>= 1) {
        for (std::size_t j = 0; j < m; ++j) {
            const std::uint64_t w = psi_powers_[m + j];
            const std::size_t base = 2 * s * j;
            for (std::size_t k = base; k < base + s; ++k) {
                const std::uint64_t u = a[k];
                const std::uint64_t v = mul_mod(a[k + s], w, p);
                a[k]     = add_mod(u, v, p);
                a[k + s] = sub_mod(u, v, p);
            }
        }
        m <<= 1;
    }
}

void NTT::inverse(std::uint64_t* a) const {
    // Gentleman-Sande, bottom-up: s = 1, 2, ..., N/2.  m = N/(2s) halves each round.
    const std::uint64_t p = p_;
    std::size_t m = N_ / 2;
    for (std::size_t s = 1; s < N_; s <<= 1) {
        for (std::size_t j = 0; j < m; ++j) {
            const std::uint64_t w = inv_psi_powers_[m + j];
            const std::size_t base = 2 * s * j;
            for (std::size_t k = base; k < base + s; ++k) {
                const std::uint64_t u = a[k];
                const std::uint64_t v = a[k + s];
                a[k]     = add_mod(u, v, p);
                a[k + s] = mul_mod(sub_mod(u, v, p), w, p);
            }
        }
        m >>= 1;
    }
    // Divide by N.
    for (std::size_t i = 0; i < N_; ++i) {
        a[i] = mul_mod(a[i], inv_N_, p);
    }
}

}  // namespace ssns::ckks
