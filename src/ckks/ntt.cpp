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
#include <ssns/ckks/params.hpp>

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

// PSM-reduced templated NTT inner loops.  Pseudo-Mersenne constants
// come from `PSM_C` in params.hpp; passing them as template arguments
// lets the compiler fully constant-fold mul_mod_psm{40,60}.
namespace {

// Templated forward butterfly — `mul_mod_psm{40,60}<P,C>` lets the compiler
// fully unroll modular reduction at compile time.
template <std::uint64_t P, std::uint64_t C, bool IS60>
inline std::uint64_t psm_mul(std::uint64_t a, std::uint64_t b) noexcept {
    if constexpr (IS60) return mul_mod_psm60<P, C>(a, b);
    else                return mul_mod_psm40<P, C>(a, b);
}

template <std::uint64_t P, std::uint64_t C, bool IS60>
void forward_typed(std::uint64_t* a, std::size_t N,
                    const std::uint64_t* psi_powers) {
    std::size_t m = 1;
    for (std::size_t s = N / 2; s >= 1; s >>= 1) {
        for (std::size_t j = 0; j < m; ++j) {
            const std::uint64_t w = psi_powers[m + j];
            const std::size_t base = 2 * s * j;
            for (std::size_t k = base; k < base + s; ++k) {
                const std::uint64_t u = a[k];
                const std::uint64_t v = psm_mul<P, C, IS60>(a[k + s], w);
                a[k]     = add_mod(u, v, P);
                a[k + s] = sub_mod(u, v, P);
            }
        }
        m <<= 1;
    }
}

template <std::uint64_t P, std::uint64_t C, bool IS60>
void inverse_typed(std::uint64_t* a, std::size_t N,
                    const std::uint64_t* inv_psi_powers, std::uint64_t inv_N) {
    std::size_t m = N / 2;
    for (std::size_t s = 1; s < N; s <<= 1) {
        for (std::size_t j = 0; j < m; ++j) {
            const std::uint64_t w = inv_psi_powers[m + j];
            const std::size_t base = 2 * s * j;
            for (std::size_t k = base; k < base + s; ++k) {
                const std::uint64_t u = a[k];
                const std::uint64_t v = a[k + s];
                a[k]     = add_mod(u, v, P);
                a[k + s] = psm_mul<P, C, IS60>(sub_mod(u, v, P), w);
            }
        }
        m >>= 1;
    }
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = psm_mul<P, C, IS60>(a[i], inv_N);
    }
}
}  // anonymous namespace

void NTT::forward(std::uint64_t* a) const {
    // Hoist the prime to a compile-time constant by dispatching to one of
    // four template instantiations.  The inner loop then runs PSM
    // reduction with constant P and C — ~2× faster than the runtime
    // mul_mod fallback below.
    const auto* psi = psi_powers_.data();
    if      (p_ == COEFF_MODULI[0]) { forward_typed<COEFF_MODULI[0], PSM_C[0], true >(a, N_, psi); return; }
    else if (p_ == COEFF_MODULI[3]) { forward_typed<COEFF_MODULI[3], PSM_C[3], true >(a, N_, psi); return; }
    else if (p_ == COEFF_MODULI[1]) { forward_typed<COEFF_MODULI[1], PSM_C[1], false>(a, N_, psi); return; }
    else if (p_ == COEFF_MODULI[2]) { forward_typed<COEFF_MODULI[2], PSM_C[2], false>(a, N_, psi); return; }

    // Generic fallback (used only by tests that construct NTT with a
    // non-CKKS prime, e.g. modarith parity tests).
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
    const auto* inv_psi = inv_psi_powers_.data();
    if      (p_ == COEFF_MODULI[0]) { inverse_typed<COEFF_MODULI[0], PSM_C[0], true >(a, N_, inv_psi, inv_N_); return; }
    else if (p_ == COEFF_MODULI[3]) { inverse_typed<COEFF_MODULI[3], PSM_C[3], true >(a, N_, inv_psi, inv_N_); return; }
    else if (p_ == COEFF_MODULI[1]) { inverse_typed<COEFF_MODULI[1], PSM_C[1], false>(a, N_, inv_psi, inv_N_); return; }
    else if (p_ == COEFF_MODULI[2]) { inverse_typed<COEFF_MODULI[2], PSM_C[2], false>(a, N_, inv_psi, inv_N_); return; }

    // Generic fallback for non-CKKS primes (tests).
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
    for (std::size_t i = 0; i < N_; ++i) {
        a[i] = mul_mod(a[i], inv_N_, p);
    }
}

}  // namespace ssns::ckks
