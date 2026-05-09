// negacyclic ntt impl
//
// forward (cooley tukey decimation in frequency no bit reversal at end output ends up bit reversed)
//   for s in N/2, N/4, ..., 1:           (m = N/(2s) doubles each round)
//       for j in 0..m:
//           psi_j = psi_powers[m + j]    (bit reversed twiddle)
//           for k in 2*s*j .. 2*s*j + s:
//               u = a[k]
//               v = a[k+s] * psi_j
//               a[k]   = u + v
//               a[k+s] = u - v
//
// inverse (gentleman sande decimation in time) mirror layout swap u/v pre/post multiply then divide by N
//
// помнить: ntt работает на butterfly стадиях нельзя менять порядок
#include <ssns/ckks/ntt.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/params.hpp>

#include <bit>
#include <stdexcept>

namespace ssns::ckks {

namespace {

// bit reversal of x using `bits` significant bits
// used at ntt setup to pack psi^bitrev(k) into psi_powers_
// at run time inner loop just indexes psi_powers_[m + j] sequentially
//
// почему bit reversed: в cooley tukey decimation in frequency на стадии s twiddle нужен в порядке psi^bitrev(k)
// пред вычисляем один раз потом просто читаем по индексу m+j без runtime bitrev в hot loop
std::uint64_t bitreverse(std::uint64_t x, int bits) {
    std::uint64_t r = 0;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

// true iff n is positive power of two
bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

}  // namespace

// build twiddle tables for prime p and degree N
NTT::NTT(std::uint64_t p, std::size_t N) : p_(p), N_(N) {
    if (!is_power_of_two(N)) {
        throw std::invalid_argument("NTT: N must be a power of two");
    }
    const std::uint64_t two_n = static_cast<std::uint64_t>(2 * N);
    if ((p - 1) % two_n != 0) {
        throw std::invalid_argument("NTT: 2N does not divide p-1; no primitive 2N-th root in F_p");
    }
    // 2N-й корень нужен потому что polynomial X^N+1 negacyclic
    const std::uint64_t psi = primitive_2n_root(p, two_n);
    const std::uint64_t inv_psi = inv_mod(psi, p);

    const int log_N = std::bit_width(N) - 1;
    psi_powers_.resize(N);
    inv_psi_powers_.resize(N);
    // build bit reversed twiddle tables
    // psi_powers_[bitrev(k)] = psi^k
    // тут именно один проход cur *= psi вместо pow_mod каждой итерации
    std::uint64_t cur = 1;
    std::uint64_t cur_inv = 1;
    for (std::size_t k = 0; k < N; ++k) {
        const std::size_t br = static_cast<std::size_t>(bitreverse(k, log_N));
        psi_powers_[br] = cur;
        inv_psi_powers_[br] = cur_inv;
        cur     = mul_mod(cur,     psi,     p);
        cur_inv = mul_mod(cur_inv, inv_psi, p);
    }
    // inv_N для финального деления на N в обратном ntt
    inv_N_ = inv_mod(static_cast<std::uint64_t>(N), p);
}

// psm reduced templated ntt inner loops
// pseudo mersenne constants come from PSM_C in params.hpp
// passing them as template args lets the compiler fully constant fold mul_mod_psm{40,60}
namespace {

// templated psm multiply
template <std::uint64_t P, std::uint64_t C, bool IS60>
inline std::uint64_t psm_mul(std::uint64_t a, std::uint64_t b) noexcept {
    if constexpr (IS60) return mul_mod_psm60<P, C>(a, b);
    else                return mul_mod_psm40<P, C>(a, b);
}

// forward ntt specialised for compile time prime / psm constants
template <std::uint64_t P, std::uint64_t C, bool IS60>
void forward_typed(std::uint64_t* a, std::size_t N,
                    const std::uint64_t* psi_powers) {
    // помнить cooley tukey DIF: butterfly (u, v*w) -> (u+v*w, u-v*w)
    // w = psi^bitrev(j) уже в bit reversed таблице
    std::size_t m = 1;
    for (std::size_t s = N / 2; s >= 1; s >>= 1) {
        for (std::size_t j = 0; j < m; ++j) {
            const std::uint64_t w = psi_powers[m + j];
            const std::size_t base = 2 * s * j;
            for (std::size_t k = base; k < base + s; ++k) {
                const std::uint64_t u = a[k];
                // psm_mul быстрее обычного mul_mod в ~2 раза в hot loop
                const std::uint64_t v = psm_mul<P, C, IS60>(a[k + s], w);
                a[k]     = add_mod(u, v, P);
                a[k + s] = sub_mod(u, v, P);
            }
        }
        m <<= 1;
    }
}

// inverse ntt specialised for compile time prime / psm constants
template <std::uint64_t P, std::uint64_t C, bool IS60>
void inverse_typed(std::uint64_t* a, std::size_t N,
                    const std::uint64_t* inv_psi_powers, std::uint64_t inv_N) {
    // gentleman sande DIT: butterfly (u, v) -> (u+v, (u-v)*w)
    // mul ПОСЛЕ subtract в отличие от forward
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
    // финальное деление на N: иначе round trip даст a*N а не a
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = psm_mul<P, C, IS60>(a[i], inv_N);
    }
}
}  // anonymous namespace

// in place forward ntt dispatching to psm path when p_ matches one of the four ckks primes
void NTT::forward(std::uint64_t* a) const {
    // hoist prime to compile time constant by dispatching to one of four template instantiations
    // inner loop runs psm reduction with constant P and C ~2x faster than runtime mul_mod
    const auto* psi = psi_powers_.data();
    if      (p_ == COEFF_MODULI[0]) { forward_typed<COEFF_MODULI[0], PSM_C[0], true >(a, N_, psi); return; }
    else if (p_ == COEFF_MODULI[3]) { forward_typed<COEFF_MODULI[3], PSM_C[3], true >(a, N_, psi); return; }
    else if (p_ == COEFF_MODULI[1]) { forward_typed<COEFF_MODULI[1], PSM_C[1], false>(a, N_, psi); return; }
    else if (p_ == COEFF_MODULI[2]) { forward_typed<COEFF_MODULI[2], PSM_C[2], false>(a, N_, psi); return; }

    // generic fallback (used only by tests with non ckks primes)
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

// in place inverse ntt dispatching to psm path when p_ matches one of the four ckks primes
void NTT::inverse(std::uint64_t* a) const {
    const auto* inv_psi = inv_psi_powers_.data();
    if      (p_ == COEFF_MODULI[0]) { inverse_typed<COEFF_MODULI[0], PSM_C[0], true >(a, N_, inv_psi, inv_N_); return; }
    else if (p_ == COEFF_MODULI[3]) { inverse_typed<COEFF_MODULI[3], PSM_C[3], true >(a, N_, inv_psi, inv_N_); return; }
    else if (p_ == COEFF_MODULI[1]) { inverse_typed<COEFF_MODULI[1], PSM_C[1], false>(a, N_, inv_psi, inv_N_); return; }
    else if (p_ == COEFF_MODULI[2]) { inverse_typed<COEFF_MODULI[2], PSM_C[2], false>(a, N_, inv_psi, inv_N_); return; }

    // generic fallback for non ckks primes (tests)
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
