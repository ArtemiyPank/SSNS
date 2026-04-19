// Pointwise polynomial operations — implementation.
#include <ssns/ckks/ntt_ops.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/params.hpp>

namespace ssns::ckks {

namespace {

// Compile-time pseudo-Mersenne constants for the four CKKS primes.
// q0/q3 are 60-bit, q1/q2 are 40-bit; all are 2^K − small_C with C < 2^21.
// Pseudo-Mersenne reduction replaces __uint128_t / p hardware divide
// (~30 cycles) with a few shift-mask-mul-add iterations (~12 cycles).
constexpr std::uint64_t PSM_C0 = (std::uint64_t{1} << 60) - COEFF_MODULI[0];
constexpr std::uint64_t PSM_C1 = (std::uint64_t{1} << 40) - COEFF_MODULI[1];
constexpr std::uint64_t PSM_C2 = (std::uint64_t{1} << 40) - COEFF_MODULI[2];
constexpr std::uint64_t PSM_C3 = (std::uint64_t{1} << 60) - COEFF_MODULI[3];

template <std::uint64_t P, std::uint64_t C, bool IS60>
inline void pointwise_mul_one_prime(const std::uint64_t* __restrict__ a,
                                     const std::uint64_t* __restrict__ b,
                                     std::uint64_t* __restrict__ o) noexcept
{
    for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
        if constexpr (IS60) {
            o[k] = mul_mod_psm60<P, C>(a[k], b[k]);
        } else {
            o[k] = mul_mod_psm40<P, C>(a[k], b[k]);
        }
    }
}

}  // namespace

Polynomial pointwise_mul_ntt(const Polynomial& a, const Polynomial& b) {
    Polynomial out;
    pointwise_mul_one_prime<COEFF_MODULI[0], PSM_C0, /*IS60=*/true >(a.residues[0].data(), b.residues[0].data(), out.residues[0].data());
    pointwise_mul_one_prime<COEFF_MODULI[1], PSM_C1, /*IS60=*/false>(a.residues[1].data(), b.residues[1].data(), out.residues[1].data());
    pointwise_mul_one_prime<COEFF_MODULI[2], PSM_C2, /*IS60=*/false>(a.residues[2].data(), b.residues[2].data(), out.residues[2].data());
    pointwise_mul_one_prime<COEFF_MODULI[3], PSM_C3, /*IS60=*/true >(a.residues[3].data(), b.residues[3].data(), out.residues[3].data());
    return out;
}

Polynomial pointwise_add(const Polynomial& a, const Polynomial& b) {
    Polynomial out;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const std::uint64_t q = COEFF_MODULI[i];
        const auto& ai = a.residues[i];
        const auto& bi = b.residues[i];
        auto& oi = out.residues[i];
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            oi[k] = add_mod(ai[k], bi[k], q);
        }
    }
    return out;
}

Polynomial pointwise_sub(const Polynomial& a, const Polynomial& b) {
    Polynomial out;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const std::uint64_t q = COEFF_MODULI[i];
        const auto& ai = a.residues[i];
        const auto& bi = b.residues[i];
        auto& oi = out.residues[i];
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            oi[k] = sub_mod(ai[k], bi[k], q);
        }
    }
    return out;
}

}  // namespace ssns::ckks
