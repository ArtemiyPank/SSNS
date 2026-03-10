// Pointwise polynomial operations — implementation.
#include <ssns/ckks/ntt_ops.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/params.hpp>

namespace ssns::ckks {

Polynomial pointwise_mul_ntt(const Polynomial& a, const Polynomial& b) {
    Polynomial out;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const std::uint64_t q = COEFF_MODULI[i];
        const auto& ai = a.residues[i];
        const auto& bi = b.residues[i];
        auto& oi = out.residues[i];
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            oi[k] = mul_mod(ai[k], bi[k], q);
        }
    }
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
