// pointwise polynomial ops impl
#include <ssns/ckks/ntt_ops.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/params.hpp>

namespace ssns::ckks {
    namespace {
        // per prime psm reduced pointwise multiply
        // templated so compiler folds P/C and inlines mul_mod_psm{40,60} fully
        template<std::uint64_t P, std::uint64_t C, bool IS60>
        inline void pointwise_mul_one_prime(const std::uint64_t *__restrict__ a,
                                            const std::uint64_t *__restrict__ b,
                                            std::uint64_t *__restrict__ o) noexcept {
            for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
                if constexpr (IS60) o[k] = mul_mod_psm60<P, C>(a[k], b[k]);
                else o[k] = mul_mod_psm40<P, C>(a[k], b[k]);
            }
        }
    } // namespace

    // pointwise multiply two ntt form polys slot by slot per prime
    Polynomial pointwise_mul_ntt(const Polynomial &a, const Polynomial &b) {
        Polynomial out;
        pointwise_mul_one_prime<COEFF_MODULI[0], PSM_C[0], true>(a.residues[0].data(), b.residues[0].data(),
                                                                 out.residues[0].data());
        pointwise_mul_one_prime<COEFF_MODULI[1], PSM_C[1], false>(a.residues[1].data(), b.residues[1].data(),
                                                                  out.residues[1].data());
        pointwise_mul_one_prime<COEFF_MODULI[2], PSM_C[2], false>(a.residues[2].data(), b.residues[2].data(),
                                                                  out.residues[2].data());
        pointwise_mul_one_prime<COEFF_MODULI[3], PSM_C[3], true>(a.residues[3].data(), b.residues[3].data(),
                                                                 out.residues[3].data());
        return out;
    }

    // pointwise add two polys slot by slot per prime
    Polynomial pointwise_add(const Polynomial &a, const Polynomial &b) {
        Polynomial out;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            const auto &ai = a.residues[i];
            const auto &bi = b.residues[i];
            auto &oi = out.residues[i];
            for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
                oi[k] = add_mod(ai[k], bi[k], q);
            }
        }
        return out;
    }

    // pointwise sub two polys slot by slot per prime
    Polynomial pointwise_sub(const Polynomial &a, const Polynomial &b) {
        Polynomial out;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            const auto &ai = a.residues[i];
            const auto &bi = b.residues[i];
            auto &oi = out.residues[i];
            for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
                oi[k] = sub_mod(ai[k], bi[k], q);
            }
        }
        return out;
    }
} // namespace ssns::ckks
