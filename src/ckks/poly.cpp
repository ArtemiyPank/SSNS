// polynomial in rns form impl see header
#include <ssns/ckks/poly.hpp>

#include <ssns/ckks/modarith.hpp>

#include <limits>
#include <stdexcept>

namespace ssns::ckks {
    // all zero polynomial
    Polynomial::Polynomial() {
        for (auto &r: residues) {
            r.assign(POLY_DEGREE, 0);
        }
    }

    // lift small signed coefs into rns form
    Polynomial Polynomial::from_coeffs(const std::vector<std::int64_t> &coeffs) {
        if (coeffs.size() > POLY_DEGREE) {
            throw std::invalid_argument("Polynomial::from_coeffs: too many coefficients");
        }
        Polynomial out;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            for (std::size_t k = 0; k < coeffs.size(); ++k) {
                const std::int64_t c = coeffs[k];
                // reduce signed coef into [0, q) add q if negative
                // operating in int64 with prime up to ~2^60 leaves room for safety on the negative branch
                std::uint64_t r;
                if (c >= 0) {
                    r = static_cast<std::uint64_t>(c) % q;
                } else {
                    // -c is positive in int64 unless c == INT64_MIN guard anyway
                    const std::uint64_t mag = (c == std::numeric_limits<std::int64_t>::min())
                                                  ? (static_cast<std::uint64_t>(std::numeric_limits<
                                                         std::int64_t>::max()) + 1ULL)
                                                  : static_cast<std::uint64_t>(-c);
                    const std::uint64_t mag_mod = mag % q;
                    r = (mag_mod == 0) ? 0 : (q - mag_mod);
                }
                out.residues[i][k] = r;
            }
        }
        return out;
    }

    // in place add this += other per prime
    Polynomial &Polynomial::add_inplace(const Polynomial &other) {
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            auto &a = residues[i];
            const auto &b = other.residues[i];
            for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
                a[k] = add_mod(a[k], b[k], q);
            }
        }
        return *this;
    }

    // in place sub this -= other per prime
    Polynomial &Polynomial::sub_inplace(const Polynomial &other) {
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            auto &a = residues[i];
            const auto &b = other.residues[i];
            for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
                a[k] = sub_mod(a[k], b[k], q);
            }
        }
        return *this;
    }

    // poly mul via per prime forward ntt pointwise mul inverse ntt
    Polynomial Polynomial::multiply(
        const Polynomial &a, const Polynomial &b,
        const std::array<NTT, NUM_PRIMES> &ntts) {
        Polynomial out;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            // defensive check ntts[i] must match prime i and N
            if (ntts[i].prime() != q || ntts[i].degree() != POLY_DEGREE) {
                throw std::invalid_argument(
                    "Polynomial::multiply: NTT instance does not match prime / degree");
            }
            // copy operands so originals stay untouched
            std::vector<std::uint64_t> fa = a.residues[i];
            std::vector<std::uint64_t> fb = b.residues[i];
            ntts[i].forward(fa.data());
            ntts[i].forward(fb.data());
            std::vector<std::uint64_t> &fc = out.residues[i];
            for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
                fc[k] = mul_mod(fa[k], fb[k], q);
            }
            ntts[i].inverse(fc.data());
        }
        return out;
    }

    // equality is bitwise on every residue
    bool Polynomial::operator==(const Polynomial &other) const noexcept {
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            if (residues[i] != other.residues[i]) return false;
        }
        return true;
    }
} // namespace ssns::ckks
