// PublicKey impl see header
#include <ssns/ckks/public_key.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/poly.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace ssns::ckks {
    namespace {
        // box muller gaussian sampler
        // std::normal_distribution would work but its internal state is opaque
        // using box muller directly makes sampling reproducible across libstdc++ versions
        double sample_gaussian(std::mt19937_64 &rng, double sigma) {
            // two uniforms in (0, 1] avoid u1=0 to keep log defined
            auto next_unit = [&]() {
                // 53 bit uniform in (0, 1] use 2^-53 spacing the +eps shift ensures u1 > 0
                constexpr double scale = 1.0 / 9007199254740992.0; // 2^-53
                std::uint64_t v;
                do {
                    v = rng() >> 11;
                } while (v == 0ULL);
                return static_cast<double>(v) * scale;
            };
            const double u1 = next_unit();
            const double u2 = next_unit();
            const double r = std::sqrt(-2.0 * std::log(u1));
            const double th = 2.0 * std::numbers::pi_v<double> * u2;
            return sigma * r * std::cos(th);
        }

        // length N poly from rounded gaussian sigma returned as signed ints
        std::vector<std::int64_t> sample_gaussian_poly(std::mt19937_64 &rng, double sigma) {
            std::vector<std::int64_t> out(POLY_DEGREE);
            for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
                const double g = sample_gaussian(rng, sigma);
                out[k] = static_cast<std::int64_t>(std::llround(g));
            }
            return out;
        }

        // uniform per prime in [0, q_i) returns coef form
        Polynomial sample_uniform_poly(std::mt19937_64 &rng) {
            Polynomial a;
            for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
                const std::uint64_t q = COEFF_MODULI[i];
                for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
                    // naive rejection free reduction
                    // q is up to 60 bits bias from one u64 % q is at most 2^-4 of a residue negligible at our params
                    // can swap for full rejection sampling if needed
                    a.residues[i][k] = rng() % q;
                }
            }
            return a;
        }

        // forward ntt in place mutates p
        void to_ntt_form(Polynomial &p, const std::array<NTT, NUM_PRIMES> &ntts) {
            for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
                ntts[i].forward(p.residues[i].data());
            }
        }
    } // namespace

    // build fresh public key (b, a) from sk
    PublicKey gen_public_key(const SecretKey &sk,
                             const std::array<NTT, NUM_PRIMES> &ntts,
                             std::mt19937_64 &rng) {
        // 1 sample uniform a in coef form per prime
        Polynomial a_coeff = sample_uniform_poly(rng);

        // 2 sample gaussian noise e in coef form
        const std::vector<std::int64_t> e_int = sample_gaussian_poly(rng, KEYGEN_NOISE_SIGMA);
        Polynomial e_coeff = Polynomial::from_coeffs(e_int);

        // 3 a*s in coef form
        //   Polynomial::multiply takes coef form transforms internally returns coef form
        Polynomial as = Polynomial::multiply(a_coeff, sk.s, ntts);

        // 4 b = e - a*s (mod q) coef form
        Polynomial b_coeff = e_coeff;
        b_coeff.sub_inplace(as);

        // 5 convert both to ntt form for storage
        PublicKey pk;
        pk.a = a_coeff;
        pk.b = b_coeff;
        to_ntt_form(pk.a, ntts);
        to_ntt_form(pk.b, ntts);
        return pk;
    }
} // namespace ssns::ckks
