// EvalKey — implementation.  See header for the BV relinearisation math.
#include <ssns/ckks/eval_key.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/public_key.hpp>  // KEYGEN_NOISE_SIGMA

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace ssns::ckks {

namespace {

// Box-Muller Gaussian sampler — duplicated from public_key.cpp.  Kept
// local so this TU has no link-time dependency on private helpers there.
double sample_gaussian(std::mt19937_64& rng, double sigma) {
    auto next_unit = [&]() {
        constexpr double scale = 1.0 / 9007199254740992.0;  // 2^-53
        std::uint64_t v;
        do {
            v = rng() >> 11;
        } while (v == 0ULL);
        return static_cast<double>(v) * scale;
    };
    const double u1 = next_unit();
    const double u2 = next_unit();
    const double r  = std::sqrt(-2.0 * std::log(u1));
    const double th = 2.0 * std::numbers::pi_v<double> * u2;
    return sigma * r * std::cos(th);
}

std::vector<std::int64_t> sample_gaussian_poly(std::mt19937_64& rng, double sigma) {
    std::vector<std::int64_t> out(POLY_DEGREE);
    for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
        const double g = sample_gaussian(rng, sigma);
        out[k] = static_cast<std::int64_t>(std::llround(g));
    }
    return out;
}

Polynomial sample_uniform_poly(std::mt19937_64& rng) {
    Polynomial a;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const std::uint64_t q = COEFF_MODULI[i];
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            a.residues[i][k] = rng() % q;
        }
    }
    return a;
}

void to_ntt_form(Polynomial& p, const std::array<NTT, NUM_PRIMES>& ntts) {
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ntts[i].forward(p.residues[i].data());
    }
}

}  // namespace

EvalKey gen_eval_key(const SecretKey& sk,
                     const std::array<NTT, NUM_PRIMES>& ntts,
                     std::mt19937_64& rng) {
    // 1. Uniform a_evk in coefficient form.
    Polynomial a_coeff = sample_uniform_poly(rng);

    // 2. Gaussian e_evk in coefficient form.
    const std::vector<std::int64_t> e_int = sample_gaussian_poly(rng, KEYGEN_NOISE_SIGMA);
    Polynomial e_coeff = Polynomial::from_coeffs(e_int);

    // 3. a_evk · s in coefficient form.
    Polynomial as = Polynomial::multiply(a_coeff, sk.s, ntts);

    // 4. s · s in coefficient form.
    Polynomial s_squared = Polynomial::multiply(sk.s, sk.s, ntts);

    // 5. b_evk = e − a_evk·s + s²  (mod q), coefficient form.
    Polynomial b_coeff = e_coeff;
    b_coeff.sub_inplace(as);
    b_coeff.add_inplace(s_squared);

    // 6. Convert both to NTT form for storage.
    EvalKey evk;
    evk.a = a_coeff;
    evk.b = b_coeff;
    to_ntt_form(evk.a, ntts);
    to_ntt_form(evk.b, ntts);
    return evk;
}

}  // namespace ssns::ckks
