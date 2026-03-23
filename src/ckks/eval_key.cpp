// EvalKey — implementation.  See header for the RNS-gadget relinearisation
// math.
#include <ssns/ckks/eval_key.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/public_key.hpp>  // PublicKey types only — keep our own σ.

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
    // Pre-compute s² in coefficient form (used to embed the s² term at the
    // appropriate RNS slot for every sub-key).  Polynomial::multiply takes
    // coefficient-form inputs and returns coefficient form.
    Polynomial s_squared_coeff = Polynomial::multiply(sk.s, sk.s, ntts);

    EvalKey evk;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        // 1. Uniform a_i in coefficient form.
        Polynomial a_coeff = sample_uniform_poly(rng);

        // 2. Gaussian e_i_noise in coefficient form.
        const std::vector<std::int64_t> e_int =
            sample_gaussian_poly(rng, EVAL_KEY_NOISE_SIGMA);
        Polynomial e_coeff = Polynomial::from_coeffs(e_int);

        // 3. a_i · s in coefficient form.
        Polynomial as = Polynomial::multiply(a_coeff, sk.s, ntts);

        // 4. b_i = e_i_noise − a_i·s + (e_i · s²)   (mod Q), coefficient form.
        //
        //    e_i · s² in RNS is "s² at slot i, zero at slots j ≠ i", so we
        //    only need to add s²'s residue at the i-th slot.  Other slots
        //    contribute nothing.
        Polynomial b_coeff = e_coeff;
        b_coeff.sub_inplace(as);
        const std::uint64_t q_i = COEFF_MODULI[i];
        auto& b_slot_i = b_coeff.residues[i];
        const auto& s2_slot_i = s_squared_coeff.residues[i];
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            b_slot_i[k] = add_mod(b_slot_i[k], s2_slot_i[k], q_i);
        }

        // 5. Convert both to NTT form for storage.
        EvalSubKey sub;
        sub.a = a_coeff;
        sub.b = b_coeff;
        to_ntt_form(sub.a, ntts);
        to_ntt_form(sub.b, ntts);
        evk.sub_keys[i] = std::move(sub);
    }
    return evk;
}

}  // namespace ssns::ckks
