// ssns::ckks::EvalKey: RNS-gadget relinearisation key
//
// cipher*cipher gives degree-2 ct (c0, c1, c2); c0 + c1*s + c2*s^2 decrypts to product
// relin folds c2*s^2 back to degree-1 via NUM_PRIMES sub-keys satisfying, for each i:
//   sub_keys[i].b + sub_keys[i].a * s = e_i_noise + e_i * s^2    (mod Q)
// where e_i is RNS-basis indicator (1 mod q_i, 0 mod q_j for j != i)
//
// in RNS form:
//   slot i:       b + a*s ~ s^2 + small noise  (mod q_i)
//   slot j != i:  b + a*s ~       small noise  (mod q_j)
//
// per-slot identity checked in coeff form; sigma=EVAL_KEY_NOISE_SIGMA (0.5)
// 6*sigma ~ 3; cap at 30 to swallow FFT slop
#include <catch.hpp>

#include <ssns/ckks/crt.hpp>
#include <ssns/ckks/eval_key.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <random>

using ssns::ckks::COEFF_MODULI;
using ssns::ckks::EVAL_KEY_NOISE_SIGMA;
using ssns::ckks::EvalKey;
using ssns::ckks::NTT;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::Polynomial;
using ssns::ckks::SecretKey;
using ssns::ckks::gen_eval_key;

namespace {

const std::array<NTT, NUM_PRIMES>& shared_ntts() {
    static const std::array<NTT, NUM_PRIMES> instance = {{
        NTT(COEFF_MODULI[0], POLY_DEGREE),
        NTT(COEFF_MODULI[1], POLY_DEGREE),
        NTT(COEFF_MODULI[2], POLY_DEGREE),
        NTT(COEFF_MODULI[3], POLY_DEGREE),
    }};
    return instance;
}

Polynomial to_coeff_form(const Polynomial& p_ntt,
                         const std::array<NTT, NUM_PRIMES>& ntts) {
    Polynomial out = p_ntt;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ntts[i].inverse(out.residues[i].data());
    }
    return out;
}

// center a single uint64 residue r mod q into (-q/2, q/2] as a double
double centre_residue(std::uint64_t r, std::uint64_t q) {
    if (r > (q >> 1)) {
        return -static_cast<double>(q - r);
    }
    return static_cast<double>(r);
}

}  // namespace

TEST_CASE("EvalKey: each sub-key matches RNS-gadget identity per slot",
          "[ckks][eval_key]") {
    std::mt19937_64 rng(0x5EED5EEDULL);
    SecretKey sk = SecretKey::sample(rng);
    EvalKey evk = gen_eval_key(sk, shared_ntts(), rng);

    // pre-compute s^2 in coeff form; used at slot i of every sub-key i
    Polynomial s_squared = Polynomial::multiply(sk.s, sk.s, shared_ntts());

    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        // lift sub_keys[i] into coeff form
        Polynomial a_coeff = to_coeff_form(evk.sub_keys[i].a, shared_ntts());
        Polynomial b_coeff = to_coeff_form(evk.sub_keys[i].b, shared_ntts());

        // a_i * s in coeff form
        Polynomial as = Polynomial::multiply(a_coeff, sk.s, shared_ntts());

        // residual = b_i + a_i*s - e_i*s^2 (~ Gaussian e_i_noise); subtract s^2 only at slot i
        Polynomial residual = b_coeff;
        residual.add_inplace(as);
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            const std::uint64_t q_i = COEFF_MODULI[i];
            // subtract s^2 only at slot i
            const std::uint64_t s2_i = s_squared.residues[i][k];
            std::uint64_t& r = residual.residues[i][k];
            r = (r >= s2_i) ? (r - s2_i) : (r + q_i - s2_i);
        }

        // residual should be small Gaussian noise on every slot; 6*sigma ~ 3, allow 30 for FFT slop
        for (std::size_t slot = 0; slot < NUM_PRIMES; ++slot) {
            const std::uint64_t q = COEFF_MODULI[slot];
            double max_abs = 0.0;
            for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
                const double centred = centre_residue(residual.residues[slot][k], q);
                if (std::abs(centred) > max_abs) max_abs = std::abs(centred);
            }
            INFO("sub_key " << i << " slot " << slot << " max_abs=" << max_abs);
            REQUIRE(max_abs < 30.0);
        }
    }
    // sanity on sigma; fail loudly if someone bumps past the noise budget
    REQUIRE(EVAL_KEY_NOISE_SIGMA <= 1.0);
}

TEST_CASE("EvalKey: same seed produces identical key",
          "[ckks][eval_key]") {
    std::mt19937_64 rng_a(0xBEEF1234ULL);
    SecretKey sk_a = SecretKey::sample(rng_a);
    EvalKey evk_a = gen_eval_key(sk_a, shared_ntts(), rng_a);

    std::mt19937_64 rng_b(0xBEEF1234ULL);
    SecretKey sk_b = SecretKey::sample(rng_b);
    EvalKey evk_b = gen_eval_key(sk_b, shared_ntts(), rng_b);

    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        REQUIRE(evk_a.sub_keys[i].a == evk_b.sub_keys[i].a);
        REQUIRE(evk_a.sub_keys[i].b == evk_b.sub_keys[i].b);
    }
}
