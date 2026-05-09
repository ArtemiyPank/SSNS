// encrypt impl see header
#include <ssns/ckks/encrypt.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/ntt_ops.hpp>
#include <ssns/ckks/poly.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace ssns::ckks {

namespace {

// box muller gaussian sampler same as public_key.cpp eval_key.cpp
// kept tu local to avoid coupling encrypt to keygen helpers
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

// sample length N poly of rounded gaussian deviates
std::vector<std::int64_t> sample_gaussian_poly(std::mt19937_64& rng, double sigma) {
    std::vector<std::int64_t> out(POLY_DEGREE);
    for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
        const double g = sample_gaussian(rng, sigma);
        out[k] = static_cast<std::int64_t>(std::llround(g));
    }
    return out;
}

// sparse ternary v for encryption iid with
//   P(+1) = 1/4 P(0) = 1/2 P(-1) = 1/4
// read two random bits 00 -> -1 11 -> +1 otherwise 0
// gives the right distribution from one u64 amortised over 32 coefs
std::vector<std::int64_t> sample_sparse_ternary(std::mt19937_64& rng) {
    std::vector<std::int64_t> out(POLY_DEGREE);
    std::uint64_t bits = 0;
    int bits_left = 0;
    for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
        if (bits_left < 2) {
            bits = rng();
            bits_left = 64;
        }
        const std::uint64_t pair = bits & 0x3ULL;
        bits >>= 2;
        bits_left -= 2;
        // 0b00 -> -1 0b11 -> +1 else 0
        if (pair == 0ULL) {
            out[k] = -1;
        } else if (pair == 3ULL) {
            out[k] = 1;
        } else {
            out[k] = 0;
        }
    }
    return out;
}

// forward ntt every residue of p in place
void to_ntt_form(Polynomial& p, const std::array<NTT, NUM_PRIMES>& ntts) {
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ntts[i].forward(p.residues[i].data());
    }
}

}  // namespace

// encrypt pt under pk with fresh randomness from rng
Ciphertext encrypt(const Plaintext& pt,
                   const PublicKey& pk,
                   const std::array<NTT, NUM_PRIMES>& ntts,
                   std::mt19937_64& rng) {
    // 1 sample v (sparse ternary) and e0 e1 (gaussian sigma=3.2) coef form lift to rns convert to ntt
    Polynomial v = Polynomial::from_coeffs(sample_sparse_ternary(rng));
    Polynomial e0 = Polynomial::from_coeffs(sample_gaussian_poly(rng, KEYGEN_NOISE_SIGMA));
    Polynomial e1 = Polynomial::from_coeffs(sample_gaussian_poly(rng, KEYGEN_NOISE_SIGMA));
    to_ntt_form(v, ntts);
    to_ntt_form(e0, ntts);
    to_ntt_form(e1, ntts);

    // 2 b*v and a*v pointwise in ntt form
    Polynomial bv = pointwise_mul_ntt(pk.b, v);
    Polynomial av = pointwise_mul_ntt(pk.a, v);

    // 3 c0 = b*v + e0 + pt    c1 = a*v + e1
    Polynomial c0 = pointwise_add(bv, e0);
    c0 = pointwise_add(c0, pt.poly);
    Polynomial c1 = pointwise_add(av, e1);

    Ciphertext ct;
    ct.c0 = std::move(c0);
    ct.c1 = std::move(c1);
    ct.scale = pt.scale;
    ct.level = pt.level;
    return ct;
}

}  // namespace ssns::ckks
