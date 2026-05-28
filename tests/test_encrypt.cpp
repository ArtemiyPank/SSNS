// ssns::ckks::encrypt: pk encryption gives (c0, c1) with c0 + c1*s ~ pt + small noise (mod q)
// noise from keygen e plus encrypt-time e0, e1 (sigma=3.2), modulated by sparse ternary blind v
//
// coverage:
//   1. scale + level pass through
//   2. same seed -> bit-identical ct
//   3. correctness: lift c0 + c1*s - pt to centered ints, sup-norm well under 2000
//      (per-coeff |.| <~ a few * sigma * sqrt(N * density))
#include <catch.hpp>

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/crt.hpp>
#include <ssns/ckks/encrypt.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/ntt_ops.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/public_key.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

using ssns::ckks::COEFF_MODULI;
using ssns::ckks::Ciphertext;
using ssns::ckks::NTT;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::Plaintext;
using ssns::ckks::Polynomial;
using ssns::ckks::PublicKey;
using ssns::ckks::SecretKey;
using ssns::ckks::U256;
using ssns::ckks::crt_center_to_double;
using ssns::ckks::crt_lift;
using ssns::ckks::encrypt;
using ssns::ckks::gen_public_key;
using ssns::ckks::pointwise_mul_ntt;

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

Polynomial to_ntt_form(const Polynomial& p_coeff,
                       const std::array<NTT, NUM_PRIMES>& ntts) {
    Polynomial out = p_coeff;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ntts[i].forward(out.residues[i].data());
    }
    return out;
}

// centered sup-norm of a coeff-form poly via CRT lift
double sup_norm_centered(const Polynomial& p_coeff) {
    double max_abs = 0.0;
    for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
        std::array<std::uint64_t, NUM_PRIMES> r;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            r[i] = p_coeff.residues[i][k];
        }
        const U256 lifted = crt_lift(r);
        const double centered = crt_center_to_double(lifted);
        if (std::abs(centered) > max_abs) max_abs = std::abs(centered);
    }
    return max_abs;
}

}  // namespace

TEST_CASE("encrypt: scale and level pass through from plaintext",
          "[ckks][encrypt]") {
    std::mt19937_64 rng(0xABCDEF01ULL);
    SecretKey sk = SecretKey::sample(rng);
    PublicKey pk = gen_public_key(sk, shared_ntts(), rng);

    // small plaintext
    std::vector<std::int64_t> coeffs = {1, -2, 3};
    Polynomial p_coeff = Polynomial::from_coeffs(coeffs);
    const double scale = 1.0e12;
    Plaintext pt = Plaintext::from_polynomial(p_coeff, scale, shared_ntts(), 3);

    Ciphertext ct = encrypt(pt, pk, shared_ntts(), rng);
    REQUIRE(ct.scale == scale);
    REQUIRE(ct.level == 3);
}

TEST_CASE("encrypt: same RNG seed produces identical ciphertext",
          "[ckks][encrypt]") {
    // keygen + encrypt share the rng stream; same seed -> identical sk, pk, ct
    auto build = [](std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        SecretKey sk = SecretKey::sample(rng);
        PublicKey pk = gen_public_key(sk, shared_ntts(), rng);
        std::vector<std::int64_t> coeffs = {7, -3, 2, 0, 1};
        Plaintext pt = Plaintext::from_polynomial(
            Polynomial::from_coeffs(coeffs), 1.0e12, shared_ntts());
        return encrypt(pt, pk, shared_ntts(), rng);
    };

    Ciphertext ct_a = build(0xFEEDBABEULL);
    Ciphertext ct_b = build(0xFEEDBABEULL);
    REQUIRE(ct_a.c0 == ct_b.c0);
    REQUIRE(ct_a.c1 == ct_b.c1);
    REQUIRE(ct_a.scale == ct_b.scale);
    REQUIRE(ct_a.level == ct_b.level);
}

TEST_CASE("encrypt: c0 + c1·s − pt is small (sup-norm under 2000)",
          "[ckks][encrypt]") {
    std::mt19937_64 rng(0x1234ABCDULL);
    SecretKey sk = SecretKey::sample(rng);
    PublicKey pk = gen_public_key(sk, shared_ntts(), rng);

    std::vector<std::int64_t> coeffs = {10, -7, 4, -1, 2};
    Polynomial pt_coeff = Polynomial::from_coeffs(coeffs);
    Plaintext pt = Plaintext::from_polynomial(pt_coeff, 1.0e12, shared_ntts());

    Ciphertext ct = encrypt(pt, pk, shared_ntts(), rng);

    // c1 * s in NTT: bring sk.s to NTT, pointwise mul with ct.c1, then iNTT
    Polynomial s_ntt = to_ntt_form(sk.s, shared_ntts());
    Polynomial c1s_ntt = pointwise_mul_ntt(ct.c1, s_ntt);
    Polynomial c1s_coeff = to_coeff_form(c1s_ntt, shared_ntts());

    // c0 in coeff form
    Polynomial c0_coeff = to_coeff_form(ct.c0, shared_ntts());

    // noise = (c0 + c1*s) - pt, all in coeff form
    Polynomial noise = c0_coeff;
    noise.add_inplace(c1s_coeff);
    noise.sub_inplace(pt_coeff);

    // noise budget = e_pk*v + e0 + e1*s; e_pk*v dominates (v has density 1/2 over N=8192)
    // per-coeff std ~ sigma * sqrt(N/2) ~ 205; sup-norm ~ 4 * sigma * sqrt(N/2) ~ 820
    // e1*s contributes far less (std ~ sigma * sqrt(64) ~ 25 since |supp(s)| = 64)
    // total comfortably under 2000
    REQUIRE(sup_norm_centered(noise) < 2000.0);
}
