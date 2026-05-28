// ckks::PublicKey
// pk = (b, a) where b = -a*s + e (mod q), e ~ discrete Gaussian sigma=3.2
// b and a stored in NTT form for fast encryption
//
// round-trip (decrypt-style): coeff-form b + a*s should be a small noise polynomial
// centered sup-norm well under 50, sigma=3.2 gives per-coeff |e| <= ~6 sigma ~ 20
#include <catch.hpp>

#include <ssns/ckks/crt.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/public_key.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <cstdint>
#include <random>

using ssns::ckks::COEFF_MODULI;
using ssns::ckks::NTT;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::Polynomial;
using ssns::ckks::PublicKey;
using ssns::ckks::SecretKey;
using ssns::ckks::U256;
using ssns::ckks::crt_lift;
using ssns::ckks::crt_center_to_double;
using ssns::ckks::gen_public_key;

namespace {

// shared NTT tables across all primes, built once
const std::array<NTT, NUM_PRIMES>& shared_ntts() {
    static const std::array<NTT, NUM_PRIMES> instance = {{
        NTT(COEFF_MODULI[0], POLY_DEGREE),
        NTT(COEFF_MODULI[1], POLY_DEGREE),
        NTT(COEFF_MODULI[2], POLY_DEGREE),
        NTT(COEFF_MODULI[3], POLY_DEGREE),
    }};
    return instance;
}

// inverse NTT per prime, returns coefficient-form polynomial
Polynomial to_coeff_form(const Polynomial& p_ntt,
                         const std::array<NTT, NUM_PRIMES>& ntts) {
    Polynomial out = p_ntt;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ntts[i].inverse(out.residues[i].data());
    }
    return out;
}

}  // namespace

TEST_CASE("PublicKey: b + a*s ≈ e (small Gaussian)",
          "[ckks][public_key]") {
    std::mt19937_64 rng(0xCAFEF00DULL);
    SecretKey sk = SecretKey::sample(rng);
    PublicKey pk = gen_public_key(sk, shared_ntts(), rng);

    // pk.a NTT -> coeff
    Polynomial a_coeff = to_coeff_form(pk.a, shared_ntts());

    // a * s in coeff form, Polynomial::multiply takes coeff inputs and returns coeff
    Polynomial as = Polynomial::multiply(a_coeff, sk.s, shared_ntts());

    // b in coeff form
    Polynomial b_coeff = to_coeff_form(pk.b, shared_ntts());

    // noise = b + a*s (mod q) in coeff form
    Polynomial noise = b_coeff;
    noise.add_inplace(as);

    // centered sup-norm via CRT lift (all primes)
    double max_abs = 0.0;
    for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
        std::array<std::uint64_t, NUM_PRIMES> r;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            r[i] = noise.residues[i][k];
        }
        const U256 lifted = crt_lift(r);
        const double centered = crt_center_to_double(lifted);
        if (std::abs(centered) > max_abs) max_abs = std::abs(centered);
    }
    REQUIRE(max_abs < 50.0);
}

TEST_CASE("PublicKey: same seed produces identical key",
          "[ckks][public_key]") {
    std::mt19937_64 rng_seed_a(0x600DBEEFULL);
    SecretKey sk_a = SecretKey::sample(rng_seed_a);
    PublicKey pk_a = gen_public_key(sk_a, shared_ntts(), rng_seed_a);

    std::mt19937_64 rng_seed_b(0x600DBEEFULL);
    SecretKey sk_b = SecretKey::sample(rng_seed_b);
    PublicKey pk_b = gen_public_key(sk_b, shared_ntts(), rng_seed_b);

    REQUIRE(pk_a.a == pk_b.a);
    REQUIRE(pk_a.b == pk_b.b);
}

TEST_CASE("PublicKey: a is uniform-ish (≥4000 of 8192 entries > q0/4 in slot 0)",
          "[ckks][public_key]") {
    std::mt19937_64 rng(0xFEEDF00D5ULL);
    SecretKey sk = SecretKey::sample(rng);
    PublicKey pk = gen_public_key(sk, shared_ntts(), rng);

    const std::uint64_t q0_quarter = COEFF_MODULI[0] / 4;
    std::size_t large = 0;
    for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
        if (pk.a.residues[0][k] > q0_quarter) ++large;
    }
    // expect ~3/4 * N to exceed q/4 under uniform sampling
    // relax to 1/2 * N for ample slack vs natural variance, works at any N
    REQUIRE(large >= POLY_DEGREE / 2);
}
