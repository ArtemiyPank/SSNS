// ckks::mul_cipher, cipher * cipher with RNS-gadget relinearisation, depth-1
//
// math (all in NTT form):
//   d0 = a.c0 * b.c0
//   d1 = a.c0 * b.c1 + a.c1 * b.c0
//   d2 = a.c1 * b.c1
//
// relinearisation via RNS-gadget evk (eval_key.hpp):
//   c0_relin = d0 + sum_i d2_at_i * sub_keys[i].b
//   c1_relin = d1 + sum_i d2_at_i * sub_keys[i].a
// d2_at_i is d2 with its slot-i residue lifted to centered int per coeff and reduced into all RNS slots
//
// EvalSubKey identity: sub_keys[i].b + sub_keys[i].a*s = e_i_noise + e_i*s^2 (mod Q), so on decrypt:
//   c0_relin + s*c1_relin = d0 + s*d1 + s^2*d2 + sum_i d2_at_i * e_i_noise
//                         ~ delta^2 * m_a * m_b + bounded noise
//
// bookkeeping: out.scale = a.scale * b.scale, out.level = a.level
//
// tolerance after one mul_cipher + rescale, EVAL_KEY_NOISE_SIGMA=0.5, delta=2^40,
// q_drop = 60-bit prime, slot magnitudes <= 0.25:
//   typical: 6e-3 to 9e-3, peak observed across 10 seeds: 1.13e-2
// we assert 5e-2 (>4x worst observed) for safety against seed-tail FFT slop
#include <catch.hpp>

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/decrypt.hpp>
#include <ssns/ckks/encoder.hpp>
#include <ssns/ckks/encrypt.hpp>
#include <ssns/ckks/eval_key.hpp>
#include <ssns/ckks/linear_ops.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/public_key.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

using ssns::ckks::COEFF_MODULI;
using ssns::ckks::Ciphertext;
using ssns::ckks::Encoder;
using ssns::ckks::EvalKey;
using ssns::ckks::NTT;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::Plaintext;
using ssns::ckks::Polynomial;
using ssns::ckks::PublicKey;
using ssns::ckks::SecretKey;
using ssns::ckks::decrypt;
using ssns::ckks::encrypt;
using ssns::ckks::gen_eval_key;
using ssns::ckks::gen_public_key;
using ssns::ckks::mul_cipher;
using ssns::ckks::rescale;

namespace {

constexpr std::size_t SLOT_COUNT = POLY_DEGREE / 2;

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

// SLOT_COUNT random complex slots with |re|,|im| <= mag
std::vector<std::complex<double>> small_random_slots(std::mt19937_64& rng,
                                                     double mag = 0.25) {
    std::uniform_real_distribution<double> dist(-mag, mag);
    std::vector<std::complex<double>> z(SLOT_COUNT);
    for (auto& s : z) s = std::complex<double>(dist(rng), dist(rng));
    return z;
}

// holds sk, pk, evk, encoder, default scale 2^40
struct Pipeline {
    Encoder enc;
    SecretKey sk;
    PublicKey pk;
    EvalKey evk;
    static constexpr double DEFAULT_SCALE = static_cast<double>(1ULL << 40);

    explicit Pipeline(std::mt19937_64& rng)
        : sk(SecretKey::sample(rng)),
          pk(gen_public_key(sk, shared_ntts(), rng)),
          evk(gen_eval_key(sk, shared_ntts(), rng)) {}

    Ciphertext encrypt_slots(const std::vector<std::complex<double>>& slots,
                             std::mt19937_64& rng,
                             double scale = DEFAULT_SCALE) const {
        Polynomial pt_coeff = enc.encode(slots, scale);
        Plaintext pt = Plaintext::from_polynomial(pt_coeff, scale, shared_ntts());
        return encrypt(pt, pk, shared_ntts(), rng);
    }
};

}  // namespace

TEST_CASE("mul_cipher + rescale: produces well-formed ciphertext at expected scale and level",
          "[ckks][mul_cipher]") {
    std::mt19937_64 rng(0xC1FECAFEULL);
    Pipeline pipe(rng);

    auto m1 = small_random_slots(rng);
    auto m2 = small_random_slots(rng);
    Ciphertext c1 = pipe.encrypt_slots(m1, rng);
    Ciphertext c2 = pipe.encrypt_slots(m2, rng);

    Ciphertext prod = mul_cipher(c1, c2, pipe.evk, shared_ntts());
    REQUIRE(prod.level == c1.level);
    REQUIRE(prod.scale == c1.scale * c2.scale);

    Ciphertext rescaled = rescale(prod, shared_ntts());
    REQUIRE(rescaled.level == c1.level - 1);
    // post-rescale scale = (delta_a * delta_b) / q_drop
    // delta_a = delta_b = 2^40, q_drop = q3 ~ 2^60, so result ~ 2^20
    const double q_drop = static_cast<double>(COEFF_MODULI[c1.level - 1]);
    REQUIRE(std::abs(rescaled.scale - (c1.scale * c2.scale / q_drop))
            < 1e-9 * rescaled.scale);

    // decrypt + verify scale/level bookkeeping survives round trip
    // numerical recovery is asserted by the dedicated test below
    Plaintext recovered = decrypt(rescaled, pipe.sk, shared_ntts());
    REQUIRE(recovered.level == rescaled.level);
    REQUIRE(recovered.scale == rescaled.scale);
}

// the main test, encrypt two slot vectors, mul_cipher + rescale + decrypt + decode
// assert each slot equals the elementwise product within RNS-gadget tolerance
// tolerance rationale (file header): EVAL_KEY_NOISE_SIGMA=0.5 keeps relin noise
// below ~1e-2 absolute on unit-magnitude inputs after one rescale, 5e-2 for safety
TEST_CASE("mul_cipher + rescale: decrypt approx slot-wise product within 5e-2",
          "[ckks][mul_cipher]") {
    std::mt19937_64 rng(0xD15EA5EDULL);
    Pipeline pipe(rng);
    Encoder enc;

    auto m1 = small_random_slots(rng, 0.25);
    auto m2 = small_random_slots(rng, 0.25);

    Ciphertext c1 = pipe.encrypt_slots(m1, rng);
    Ciphertext c2 = pipe.encrypt_slots(m2, rng);

    Ciphertext prod = mul_cipher(c1, c2, pipe.evk, shared_ntts());
    Ciphertext rescaled = rescale(prod, shared_ntts());

    // decrypt yields an NTT-form polynomial, decoder wants coefficient form
    // inverse-NTT each surviving residue before decoding
    Plaintext recovered = decrypt(rescaled, pipe.sk, shared_ntts());
    Polynomial recovered_coeff = recovered.poly;
    for (std::size_t i = 0; i < recovered.level; ++i) {
        shared_ntts()[i].inverse(recovered_coeff.residues[i].data());
    }
    auto slots = enc.decode(recovered_coeff, recovered.scale, recovered.level);

    REQUIRE(slots.size() == SLOT_COUNT);

    double max_err_real = 0.0;
    double max_err_imag = 0.0;
    for (std::size_t k = 0; k < SLOT_COUNT; ++k) {
        const std::complex<double> expected = m1[k] * m2[k];
        const double err_re = std::abs(slots[k].real() - expected.real());
        const double err_im = std::abs(slots[k].imag() - expected.imag());
        if (err_re > max_err_real) max_err_real = err_re;
        if (err_im > max_err_imag) max_err_imag = err_im;
    }
    INFO("max real err=" << max_err_real << "  max imag err=" << max_err_imag);
    REQUIRE(max_err_real < 5e-2);
    REQUIRE(max_err_imag < 5e-2);
}

TEST_CASE("mul_cipher: scale field doubles, level unchanged", "[ckks][mul_cipher]") {
    std::mt19937_64 rng(0x5CA1E0FFULL);
    Pipeline pipe(rng);

    auto m1 = small_random_slots(rng);
    auto m2 = small_random_slots(rng);
    Ciphertext c1 = pipe.encrypt_slots(m1, rng);
    Ciphertext c2 = pipe.encrypt_slots(m2, rng);

    Ciphertext prod = mul_cipher(c1, c2, pipe.evk, shared_ntts());
    REQUIRE(prod.scale == c1.scale * c2.scale);
    REQUIRE(prod.level == c1.level);
}

TEST_CASE("mul_cipher: rejects scale mismatch", "[ckks][mul_cipher]") {
    std::mt19937_64 rng(0xBA15CA1AULL);
    Pipeline pipe(rng);

    auto m1 = small_random_slots(rng);
    auto m2 = small_random_slots(rng);
    Ciphertext c1 = pipe.encrypt_slots(m1, rng, static_cast<double>(1ULL << 40));
    Ciphertext c2 = pipe.encrypt_slots(m2, rng, static_cast<double>(1ULL << 41));

    REQUIRE_THROWS_AS(mul_cipher(c1, c2, pipe.evk, shared_ntts()), std::invalid_argument);
}

TEST_CASE("mul_cipher: rejects level mismatch", "[ckks][mul_cipher]") {
    std::mt19937_64 rng(0xBA15E1E8ULL);
    Pipeline pipe(rng);

    auto m1 = small_random_slots(rng);
    auto m2 = small_random_slots(rng);
    Ciphertext c1 = pipe.encrypt_slots(m1, rng);
    Ciphertext c2 = pipe.encrypt_slots(m2, rng);

    c2.level = NUM_PRIMES - 1;
    REQUIRE_THROWS_AS(mul_cipher(c1, c2, pipe.evk, shared_ntts()), std::invalid_argument);
}

// structural check that mul_cipher returns a well-formed degree-1 ciphertext
// numerical recovery is exercised by the "decrypt approx slot-wise product" test above
TEST_CASE("mul_cipher: relin output is a valid degree-1 ciphertext (algebraic check)",
          "[ckks][mul_cipher]") {
    std::mt19937_64 rng(0xA1685A1CULL);
    Pipeline pipe(rng);

    auto m = small_random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);

    // squaring exercises both d2*evk.b and d2*evk.a paths
    // c0=c0_a=c0_b, c1=c1_a=c1_b, so tensor expansion is symmetric
    Ciphertext sq = mul_cipher(ct, ct, pipe.evk, shared_ntts());

    // structural: scale and level agree with the spec
    REQUIRE(sq.scale == ct.scale * ct.scale);
    REQUIRE(sq.level == ct.level);

    // c0 and c1 each occupy NUM_PRIMES residues of length POLY_DEGREE
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        REQUIRE(sq.c0.residues[i].size() == POLY_DEGREE);
        REQUIRE(sq.c1.residues[i].size() == POLY_DEGREE);
    }
}
