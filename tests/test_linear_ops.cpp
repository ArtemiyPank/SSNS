// ckks linear (depth-0) ops: add/sub (cipher and plain), mul_scalar, rescale
// every test runs the encode -> encrypt -> op -> decrypt -> decode pipeline
// tolerance: 1e-3 absolute on slot values
#include <catch.hpp>

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/decrypt.hpp>
#include <ssns/ckks/encoder.hpp>
#include <ssns/ckks/encrypt.hpp>
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
using ssns::ckks::NTT;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::Plaintext;
using ssns::ckks::Polynomial;
using ssns::ckks::PublicKey;
using ssns::ckks::SecretKey;
using ssns::ckks::add;
using ssns::ckks::add_plain;
using ssns::ckks::decrypt;
using ssns::ckks::encrypt;
using ssns::ckks::gen_public_key;
using ssns::ckks::mul_scalar;
using ssns::ckks::rescale;
using ssns::ckks::sub;
using ssns::ckks::sub_plain;

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

// inverse NTT for the first `level` residue rings
Polynomial inverse_ntt(const Polynomial& p_ntt, std::size_t level) {
    Polynomial out = p_ntt;
    for (std::size_t i = 0; i < level; ++i) {
        shared_ntts()[i].inverse(out.residues[i].data());
    }
    return out;
}

// SLOT_COUNT random complex slots in [-1,1] x [-1,1]
std::vector<std::complex<double>> random_slots(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<double>> z(SLOT_COUNT);
    for (auto& s : z) s = std::complex<double>(dist(rng), dist(rng));
    return z;
}

// encode + encrypt helper, fresh ciphertext at scale 2^40
struct Pipeline {
    Encoder enc;
    SecretKey sk;
    PublicKey pk;
    static constexpr double DEFAULT_SCALE = static_cast<double>(1ULL << 40);

    explicit Pipeline(std::mt19937_64& rng)
        : sk(SecretKey::sample(rng)),
          pk(gen_public_key(sk, shared_ntts(), rng)) {}

    Ciphertext encrypt_slots(const std::vector<std::complex<double>>& slots,
                             std::mt19937_64& rng,
                             double scale = DEFAULT_SCALE) const {
        Polynomial pt_coeff = enc.encode(slots, scale);
        Plaintext pt = Plaintext::from_polynomial(pt_coeff, scale, shared_ntts());
        return encrypt(pt, pk, shared_ntts(), rng);
    }

    Plaintext encode_plaintext(const std::vector<std::complex<double>>& slots,
                               double scale = DEFAULT_SCALE) const {
        Polynomial pt_coeff = enc.encode(slots, scale);
        return Plaintext::from_polynomial(pt_coeff, scale, shared_ntts());
    }

    std::vector<std::complex<double>> decrypt_decode(const Ciphertext& ct) const {
        Plaintext recovered = decrypt(ct, sk, shared_ntts());
        Polynomial coeff = inverse_ntt(recovered.poly, recovered.level);
        return enc.decode(coeff, recovered.scale, recovered.level);
    }
};

// max |a[i] - b[i]| over the shorter range
double max_abs_diff(const std::vector<std::complex<double>>& a,
                    const std::vector<std::complex<double>>& b) {
    double m = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

}  // namespace

TEST_CASE("add: decrypt(c1+c2) approx m1+m2 within 1e-3", "[ckks][linear_ops]") {
    std::mt19937_64 rng(0xADD0CAFEULL);
    Pipeline pipe(rng);

    auto m1 = random_slots(rng);
    auto m2 = random_slots(rng);
    Ciphertext ct1 = pipe.encrypt_slots(m1, rng);
    Ciphertext ct2 = pipe.encrypt_slots(m2, rng);

    Ciphertext sum = add(ct1, ct2);
    REQUIRE(sum.scale == ct1.scale);
    REQUIRE(sum.level == ct1.level);

    auto out = pipe.decrypt_decode(sum);
    std::vector<std::complex<double>> expected(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) expected[i] = m1[i] + m2[i];
    REQUIRE(max_abs_diff(out, expected) < 1e-3);
}

TEST_CASE("sub: decrypt(c1-c2) approx m1-m2 within 1e-3", "[ckks][linear_ops]") {
    std::mt19937_64 rng(0x5B0CAFE0ULL);
    Pipeline pipe(rng);

    auto m1 = random_slots(rng);
    auto m2 = random_slots(rng);
    Ciphertext ct1 = pipe.encrypt_slots(m1, rng);
    Ciphertext ct2 = pipe.encrypt_slots(m2, rng);

    Ciphertext diff = sub(ct1, ct2);
    REQUIRE(diff.scale == ct1.scale);
    REQUIRE(diff.level == ct1.level);

    auto out = pipe.decrypt_decode(diff);
    std::vector<std::complex<double>> expected(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) expected[i] = m1[i] - m2[i];
    REQUIRE(max_abs_diff(out, expected) < 1e-3);
}

TEST_CASE("add_plain: decrypt(c+pt) approx m+pt_msg within 1e-3", "[ckks][linear_ops]") {
    std::mt19937_64 rng(0xADD9914EULL);
    Pipeline pipe(rng);

    auto m = random_slots(rng);
    auto pt_msg = random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);
    Plaintext pt = pipe.encode_plaintext(pt_msg);

    Ciphertext sum = add_plain(ct, pt);
    REQUIRE(sum.scale == ct.scale);
    REQUIRE(sum.level == ct.level);

    auto out = pipe.decrypt_decode(sum);
    std::vector<std::complex<double>> expected(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) expected[i] = m[i] + pt_msg[i];
    REQUIRE(max_abs_diff(out, expected) < 1e-3);
}

TEST_CASE("sub_plain: decrypt(c-pt) approx m-pt_msg within 1e-3", "[ckks][linear_ops]") {
    std::mt19937_64 rng(0x5B9914EFULL);
    Pipeline pipe(rng);

    auto m = random_slots(rng);
    auto pt_msg = random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);
    Plaintext pt = pipe.encode_plaintext(pt_msg);

    Ciphertext diff = sub_plain(ct, pt);
    REQUIRE(diff.scale == ct.scale);
    REQUIRE(diff.level == ct.level);

    auto out = pipe.decrypt_decode(diff);
    std::vector<std::complex<double>> expected(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) expected[i] = m[i] - pt_msg[i];
    REQUIRE(max_abs_diff(out, expected) < 1e-3);
}

TEST_CASE("add: rejects mismatched scale", "[ckks][linear_ops]") {
    std::mt19937_64 rng(0xBA15CA1EULL);
    Pipeline pipe(rng);

    auto m1 = random_slots(rng);
    auto m2 = random_slots(rng);
    Ciphertext ct1 = pipe.encrypt_slots(m1, rng, static_cast<double>(1ULL << 40));
    Ciphertext ct2 = pipe.encrypt_slots(m2, rng, static_cast<double>(1ULL << 41));

    REQUIRE_THROWS_AS(add(ct1, ct2), std::invalid_argument);
}

TEST_CASE("mul_scalar: decrypt(c * 0.5) approx 0.5*m within 1e-3", "[ckks][linear_ops]") {
    std::mt19937_64 rng(0x5CA1A1F0ULL);
    Pipeline pipe(rng);

    auto m = random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);
    const double original_scale = ct.scale;

    Ciphertext scaled = mul_scalar(ct, 0.5);
    REQUIRE(scaled.level == ct.level);
    REQUIRE(scaled.scale > original_scale);

    auto out = pipe.decrypt_decode(scaled);
    std::vector<std::complex<double>> expected(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) expected[i] = 0.5 * m[i];
    REQUIRE(max_abs_diff(out, expected) < 1e-3);
}

TEST_CASE("mul_scalar: decrypt(c * 1/64) approx m/64 within 1e-3", "[ckks][linear_ops]") {
    std::mt19937_64 rng(0xBA7C5012ULL);
    Pipeline pipe(rng);

    auto m = random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);

    const double scalar = 1.0 / 64.0;
    Ciphertext scaled = mul_scalar(ct, scalar);
    REQUIRE(scaled.level == ct.level);

    auto out = pipe.decrypt_decode(scaled);
    std::vector<std::complex<double>> expected(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) expected[i] = scalar * m[i];
    REQUIRE(max_abs_diff(out, expected) < 1e-3);
}

TEST_CASE("mul_scalar: scale field bumps by 2^60", "[ckks][linear_ops]") {
    std::mt19937_64 rng(0x5CA1EBADULL);
    Pipeline pipe(rng);

    auto m = random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);
    const double bump = std::ldexp(1.0, 60);

    Ciphertext scaled = mul_scalar(ct, 0.5);
    REQUIRE(scaled.scale == ct.scale * bump);
    REQUIRE(scaled.level == ct.level);
}

TEST_CASE("add: rejects mismatched level", "[ckks][linear_ops]") {
    std::mt19937_64 rng(0xBA15E1E7ULL);
    Pipeline pipe(rng);

    auto m1 = random_slots(rng);
    auto m2 = random_slots(rng);
    Ciphertext ct1 = pipe.encrypt_slots(m1, rng);
    Ciphertext ct2 = pipe.encrypt_slots(m2, rng);

    // drop one level manually to force a mismatch
    ct2.level = NUM_PRIMES - 1;
    REQUIRE_THROWS_AS(add(ct1, ct2), std::invalid_argument);
}

TEST_CASE("mul_scalar then rescale: decrypt approx scalar*m within 1e-3",
          "[ckks][linear_ops][rescale]") {
    std::mt19937_64 rng(0x015CA1E1ULL);
    Pipeline pipe(rng);

    auto m = random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);
    const std::size_t initial_level = ct.level;
    const double initial_scale = ct.scale;

    // bump scale by 2^60, then drop the 60-bit prime to land near 2^40
    Ciphertext bumped = mul_scalar(ct, 0.5);
    Ciphertext rescaled = rescale(bumped, shared_ntts());

    REQUIRE(rescaled.level == initial_level - 1);
    // post-rescale scale should be close (not exactly) to the original
    // scale: 2^40 * 2^60 / q3, where q3 is ~2^60 a few thousand under
    const double expected_scale = bumped.scale / static_cast<double>(COEFF_MODULI[NUM_PRIMES - 1]);
    REQUIRE(rescaled.scale == expected_scale);
    REQUIRE(std::abs(rescaled.scale - initial_scale) / initial_scale < 1e-9);

    auto out = pipe.decrypt_decode(rescaled);
    std::vector<std::complex<double>> expected(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) expected[i] = 0.5 * m[i];
    REQUIRE(max_abs_diff(out, expected) < 1e-3);
}

TEST_CASE("rescale: rejects level=1", "[ckks][linear_ops][rescale]") {
    std::mt19937_64 rng(0xBA1E5E1FULL);
    Pipeline pipe(rng);

    auto m = random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);
    ct.level = 1;  // pretend we already rescaled down to one prime

    REQUIRE_THROWS_AS(rescale(ct, shared_ntts()), std::invalid_argument);
}

TEST_CASE("rescale: scale and level fields update correctly",
          "[ckks][linear_ops][rescale]") {
    std::mt19937_64 rng(0xF1E1D5C0ULL);
    Pipeline pipe(rng);

    auto m = random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);
    Ciphertext bumped = mul_scalar(ct, 1.0 / 64.0);
    const std::size_t expected_level = bumped.level - 1;
    const double expected_scale = bumped.scale / static_cast<double>(COEFF_MODULI[bumped.level - 1]);

    Ciphertext rescaled = rescale(bumped, shared_ntts());
    REQUIRE(rescaled.level == expected_level);
    REQUIRE(rescaled.scale == expected_scale);
}
