// ckks::mul_plain, cipher * plaintext, depth-1
//
// math: result.{c0,c1} = pointwise_mul_ntt(ct.{c0,c1}, pt.poly)
//       result.scale = ct.scale * pt.scale
//       result.level = min(ct.level, pt.level)
//
// decoding trick: ct at 2^40 * pt at 2^40 -> ct at 2^80, rescale lands at ~2^20 (too low to decode)
// instead encode pt at scale = q3 (~2^60), mul -> 2^40 * q3 ~ 2^100, rescale by q3 -> clean 2^40
//
// |m_i|, |pt_i| <= 0.5 keeps product in [-0.25, 0.25], easy to verify at 1e-3
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
using ssns::ckks::decrypt;
using ssns::ckks::encrypt;
using ssns::ckks::gen_public_key;
using ssns::ckks::mul_plain;
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

// inverse NTT for the first `level` residue rings
Polynomial inverse_ntt(const Polynomial& p_ntt, std::size_t level) {
    Polynomial out = p_ntt;
    for (std::size_t i = 0; i < level; ++i) {
        shared_ntts()[i].inverse(out.residues[i].data());
    }
    return out;
}

// SLOT_COUNT random complex slots in [-0.5, 0.5]
std::vector<std::complex<double>> small_random_slots(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<std::complex<double>> z(SLOT_COUNT);
    for (auto& s : z) s = std::complex<double>(dist(rng), dist(rng));
    return z;
}

// holds sk, pk, encoder, default scale 2^40
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

TEST_CASE("mul_plain: decrypt(c * pt) approx m * pt_msg within 1e-3 (after rescale)",
          "[ckks][mul_plain]") {
    std::mt19937_64 rng(0xCAFE9914ULL);
    Pipeline pipe(rng);

    auto m = small_random_slots(rng);
    auto pt_msg = small_random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);  // scale 2^40

    // encode pt at scale = q3 (~2^60), mul_plain -> 2^40 * q3, rescale -> 2^40 (clean)
    const double pt_scale = static_cast<double>(COEFF_MODULI[NUM_PRIMES - 1]);
    Plaintext pt = pipe.encode_plaintext(pt_msg, pt_scale);

    Ciphertext prod = mul_plain(ct, pt);
    REQUIRE(prod.level == ct.level);  // mul_plain does not drop level

    Ciphertext rescaled = rescale(prod, shared_ntts());
    REQUIRE(rescaled.level == ct.level - 1);

    // post-rescale scale ~ ct.scale * pt.scale / q3 = 2^40 * q3 / q3 = 2^40
    REQUIRE(std::abs(rescaled.scale - ct.scale) / ct.scale < 1e-9);

    auto out = pipe.decrypt_decode(rescaled);
    std::vector<std::complex<double>> expected(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) expected[i] = m[i] * pt_msg[i];
    REQUIRE(max_abs_diff(out, expected) < 1e-3);
}

TEST_CASE("mul_plain: scale field bumps by pt.scale", "[ckks][mul_plain]") {
    std::mt19937_64 rng(0xBEEFCAFEULL);
    Pipeline pipe(rng);

    auto m = small_random_slots(rng);
    auto pt_msg = small_random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);
    Plaintext pt = pipe.encode_plaintext(pt_msg);  // scale 2^40

    Ciphertext prod = mul_plain(ct, pt);
    REQUIRE(prod.scale == ct.scale * pt.scale);
    REQUIRE(prod.level == ct.level);
}

TEST_CASE("mul_plain: rejects level mismatch", "[ckks][mul_plain]") {
    std::mt19937_64 rng(0xBA15FA17ULL);
    Pipeline pipe(rng);

    auto m = small_random_slots(rng);
    auto pt_msg = small_random_slots(rng);
    Ciphertext ct = pipe.encrypt_slots(m, rng);
    Plaintext pt = pipe.encode_plaintext(pt_msg);

    // manually drop pt level to force a mismatch
    pt.level = NUM_PRIMES - 1;
    REQUIRE_THROWS_AS(mul_plain(ct, pt), std::invalid_argument);
}
