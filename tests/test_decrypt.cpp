// ssns::ckks::decrypt round-trip: encode -> Plaintext -> encrypt -> decrypt -> iNTT -> decode
// recovered slots match originals up to encryption noise (<< 1/scale at 2^40, sigma=3.2)
// well under 1e-3 absolute
#include <catch.hpp>

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/decrypt.hpp>
#include <ssns/ckks/encoder.hpp>
#include <ssns/ckks/encrypt.hpp>
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

namespace {

constexpr std::size_t SLOT_COUNT = POLY_DEGREE / 2;

const std::array<NTT, NUM_PRIMES>& shared_ntts() {
    static const std::array<NTT, NUM_PRIMES> instance = {{
        NTT(COEFF_MODULI[0], POLY_DEGREE),
        NTT(COEFF_MODULI[1], POLY_DEGREE),
        NTT(COEFF_MODULI[2], POLY_DEGREE),
        NTT(COEFF_MODULI[3], POLY_DEGREE),
    }};
    return instance;
}

// bring NTT-form poly back to coeff form so encoder.decode can consume it
Polynomial inverse_ntt(const Polynomial& p_ntt) {
    Polynomial out = p_ntt;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        shared_ntts()[i].inverse(out.residues[i].data());
    }
    return out;
}

}  // namespace

TEST_CASE("decrypt: round-trip via encode → encrypt → decrypt → decode within 1e-3",
          "[ckks][decrypt]") {
    std::mt19937_64 rng(0xDECAF1234ULL);
    SecretKey sk = SecretKey::sample(rng);
    PublicKey pk = gen_public_key(sk, shared_ntts(), rng);

    // random unit-magnitude slot vector
    Encoder enc;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<double>> slots(SLOT_COUNT);
    for (auto& s : slots) s = std::complex<double>(dist(rng), dist(rng));

    const double scale = static_cast<double>(1ULL << 40);  // 2^40
    Polynomial pt_coeff = enc.encode(slots, scale);
    Plaintext pt = Plaintext::from_polynomial(pt_coeff, scale, shared_ntts());

    Ciphertext ct = encrypt(pt, pk, shared_ntts(), rng);
    Plaintext recovered = decrypt(ct, sk, shared_ntts());
    REQUIRE(recovered.scale == scale);
    REQUIRE(recovered.level == NUM_PRIMES);

    // recovered plaintext is NTT-form; iNTT for decoder
    Polynomial recovered_coeff = inverse_ntt(recovered.poly);
    auto out = enc.decode(recovered_coeff, scale);

    double max_err = 0.0;
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) {
        max_err = std::max(max_err, std::abs(out[i] - slots[i]));
    }
    REQUIRE(max_err < 1e-3);
}

TEST_CASE("decrypt: zero plaintext round-trips to ~zero slots",
          "[ckks][decrypt]") {
    std::mt19937_64 rng(0xDEADC0DEULL);
    SecretKey sk = SecretKey::sample(rng);
    PublicKey pk = gen_public_key(sk, shared_ntts(), rng);

    Encoder enc;
    std::vector<std::complex<double>> zeros(SLOT_COUNT, {0.0, 0.0});
    const double scale = static_cast<double>(1ULL << 40);
    Polynomial pt_coeff = enc.encode(zeros, scale);
    Plaintext pt = Plaintext::from_polynomial(pt_coeff, scale, shared_ntts());

    Ciphertext ct = encrypt(pt, pk, shared_ntts(), rng);
    Plaintext recovered = decrypt(ct, sk, shared_ntts());

    Polynomial recovered_coeff = inverse_ntt(recovered.poly);
    auto out = enc.decode(recovered_coeff, scale);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) {
        REQUIRE(std::abs(out[i]) < 1e-3);
    }
}
