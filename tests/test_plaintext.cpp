// ckks::Plaintext wraps a coefficient-form polynomial in NTT form
// captures the encoding scale + active RNS level
// inverse NTT residue-by-residue must recover the original coeff form (round-trip)
#include <catch.hpp>

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/poly.hpp>

#include <array>
#include <cstdint>
#include <vector>

using ssns::ckks::COEFF_MODULI;
using ssns::ckks::NTT;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::Plaintext;
using ssns::ckks::Polynomial;

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

}  // namespace

TEST_CASE("Plaintext: from_polynomial round-trips through NTT",
          "[ckks][plaintext]") {
    // small coefficient-form polynomial
    std::vector<std::int64_t> coeffs = {1, -2, 3, 0, 5};
    Polynomial p_coeff = Polynomial::from_coeffs(coeffs);

    Plaintext pt = Plaintext::from_polynomial(p_coeff, 1.0e12, shared_ntts());

    // invert each residue back to coefficient form, then compare
    Polynomial round_trip = pt.poly;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        shared_ntts()[i].inverse(round_trip.residues[i].data());
    }
    REQUIRE(round_trip == p_coeff);
}

TEST_CASE("Plaintext: scale and level fields are captured correctly",
          "[ckks][plaintext]") {
    Polynomial zeros;  // all-zero polynomial
    const double s = 2.0e12;
    Plaintext pt = Plaintext::from_polynomial(zeros, s, shared_ntts());

    REQUIRE(pt.scale == s);
    REQUIRE(pt.level == NUM_PRIMES);

    // explicit level override should be honoured
    Plaintext pt2 = Plaintext::from_polynomial(zeros, s, shared_ntts(), 2);
    REQUIRE(pt2.level == 2);
}
