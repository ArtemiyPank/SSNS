// ssns::ckks::Backend: bundle of NTT cache + encoder + (sk, pk, evk) for FHE training
//
// determinism: same seed -> bit-identical (sk, pk, evk)
// fingerprint = sum of first 32 residue-0 coeffs (centered mod q_0); cheap, sensitive,
// slot-semantic free so it works on polys in NTT form too
#include <catch.hpp>

#include <ssns/ckks/backend.hpp>
#include <ssns/ckks/params.hpp>

#include <cstddef>
#include <cstdint>

using ssns::ckks::Backend;
using ssns::ckks::COEFF_MODULI;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::SCALE_BITS;

namespace {

// sum of first `window` residue-0 coeffs, centered into (-q/2, q/2]; cheap fingerprint
long long residue0_fingerprint(
    const std::array<std::vector<std::uint64_t>, NUM_PRIMES>& residues,
    std::size_t window = 32)
{
    const std::uint64_t q = COEFF_MODULI[0];
    const std::uint64_t half = q / 2;
    long long acc = 0;
    const std::size_t n = std::min(window, POLY_DEGREE);
    for (std::size_t k = 0; k < n; ++k) {
        const std::uint64_t v = residues[0][k];
        const long long centered = (v <= half)
            ? static_cast<long long>(v)
            : static_cast<long long>(v) - static_cast<long long>(q);
        acc += centered;
    }
    return acc;
}

}  // namespace

TEST_CASE("Backend::create: deterministic for same seed", "[ckks][backend]") {
    Backend a = Backend::create(0xC0FFEEULL);
    Backend b = Backend::create(0xC0FFEEULL);

    REQUIRE(a.scale == static_cast<double>(1ULL << SCALE_BITS));
    REQUIRE(b.scale == static_cast<double>(1ULL << SCALE_BITS));

    // sk fingerprint must match exactly; sparse ternary, coeffs lift to {-1,0,+1}
    REQUIRE(residue0_fingerprint(a.sk.s.residues)
            == residue0_fingerprint(b.sk.s.residues));

    // pk uniform mod q_i in NTT form; fingerprint via first 32 residue-0 coeffs of b = -a*s + e
    // same seed -> same draws -> bit-identical output
    REQUIRE(residue0_fingerprint(a.pk.b.residues)
            == residue0_fingerprint(b.pk.b.residues));
    REQUIRE(residue0_fingerprint(a.pk.a.residues)
            == residue0_fingerprint(b.pk.a.residues));

    // evk = NUM_PRIMES sub-keys; fingerprint each
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        REQUIRE(residue0_fingerprint(a.evk.sub_keys[i].b.residues)
                == residue0_fingerprint(b.evk.sub_keys[i].b.residues));
        REQUIRE(residue0_fingerprint(a.evk.sub_keys[i].a.residues)
                == residue0_fingerprint(b.evk.sub_keys[i].a.residues));
    }
}

TEST_CASE("Backend::create: different seeds produce different keys",
          "[ckks][backend]") {
    Backend a = Backend::create(1u);
    Backend b = Backend::create(2u);

    // collision chance ~2^-60 over 32 coeffs in a 60-bit prime; if this flakes, determinism is broken
    REQUIRE(residue0_fingerprint(a.pk.b.residues)
            != residue0_fingerprint(b.pk.b.residues));
}

TEST_CASE("Backend::create: NTTs and scale populated", "[ckks][backend]") {
    Backend b = Backend::create(7u);
    REQUIRE(b.scale == static_cast<double>(1ULL << SCALE_BITS));
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        REQUIRE(b.ntts[i].prime()  == COEFF_MODULI[i]);
        REQUIRE(b.ntts[i].degree() == POLY_DEGREE);
    }
}
