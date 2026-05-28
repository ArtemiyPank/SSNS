// ckks::SecretKey, the RLWE secret in CKKS
// ternary polynomial s in {-1, 0, +1}[X]/(X^N+1), fixed Hamming weight H = 64
// stored in coefficient form (NOT NTT form), see secret_key.hpp
//
// coverage
//   sample produces exactly H=64 nonzero coeffs per RNS slot
//   nonzero coeffs are exactly +/-1 in centered representation
//   determinism: same RNG seed -> identical keys
//   sanity: different seeds -> different keys
#include <catch.hpp>

#include <ssns/ckks/params.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <cstddef>
#include <cstdint>
#include <random>

using ssns::ckks::COEFF_MODULI;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::SecretKey;

namespace {

constexpr std::size_t HAMMING_WEIGHT = 64;

// lift residue mod q to centered repr in [-(q-1)/2, q/2]
// for ternary secrets only -1, 0, +1 are legal centered values
std::int64_t to_centered(std::uint64_t r, std::uint64_t q) {
    return (r <= q / 2) ? static_cast<std::int64_t>(r)
                        : static_cast<std::int64_t>(r) - static_cast<std::int64_t>(q);
}

}  // namespace

TEST_CASE("SecretKey: sample produces exactly 64 nonzero coefficients",
          "[ckks][secret_key]") {
    std::mt19937_64 rng(0xA5A5A5A5ULL);
    SecretKey sk = SecretKey::sample(rng);

    // use slot 0, a coeff is "active" iff residue is nonzero
    // ternary secrets => all RNS slots agree on the support set
    std::size_t nonzero = 0;
    for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
        if (sk.s.residues[0][k] != 0) ++nonzero;
    }
    REQUIRE(nonzero == HAMMING_WEIGHT);

    // all RNS slots must agree on the support
    for (std::size_t i = 1; i < NUM_PRIMES; ++i) {
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            const bool z0 = (sk.s.residues[0][k] == 0);
            const bool zi = (sk.s.residues[i][k] == 0);
            REQUIRE(z0 == zi);
        }
    }
}

TEST_CASE("SecretKey: nonzero coefficients are ±1 in centered form",
          "[ckks][secret_key]") {
    std::mt19937_64 rng(0xDEADBEEFULL);
    SecretKey sk = SecretKey::sample(rng);
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const std::uint64_t q = COEFF_MODULI[i];
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            const std::uint64_t r = sk.s.residues[i][k];
            if (r == 0) continue;
            // either +1 (residue == 1) or -1 (residue == q - 1)
            const std::int64_t centered = to_centered(r, q);
            REQUIRE((centered == 1 || centered == -1));
        }
    }
}

TEST_CASE("SecretKey: same seed produces identical key",
          "[ckks][secret_key]") {
    std::mt19937_64 rng_a(0x12345678ULL);
    std::mt19937_64 rng_b(0x12345678ULL);
    SecretKey sk_a = SecretKey::sample(rng_a);
    SecretKey sk_b = SecretKey::sample(rng_b);
    REQUIRE(sk_a.s == sk_b.s);
}

TEST_CASE("SecretKey: different seeds produce different keys",
          "[ckks][secret_key]") {
    std::mt19937_64 rng_a(0x11111111ULL);
    std::mt19937_64 rng_b(0x22222222ULL);
    SecretKey sk_a = SecretKey::sample(rng_a);
    SecretKey sk_b = SecretKey::sample(rng_b);
    REQUIRE(sk_a.s != sk_b.s);
}
