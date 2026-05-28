// ckks modular arithmetic primitives + four hard-coded CKKS primes (params.hpp)
// boundary cases on add/sub with 64-bit overflow, mul_mod vs 128-bit reference
// Fermat identity for inv_mod, primitive-root order checks
// validates COEFF_MODULI: prime, q == 1 (mod 2N) for negacyclic NTT, pairwise distinct (CRT)
#include <catch.hpp>

#include <random>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/params.hpp>

#include <cstdint>

using ssns::ckks::add_mod;
using ssns::ckks::sub_mod;
using ssns::ckks::mul_mod;
using ssns::ckks::pow_mod;
using ssns::ckks::inv_mod;
using ssns::ckks::primitive_2n_root;
using ssns::ckks::is_prime;

namespace {

// 128-bit reference for mul_mod cross-check, GCC unsigned __int128 extension
// independent of the header impl so the test catches regressions
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
std::uint64_t ref_mul_mod(std::uint64_t a, std::uint64_t b, std::uint64_t p) {
    using u128 = unsigned __int128;
    u128 prod = static_cast<u128>(a) * b;
    return static_cast<std::uint64_t>(prod % p);
}
#pragma GCC diagnostic pop

}  // namespace

TEST_CASE("add_mod: small operands behave like ordinary addition", "[ckks][modarith]") {
    REQUIRE(add_mod(0, 0, 7) == 0);
    REQUIRE(add_mod(3, 4, 7) == 0);
    REQUIRE(add_mod(3, 5, 7) == 1);
    REQUIRE(add_mod(6, 6, 7) == 5);
}

TEST_CASE("add_mod: handles operands close to p without wrap", "[ckks][modarith]") {
    // for 60-bit p, a + b <= 2*(p-1) < 2^61 so no 64-bit wrap, exercises s >= p branch only
    const std::uint64_t p = 1152921504606830593ULL;
    REQUIRE(add_mod(p - 1, p - 2, p) == p - 3);          // (p-1)+(p-2) == -3 == p-3
    REQUIRE(add_mod(p - 1, 1, p) == 0);                  // exact wrap to zero
    REQUIRE(add_mod(p / 2, p / 2 + 1, p) == 0);          // crosses p
}

TEST_CASE("add_mod: handles 64-bit unsigned wrap when both operands are < p",
          "[ckks][modarith]") {
    // contract: both inputs < p, but for p near 2^63 the raw sum (a+b) can wrap past 2^64
    // the `s < a` branch in add_mod corrects for this
    // 2^63 - 25 is a well-known prime, well-suited as a "would-wrap" prime
    constexpr std::uint64_t p = (std::uint64_t{1} << 63) - 25;
    REQUIRE(is_prime(p));
    const std::uint64_t a = p - 1;
    const std::uint64_t b = p - 1;
    // (p-1) + (p-1) = 2p - 2, in uint64 this wraps if p ~ 2^63
    // expected result: (-1 + -1) (mod p) = p - 2
    REQUIRE(add_mod(a, b, p) == p - 2);
}

TEST_CASE("sub_mod: handles a < b by adding p", "[ckks][modarith]") {
    REQUIRE(sub_mod(0, 0, 7) == 0);
    REQUIRE(sub_mod(2, 5, 7) == 4);   // (2 - 5) mod 7 = -3 mod 7 = 4
    REQUIRE(sub_mod(5, 2, 7) == 3);
    const std::uint64_t p = 1099511480321ULL;
    REQUIRE(sub_mod(0, 1, p) == p - 1);
}

TEST_CASE("mul_mod: matches 128-bit reference at a 60-bit prime", "[ckks][modarith]") {
    const std::uint64_t p = 1152921504606830593ULL;
    // hand-picked operands + some near p-1 to stress the wide product
    const std::uint64_t pairs[][2] = {
        {0, 0},
        {1, p - 1},
        {2, 3},
        {p / 2, p / 2},
        {p - 1, p - 1},
        {123456789ULL, 987654321ULL},
    };
    for (const auto& xy : pairs) {
        REQUIRE(mul_mod(xy[0], xy[1], p) == ref_mul_mod(xy[0], xy[1], p));
    }
}

TEST_CASE("mul_mod_psm60: parity with mul_mod across q0 / q3",
          "[ckks][modarith][psm]") {
    constexpr std::uint64_t Q0 = 1152921504606830593ULL;  // 2^60 - 16383
    constexpr std::uint64_t Q3 = 1152921504606748673ULL;  // 2^60 - 98303
    constexpr std::uint64_t C0 = 16383;
    constexpr std::uint64_t C3 = 98303;
    std::mt19937_64 rng(0xC0FFEE);
    for (int i = 0; i < 10000; ++i) {
        std::uint64_t a = rng() % Q0;
        std::uint64_t b = rng() % Q0;
        REQUIRE(ssns::ckks::mul_mod_psm60<Q0, C0>(a, b) == mul_mod(a, b, Q0));
        std::uint64_t a3 = rng() % Q3;
        std::uint64_t b3 = rng() % Q3;
        REQUIRE(ssns::ckks::mul_mod_psm60<Q3, C3>(a3, b3) == mul_mod(a3, b3, Q3));
    }
    // boundary cases
    REQUIRE(ssns::ckks::mul_mod_psm60<Q0, C0>(0, 0)              == 0);
    REQUIRE(ssns::ckks::mul_mod_psm60<Q0, C0>(Q0 - 1, Q0 - 1)    == mul_mod(Q0 - 1, Q0 - 1, Q0));
    REQUIRE(ssns::ckks::mul_mod_psm60<Q0, C0>(Q0 - 1, 1)         == Q0 - 1);
}

TEST_CASE("mul_mod_psm40: parity with mul_mod across q1 / q2",
          "[ckks][modarith][psm]") {
    constexpr std::uint64_t Q1 = 1099511480321ULL;        // 2^40 - 147455
    constexpr std::uint64_t Q2 = 1099510890497ULL;        // 2^40 - 737279
    constexpr std::uint64_t C1 = 147455;
    constexpr std::uint64_t C2 = 737279;
    std::mt19937_64 rng(0xBADC0FFEE);
    for (int i = 0; i < 10000; ++i) {
        std::uint64_t a = rng() % Q1;
        std::uint64_t b = rng() % Q1;
        REQUIRE(ssns::ckks::mul_mod_psm40<Q1, C1>(a, b) == mul_mod(a, b, Q1));
        std::uint64_t a2 = rng() % Q2;
        std::uint64_t b2 = rng() % Q2;
        REQUIRE(ssns::ckks::mul_mod_psm40<Q2, C2>(a2, b2) == mul_mod(a2, b2, Q2));
    }
    REQUIRE(ssns::ckks::mul_mod_psm40<Q1, C1>(Q1 - 1, Q1 - 1) == mul_mod(Q1 - 1, Q1 - 1, Q1));
}

TEST_CASE("pow_mod: basic identities", "[ckks][modarith]") {
    const std::uint64_t p = 1099511480321ULL;
    REQUIRE(pow_mod(0, 0, p) == 1);     // 0^0 = 1 by convention
    REQUIRE(pow_mod(7, 0, p) == 1);
    REQUIRE(pow_mod(7, 1, p) == 7);
    // Fermat's little theorem: a^(p-1) == 1 (mod p) for a coprime to p
    REQUIRE(pow_mod(2, p - 1, p) == 1);
    REQUIRE(pow_mod(123456789ULL, p - 1, p) == 1);
}

TEST_CASE("inv_mod: a * inv_mod(a) ≡ 1 (mod p)", "[ckks][modarith]") {
    const std::uint64_t p = 1099511480321ULL;
    const std::uint64_t cases[] = {2ULL, 3ULL, 7ULL, 11ULL, 1234567ULL, p - 1};
    for (std::uint64_t a : cases) {
        std::uint64_t inv = inv_mod(a, p);
        REQUIRE(mul_mod(a, inv, p) == 1);
    }
}

TEST_CASE("primitive_2n_root: ψ has true order 2N (not a proper divisor)",
          "[ckks][modarith]") {
    constexpr std::uint64_t two_n = 16384;  // matches POLY_DEGREE * 2
    for (std::uint64_t p : ssns::ckks::COEFF_MODULI) {
        std::uint64_t psi = primitive_2n_root(p, two_n);
        // psi^(2N) = 1 by definition
        REQUIRE(pow_mod(psi, two_n, p) == 1);
        // psi^N != 1, else psi is only N-th root, not 2N-th
        REQUIRE(pow_mod(psi, two_n / 2, p) != 1);
    }
}

TEST_CASE("is_prime: small known primes and composites", "[ckks][modarith]") {
    REQUIRE_FALSE(is_prime(0));
    REQUIRE_FALSE(is_prime(1));
    REQUIRE(is_prime(2));
    REQUIRE(is_prime(3));
    REQUIRE_FALSE(is_prime(4));
    REQUIRE(is_prime(5));
    REQUIRE_FALSE(is_prime(9));
    REQUIRE(is_prime(97));
    REQUIRE_FALSE(is_prime(100));
    // 561 is a Carmichael number, fools weak primality tests
    REQUIRE_FALSE(is_prime(561));
    // larger semiprimes and known primes
    REQUIRE(is_prime(1000000007ULL));
    REQUIRE_FALSE(is_prime(1000000007ULL * 3ULL));
}

TEST_CASE("CKKS params: every prime is genuinely prime", "[ckks][params]") {
    for (std::uint64_t q : ssns::ckks::COEFF_MODULI) {
        REQUIRE(is_prime(q));
    }
}

TEST_CASE("CKKS params: every prime ≡ 1 (mod 2N) - required for NTT",
          "[ckks][params]") {
    constexpr std::uint64_t two_n = ssns::ckks::TWO_N;
    static_assert(two_n == 2 * ssns::ckks::POLY_DEGREE);
    for (std::uint64_t q : ssns::ckks::COEFF_MODULI) {
        REQUIRE(q % two_n == 1);
    }
}

TEST_CASE("CKKS params: primes are pairwise distinct (CRT requirement)",
          "[ckks][params]") {
    const auto& q = ssns::ckks::COEFF_MODULI;
    for (std::size_t i = 0; i < q.size(); ++i) {
        for (std::size_t j = i + 1; j < q.size(); ++j) {
            REQUIRE(q[i] != q[j]);
        }
    }
}

TEST_CASE("CKKS params: bit-sizes match the declared profile [60, 40, 40, 60]",
          "[ckks][params]") {
    const auto& q = ssns::ckks::COEFF_MODULI;
    const auto& bits = ssns::ckks::COEFF_MOD_BIT_SIZES;
    for (std::size_t i = 0; i < q.size(); ++i) {
        // bit_width(q) == declared bit-size, q in [2^(b-1), 2^b)
        const int actual = 64 - __builtin_clzll(q[i]);
        REQUIRE(actual == bits[i]);
    }
}
