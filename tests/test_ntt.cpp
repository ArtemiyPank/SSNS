// ckks::NTT tests
//   ctor preconditions: power-of-two N, 2N divides p-1
//   round-trip: inverse(forward(a)) == a at N=16, 64, full N=8192 across all four primes
//   negacyclic-twist NTT: forward([c, 0, ...]) is NOT uniformly c (psi^i pre-mul kills the constant)
//     the round-trip still holds, which is what matters for poly multiply
//   linearity: forward(a + b) == forward(a) + forward(b) pointwise
//   scalar: forward(c * a) == c * forward(a)
//   multiplication parity: NTT negacyclic multiply matches naive O(N^2) reference at small N
#include <catch.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

using ssns::ckks::NTT;
using ssns::ckks::add_mod;
using ssns::ckks::sub_mod;
using ssns::ckks::mul_mod;
using ssns::ckks::COEFF_MODULI;

namespace {

// deterministic random vector of N coefficients reduced mod p
std::vector<std::uint64_t> rand_vec(std::size_t N, std::uint64_t p, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint64_t> v(N);
    for (auto& x : v) x = rng() % p;
    return v;
}

// O(N^2) reference for negacyclic poly multiplication mod (X^N+1, p)
//   c_k = sum_{i+j == k (mod N)} sign_{i,j} * a_i * b_j (mod p)
// sign_{i,j} = +1 if i+j < N, else -1 (i.e. p - x), since X^N = -1 in Z[X]/(X^N+1)
std::vector<std::uint64_t> naive_negacyclic_multiply(
    const std::vector<std::uint64_t>& a,
    const std::vector<std::uint64_t>& b,
    std::uint64_t p) {
    const std::size_t N = a.size();
    std::vector<std::uint64_t> c(N, 0);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            const std::uint64_t prod = mul_mod(a[i], b[j], p);
            if (i + j < N) {
                c[i + j] = add_mod(c[i + j], prod, p);
            } else {
                // X^(i+j) = X^(i+j-N) * X^N = -X^(i+j-N) in Z[X]/(X^N+1)
                c[i + j - N] = sub_mod(c[i + j - N], prod, p);
            }
        }
    }
    return c;
}

}  // namespace

TEST_CASE("NTT: constructor rejects non-power-of-two N", "[ckks][ntt]") {
    REQUIRE_THROWS_AS(NTT(COEFF_MODULI[0], 7), std::invalid_argument);
    REQUIRE_THROWS_AS(NTT(COEFF_MODULI[0], 100), std::invalid_argument);
}

TEST_CASE("NTT: constructor rejects p with 2N ∤ p-1", "[ckks][ntt]") {
    // 17 is prime, 2N = 256 does not divide 16 = 17 - 1
    REQUIRE_THROWS_AS(NTT(17, 128), std::invalid_argument);
}

TEST_CASE("NTT: round-trip identity at small N=16 across all primes",
          "[ckks][ntt]") {
    constexpr std::size_t N = 16;
    for (std::uint64_t p : COEFF_MODULI) {
        NTT ntt(p, N);
        auto a = rand_vec(N, p, /*seed=*/0xC0FFEEULL ^ p);
        auto a_copy = a;
        ntt.forward(a.data());
        ntt.inverse(a.data());
        REQUIRE(a == a_copy);
    }
}

TEST_CASE("NTT: round-trip identity at N=64 across all primes", "[ckks][ntt]") {
    constexpr std::size_t N = 64;
    for (std::uint64_t p : COEFF_MODULI) {
        NTT ntt(p, N);
        auto a = rand_vec(N, p, /*seed=*/0xDEADBEEFULL ^ p);
        auto a_copy = a;
        ntt.forward(a.data());
        ntt.inverse(a.data());
        REQUIRE(a == a_copy);
    }
}

TEST_CASE("NTT: round-trip identity at full N=8192 across all primes",
          "[ckks][ntt]") {
    constexpr std::size_t N = ssns::ckks::POLY_DEGREE;
    for (std::uint64_t p : COEFF_MODULI) {
        NTT ntt(p, N);
        auto a = rand_vec(N, p, /*seed=*/0x12345678ULL ^ p);
        auto a_copy = a;
        ntt.forward(a.data());
        ntt.inverse(a.data());
        REQUIRE(a == a_copy);
    }
}

TEST_CASE("NTT: linearity - forward(a+b) == forward(a) + forward(b)",
          "[ckks][ntt]") {
    constexpr std::size_t N = 64;
    const std::uint64_t p = COEFF_MODULI[0];
    NTT ntt(p, N);
    auto a = rand_vec(N, p, 1);
    auto b = rand_vec(N, p, 2);
    std::vector<std::uint64_t> sum(N);
    for (std::size_t i = 0; i < N; ++i) sum[i] = add_mod(a[i], b[i], p);
    auto fa = a;
    auto fb = b;
    auto fsum = sum;
    ntt.forward(fa.data());
    ntt.forward(fb.data());
    ntt.forward(fsum.data());
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(fsum[i] == add_mod(fa[i], fb[i], p));
    }
}

TEST_CASE("NTT: scalar consistency - forward(c·a) == c·forward(a)",
          "[ckks][ntt]") {
    constexpr std::size_t N = 64;
    const std::uint64_t p = COEFF_MODULI[1];
    NTT ntt(p, N);
    auto a = rand_vec(N, p, 7);
    const std::uint64_t c = 12345 % p;
    std::vector<std::uint64_t> ca(N);
    for (std::size_t i = 0; i < N; ++i) ca[i] = mul_mod(a[i], c, p);
    auto fa = a;
    auto fca = ca;
    ntt.forward(fa.data());
    ntt.forward(fca.data());
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(fca[i] == mul_mod(fa[i], c, p));
    }
}

TEST_CASE("NTT: multiplication parity vs naive negacyclic O(N²) reference",
          "[ckks][ntt]") {
    // N small enough that the naive reference runs instantly
    constexpr std::size_t N = 64;
    for (std::uint64_t p : COEFF_MODULI) {
        NTT ntt(p, N);
        auto a = rand_vec(N, p, 11);
        auto b = rand_vec(N, p, 13);
        const auto expected = naive_negacyclic_multiply(a, b, p);
        // NTT path: forward both, pointwise multiply, inverse
        auto fa = a;
        auto fb = b;
        ntt.forward(fa.data());
        ntt.forward(fb.data());
        std::vector<std::uint64_t> prod(N);
        for (std::size_t i = 0; i < N; ++i) prod[i] = mul_mod(fa[i], fb[i], p);
        ntt.inverse(prod.data());
        REQUIRE(prod == expected);
    }
}

TEST_CASE("NTT: zero polynomial is fixed by forward", "[ckks][ntt]") {
    constexpr std::size_t N = 64;
    const std::uint64_t p = COEFF_MODULI[0];
    NTT ntt(p, N);
    std::vector<std::uint64_t> a(N, 0);
    ntt.forward(a.data());
    for (auto x : a) REQUIRE(x == 0);
}
