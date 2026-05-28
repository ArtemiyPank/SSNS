// ckks::Polynomial: RNS element of Z[X]/(X^N+1) at N = POLY_DEGREE = 8192
//
// coverage
//   from_coeffs lifts small signed ints to RNS
//   add/sub: zero is identity, (a + b) - b = a
//   multiply by zero is zero
//   X * X^(N-1) = -1 (negacyclic ring identity)
//   multiply matches naive reference at sparse / low-degree inputs (cheap O(d^2))
#include <catch.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

using ssns::ckks::COEFF_MODULI;
using ssns::ckks::NTT;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::Polynomial;

namespace {

// shared NTT tables across all primes, twiddle-table build at N=8192 is expensive
const std::array<NTT, NUM_PRIMES>& shared_ntts() {
    static const std::array<NTT, NUM_PRIMES> instance = {{
        NTT(COEFF_MODULI[0], POLY_DEGREE),
        NTT(COEFF_MODULI[1], POLY_DEGREE),
        NTT(COEFF_MODULI[2], POLY_DEGREE),
        NTT(COEFF_MODULI[3], POLY_DEGREE),
    }};
    return instance;
}

// O(N^2) negacyclic multiply at full ring degree, single prime
// used to verify Polynomial::multiply on sparse inputs, cost is O(nz^2) for nz non-zero terms
std::vector<std::uint64_t> naive_negacyclic_multiply(
    const std::vector<std::uint64_t>& a,
    const std::vector<std::uint64_t>& b,
    std::uint64_t p) {
    const std::size_t N = a.size();
    std::vector<std::uint64_t> c(N, 0);
    for (std::size_t i = 0; i < N; ++i) {
        if (a[i] == 0) continue;
        for (std::size_t j = 0; j < N; ++j) {
            if (b[j] == 0) continue;
            const std::uint64_t prod = ssns::ckks::mul_mod(a[i], b[j], p);
            if (i + j < N) {
                c[i + j] = ssns::ckks::add_mod(c[i + j], prod, p);
            } else {
                c[i + j - N] = ssns::ckks::sub_mod(c[i + j - N], prod, p);
            }
        }
    }
    return c;
}

}  // namespace

TEST_CASE("Polynomial: default constructor is the zero polynomial",
          "[ckks][poly]") {
    Polynomial z;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        REQUIRE(z.residues[i].size() == POLY_DEGREE);
        for (auto x : z.residues[i]) REQUIRE(x == 0);
    }
}

TEST_CASE("Polynomial::from_coeffs lifts small integers correctly",
          "[ckks][poly]") {
    const std::vector<std::int64_t> coeffs = {1, 2, -3, 0, 7, -100, 0, 0};
    auto p = Polynomial::from_coeffs(coeffs);
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const std::uint64_t q = COEFF_MODULI[i];
        REQUIRE(p.residues[i].size() == POLY_DEGREE);
        REQUIRE(p.residues[i][0] == 1);
        REQUIRE(p.residues[i][1] == 2);
        REQUIRE(p.residues[i][2] == q - 3);   // -3 == q - 3
        REQUIRE(p.residues[i][3] == 0);
        REQUIRE(p.residues[i][4] == 7);
        REQUIRE(p.residues[i][5] == q - 100); // -100 == q - 100
        // trailing positions are zero
        for (std::size_t k = coeffs.size(); k < POLY_DEGREE; ++k) {
            REQUIRE(p.residues[i][k] == 0);
        }
    }
}

TEST_CASE("Polynomial: zero is the identity for add", "[ckks][poly]") {
    auto a = Polynomial::from_coeffs({1, 2, 3, 4, 5});
    Polynomial z;
    auto a_copy = a;
    a.add_inplace(z);
    REQUIRE(a == a_copy);
}

TEST_CASE("Polynomial: (a + b) - b == a", "[ckks][poly]") {
    auto a = Polynomial::from_coeffs({1, 2, -3, 4, -5, 6});
    auto b = Polynomial::from_coeffs({100, -50, 25, -12, 6, -3});
    auto a_copy = a;
    a.add_inplace(b);
    a.sub_inplace(b);
    REQUIRE(a == a_copy);
}

TEST_CASE("Polynomial: multiply by zero is zero", "[ckks][poly]") {
    auto a = Polynomial::from_coeffs({1, 2, 3, 4, 5, 6, 7});
    Polynomial z;
    auto prod = Polynomial::multiply(a, z, shared_ntts());
    REQUIRE(prod == Polynomial());
}

TEST_CASE("Polynomial: X · X^(N-1) = -1 (negacyclic ring identity)",
          "[ckks][poly]") {
    // a = X = [0, 1, 0, ..., 0]
    Polynomial a;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) a.residues[i][1] = 1;
    // b = X^(N-1) = [0, ..., 0, 1]
    Polynomial b;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) b.residues[i][POLY_DEGREE - 1] = 1;
    auto prod = Polynomial::multiply(a, b, shared_ntts());
    // expected: c = -1 in Z_q[X]/(X^N+1), so c_0 = q - 1 and all higher = 0
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const std::uint64_t q = COEFF_MODULI[i];
        REQUIRE(prod.residues[i][0] == q - 1);
        for (std::size_t k = 1; k < POLY_DEGREE; ++k) {
            REQUIRE(prod.residues[i][k] == 0);
        }
    }
}

TEST_CASE("Polynomial: multiply matches naive at dense low-degree inputs (a few hundred terms)",
          "[ckks][poly]") {
    // ~300 contiguous random coefficients per operand
    // 2*300 = 600 << N = 8192, so few wrap-around terms (i + j >= N path lightly exercised)
    // naive O(d^2) reference is ~90k mul_mod per prime, cheap
    constexpr std::size_t D = 300;
    std::mt19937_64 rng(0xFEEDF00DULL);
    auto fill_dense = [&](Polynomial& p) {
        for (std::size_t k = 0; k < D; ++k) {
            const std::int64_t v = static_cast<std::int64_t>((rng() % 4000) - 2000);
            for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
                const std::uint64_t q = COEFF_MODULI[i];
                std::uint64_t r;
                if (v >= 0) r = static_cast<std::uint64_t>(v) % q;
                else        r = q - (static_cast<std::uint64_t>(-v) % q);
                p.residues[i][k] = r;
            }
        }
    };
    Polynomial a, b;
    fill_dense(a);
    fill_dense(b);
    auto prod = Polynomial::multiply(a, b, shared_ntts());
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const auto expected = naive_negacyclic_multiply(
            a.residues[i], b.residues[i], COEFF_MODULI[i]);
        REQUIRE(prod.residues[i] == expected);
    }
}

TEST_CASE("Polynomial: multiply matches naive negacyclic at sparse low-degree inputs",
          "[ckks][poly]") {
    // 8 random non-zero terms each, O(64) naive products per prime
    // negligible vs the N=8192 NTT work
    constexpr std::size_t NZ = 8;
    std::mt19937_64 rng(0xCAFEBABEULL);
    auto fill_sparse = [&](Polynomial& p) {
        for (std::size_t s = 0; s < NZ; ++s) {
            const std::size_t idx = rng() % POLY_DEGREE;
            const std::int64_t v = static_cast<std::int64_t>((rng() % 2000) - 1000);
            for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
                const std::uint64_t q = COEFF_MODULI[i];
                std::uint64_t r;
                if (v >= 0) r = static_cast<std::uint64_t>(v) % q;
                else        r = q - (static_cast<std::uint64_t>(-v) % q);
                p.residues[i][idx] = r;
            }
        }
    };
    Polynomial a, b;
    fill_sparse(a);
    fill_sparse(b);
    auto prod = Polynomial::multiply(a, b, shared_ntts());
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const auto expected = naive_negacyclic_multiply(
            a.residues[i], b.residues[i], COEFF_MODULI[i]);
        REQUIRE(prod.residues[i] == expected);
    }
}
