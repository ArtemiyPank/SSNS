// ssns::ckks::Ciphertext: plain aggregate (two NTT-form polys + scale + level)
// locks in default values and level-field assignability
#include <catch.hpp>

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>

using ssns::ckks::Ciphertext;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::Polynomial;

TEST_CASE("Ciphertext: default-constructed has zero polys, level=NUM_PRIMES, scale=0",
          "[ckks][ciphertext]") {
    Ciphertext ct;
    REQUIRE(ct.c0 == Polynomial{});
    REQUIRE(ct.c1 == Polynomial{});
    REQUIRE(ct.level == NUM_PRIMES);
    REQUIRE(ct.scale == 0.0);
}

TEST_CASE("Ciphertext: level boundary values are honoured",
          "[ckks][ciphertext]") {
    Ciphertext ct;
    ct.level = 1;
    REQUIRE(ct.level == 1);
    ct.level = NUM_PRIMES;
    REQUIRE(ct.level == NUM_PRIMES);
}
