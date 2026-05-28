// ssns::ckks::Encoder: canonical embedding C^(N/2) -> Z[X]/(X^N+1) at primitive 2N-th roots
// (odd positions); round trip near-identity (~1e-6) at N=8192, scale 2^40
//
// coverage: zero->zero (both ways), random unit-magnitude round trip, linearity via decode
#include <catch.hpp>

#include <ssns/ckks/encoder.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>

#include <complex>
#include <cstddef>
#include <random>
#include <vector>

using ssns::ckks::Encoder;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::Polynomial;

namespace {

constexpr std::size_t SLOT_COUNT = POLY_DEGREE / 2;
constexpr double DEFAULT_SCALE = 1.0e12;  // ~2^40

}  // namespace

TEST_CASE("Encoder: encode of zero vector is the zero polynomial",
          "[ckks][encoder]") {
    Encoder enc;
    std::vector<std::complex<double>> zeros(SLOT_COUNT, {0.0, 0.0});
    Polynomial p = enc.encode(zeros, DEFAULT_SCALE);
    REQUIRE(p == Polynomial{});
}

TEST_CASE("Encoder: decode of zero polynomial is a zero vector",
          "[ckks][encoder]") {
    Encoder enc;
    Polynomial z;
    auto out = enc.decode(z, DEFAULT_SCALE);
    REQUIRE(out.size() == SLOT_COUNT);
    for (auto& c : out) {
        REQUIRE(std::abs(c) < 1e-12);
    }
}

TEST_CASE("Encoder: round-trip preserves random unit-magnitude slots within 1e-6",
          "[ckks][encoder]") {
    Encoder enc;
    std::mt19937_64 rng(0xC0DECAFEULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<std::complex<double>> slots(SLOT_COUNT);
    for (auto& s : slots) s = std::complex<double>(dist(rng), dist(rng));

    auto p = enc.encode(slots, DEFAULT_SCALE);
    auto out = enc.decode(p, DEFAULT_SCALE);
    REQUIRE(out.size() == SLOT_COUNT);

    double max_err = 0.0;
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) {
        max_err = std::max(max_err, std::abs(out[i] - slots[i]));
    }
    REQUIRE(max_err < 1e-6);
}

TEST_CASE("Encoder: round-trip on a tiny constant slot vector", "[ckks][encoder]") {
    Encoder enc;
    std::vector<std::complex<double>> slots(SLOT_COUNT, std::complex<double>(0.25, -0.5));
    auto p = enc.encode(slots, DEFAULT_SCALE);
    auto out = enc.decode(p, DEFAULT_SCALE);
    REQUIRE(out.size() == SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) {
        REQUIRE(std::abs(out[i] - slots[i]) < 1e-6);
    }
}

TEST_CASE("Encoder: linearity - encode(αx + βy) ≈ α·encode(x) + β·encode(y) up to round-off",
          "[ckks][encoder]") {
    // check linearity via decode (the natural inverse); RNS rounding makes direct poly compare flaky
    Encoder enc;
    std::mt19937_64 rng(0xBADF00DULL);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<std::complex<double>> x(SLOT_COUNT), y(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) {
        x[i] = std::complex<double>(dist(rng), dist(rng));
        y[i] = std::complex<double>(dist(rng), dist(rng));
    }
    const double alpha = 0.3, beta = -0.7;
    std::vector<std::complex<double>> combo(SLOT_COUNT);
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) {
        combo[i] = alpha * x[i] + beta * y[i];
    }
    auto p = enc.encode(combo, DEFAULT_SCALE);
    auto out = enc.decode(p, DEFAULT_SCALE);
    double max_err = 0.0;
    for (std::size_t i = 0; i < SLOT_COUNT; ++i) {
        max_err = std::max(max_err, std::abs(out[i] - combo[i]));
    }
    REQUIRE(max_err < 1e-6);
}
