// nn::Rng wrapper + He init
// determinism via std::mt19937_64 seeded explicitly; same seed -> same sequence on every platform
// numerical parity with torch.Generator lives in phase-2 golden fixtures; this file pins C++ API contracts
#include <catch.hpp>

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/init.hpp>

using ssns::linalg::Matrix;
using ssns::nn::Rng;
using ssns::nn::he_init;

TEST_CASE("Rng: same seed -> same draw sequence", "[nn][init]") {
    Rng a(42);
    Rng b(42);
    auto Ma = he_init(4, 8, /*fan_in=*/4, a);
    auto Mb = he_init(4, 8, /*fan_in=*/4, b);
    for (std::size_t i = 0; i < Ma.rows(); ++i)
        for (std::size_t j = 0; j < Ma.cols(); ++j)
            REQUIRE(Ma(i, j) == Mb(i, j));
}

TEST_CASE("Rng: different seeds -> different sequences", "[nn][init]") {
    Rng a(1);
    Rng b(2);
    auto Ma = he_init(2, 3, /*fan_in=*/2, a);
    auto Mb = he_init(2, 3, /*fan_in=*/2, b);
    bool any_diff = false;
    for (std::size_t i = 0; i < Ma.size() && !any_diff; ++i)
        if (Ma.data()[i] != Mb.data()[i]) any_diff = true;
    REQUIRE(any_diff);
}

TEST_CASE("he_init: produces correct shape", "[nn][init]") {
    Rng g(0);
    auto M = he_init(64, 192, /*fan_in=*/64, g);
    REQUIRE(M.rows() == 64);
    REQUIRE(M.cols() == 192);
}

TEST_CASE("he_init: variance close to 2/fan_in for large matrices", "[nn][init]") {
    Rng g(2024);
    const std::size_t fan_in = 256;
    auto M = he_init(fan_in, 1024, fan_in, g);

    double mean = 0.0;
    for (std::size_t i = 0; i < M.size(); ++i) mean += M.data()[i];
    mean /= static_cast<double>(M.size());

    double var = 0.0;
    for (std::size_t i = 0; i < M.size(); ++i) {
        const double d = M.data()[i] - mean;
        var += d * d;
    }
    var /= static_cast<double>(M.size());

    // expected var = 2/fan_in = 2/256 = 0.0078125
    // N = 256*1024 = 262144; SE of var ~ var*sqrt(2/N) ~ 1e-5, allow 5% margin for stability
    const double expected = 2.0 / static_cast<double>(fan_in);
    REQUIRE(var == Approx(expected).epsilon(0.05));
    REQUIRE(std::abs(mean) < 0.005);
}
