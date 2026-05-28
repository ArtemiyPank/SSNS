// hidden microbenchmark for ssns::linalg::matmul, tagged [.benchmark]
// invoke explicitly: ./ssns_tests "[matmul][benchmark]"
// reports wall-clock + GFLOPS at 1024x1024, asserts a generous timing cap to catch gross regressions
#include <catch.hpp>

#include <ssns/linalg/matrix.hpp>

#include <chrono>
#include <cstdio>
#include <random>

using ssns::linalg::Matrix;

namespace {

// n x n matrix filled from U(-1, 1) using mt19937(seed)
Matrix random_square(std::size_t n, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix m(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            m(i, j) = dist(gen);
    return m;
}

}  // namespace

TEST_CASE("matmul benchmark @ 1024x1024", "[.benchmark][matmul]") {
    constexpr std::size_t N = 1024;
    auto A = random_square(N, 11u);
    auto B = random_square(N, 13u);

    // warm-up, first call may include cache fill or thread spin-up
    auto C = ssns::linalg::matmul(A, B);

    constexpr int reps = 3;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        C = ssns::linalg::matmul(A, B);
    }
    auto t1 = std::chrono::steady_clock::now();

    const double secs =
        std::chrono::duration<double>(t1 - t0).count() / reps;
    const double flops = 2.0 * static_cast<double>(N) * N * N;
    const double gflops = flops / secs * 1e-9;

    std::printf("\n[matmul-bench] N=%zu  %.3f s/iter  %.2f GFLOPS\n",
                N, secs, gflops);

    // sanity bound: > 10 s/iter at 1024^3 means kernel is broken (e.g. scalar fallback)
    // real targets are ~0.05 to 0.2 s on this hardware, 10 s leaves 50-200x slack
    REQUIRE(secs < 10.0);

    // spot-check a few cells with manual dot product to catch threading data races
    for (std::size_t i : {0u, 17u, 333u, 999u}) {
        for (std::size_t j : {0u, 5u, 512u, 1023u}) {
            double acc = 0.0;
            for (std::size_t k = 0; k < N; ++k) acc += A(i, k) * B(k, j);
            REQUIRE(std::fabs(C(i, j) - acc) < 1e-9);
        }
    }
}
