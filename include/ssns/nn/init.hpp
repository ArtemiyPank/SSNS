// Random number generation + He initialisation for ReLU networks.
//
// Rng is a thin wrapper around std::mt19937_64 so call sites have a single
// type to pass around and seeds are deterministic.  Numerical parity with
// PyTorch is NOT bit-exact (different generators) — parity is established
// in Phase 2 by feeding identical W1/W2 into both implementations from
// dumped golden fixtures, not by replaying the RNG itself.
#ifndef SSNS_NN_INIT_HPP
#define SSNS_NN_INIT_HPP

#include <cstdint>
#include <random>

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

class Rng {
public:
    explicit Rng(std::uint64_t seed) : gen_(seed) {}
    std::mt19937_64& engine() noexcept { return gen_; }

private:
    std::mt19937_64 gen_;
};

// Fills a fresh Matrix with N(0, sqrt(2/fan_in)) — He init for ReLU networks.
[[nodiscard]] linalg::Matrix he_init(std::size_t rows, std::size_t cols,
                                     std::size_t fan_in, Rng& rng);

}  // namespace ssns::nn

#endif  // SSNS_NN_INIT_HPP
