// rng and weight init for relu nets
//
// he init: var = 2/fan_in for relu else half signal dies per layer
// factor 2 vs xavier 1 because relu zeros ~half of activations
//
// Rng wraps mt19937_64 so call sites pass one type and seeds stay deterministic
//
// not bit-exact with pytorch (different generators)
// phase 2 parity uses dumped golden weights not rng replay
#ifndef SSNS_NN_INIT_HPP
#define SSNS_NN_INIT_HPP

#include <cstdint>
#include <random>

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

// thin wrap around mt19937_64
// custom type lets us swap engine later (e.g. csprng) without touching call sites
class Rng {
public:
    // 64-bit seed mt19937_64 to avoid mt19937 cycle issues
    explicit Rng(std::uint64_t seed) : gen_(seed) {}
    // engine access for std::normal_distribution etc
    std::mt19937_64& engine() noexcept { return gen_; }

private:
    std::mt19937_64 gen_;
};

// he init var = 2/fan_in для relu иначе половина сигнала
// returns rows x cols matrix from N(0, sqrt(2/fan_in))
[[nodiscard]] linalg::Matrix he_init(std::size_t rows, std::size_t cols, std::size_t fan_in, Rng& rng);

}  // namespace ssns::nn

#endif  // SSNS_NN_INIT_HPP
