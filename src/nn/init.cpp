#include <ssns/nn/init.hpp>

#include <cmath>

namespace ssns::nn {
    // he init
    // var = 2/fan_in для relu иначе половина сигнала умирает на каждом слое
    // factor 2 vs xavier 1 because relu zeros ~half of activations
    //
    // returns rows x cols matrix from N(0, sqrt(2/fan_in))
    // std::normal_distribution not bit-exact across stdlibs but ok
    // phase 2 parity uses dumped golden weights not rng replay
    linalg::Matrix he_init(std::size_t rows, std::size_t cols,
                           std::size_t fan_in, Rng &rng) {
        // sigma = sqrt(var) std-параметр normal_distribution принимает std не var
        // sigma = sqrt(2/fan_in) держит Var(W @ x) = 1 после relu
        const double sigma = std::sqrt(2.0 / static_cast<double>(fan_in));
        std::normal_distribution<double> dist(0.0, sigma);
        linalg::Matrix M(rows, cols);
        double *d = M.data();
        const std::size_t n = M.size();
        auto &g = rng.engine();
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = dist(g);
        }
        return M;
    }
} // namespace ssns::nn
