#include <ssns/nn/init.hpp>

#include <cmath>

namespace ssns::nn {

linalg::Matrix he_init(std::size_t rows, std::size_t cols,
                       std::size_t fan_in, Rng& rng) {
    const double sigma = std::sqrt(2.0 / static_cast<double>(fan_in));
    std::normal_distribution<double> dist(0.0, sigma);
    linalg::Matrix M(rows, cols);
    double* d = M.data();
    const std::size_t n = M.size();
    auto& g = rng.engine();
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = dist(g);
    }
    return M;
}

}  // namespace ssns::nn
