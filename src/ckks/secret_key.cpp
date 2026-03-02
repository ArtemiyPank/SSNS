// SecretKey — implementation.  See header for contracts and rationale.
#include <ssns/ckks/secret_key.hpp>

#include <ssns/ckks/params.hpp>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace ssns::ckks {

SecretKey SecretKey::sample(std::mt19937_64& rng) {
    SecretKey sk;
    // sk.s residues are zero-initialised by Polynomial's default ctor.

    // Partial Fisher-Yates: build [0, N) and shuffle the first H slots.
    // O(N) memory, O(H) RNG draws after the initial fill — H << N so
    // total cost is dominated by the initial std::iota.
    std::vector<std::size_t> indices(POLY_DEGREE);
    std::iota(indices.begin(), indices.end(), 0);
    for (std::size_t i = 0; i < SECRET_HAMMING_WEIGHT; ++i) {
        // Uniform j ∈ [i, N).  std::uniform_int_distribution is overkill
        // for a 13-bit range; modulo bias on a 64-bit MT output is
        // negligible (≪ 2^-50).
        const std::size_t span = POLY_DEGREE - i;
        const std::size_t j = i + static_cast<std::size_t>(rng() % span);
        std::swap(indices[i], indices[j]);
    }

    // For each chosen position, draw a sign and lift into RNS.
    for (std::size_t k = 0; k < SECRET_HAMMING_WEIGHT; ++k) {
        const std::size_t pos = indices[k];
        // Uniform sign: read the low bit of a fresh RNG draw.
        const bool positive = (rng() & 1ULL) == 1ULL;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            sk.s.residues[i][pos] = positive ? 1ULL : (q - 1ULL);
        }
    }
    return sk;
}

}  // namespace ssns::ckks
