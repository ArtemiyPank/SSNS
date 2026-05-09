// SecretKey impl see header
#include <ssns/ckks/secret_key.hpp>

#include <ssns/ckks/params.hpp>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace ssns::ckks {

// sample fresh sparse ternary secret key
// pick H positions out of N via partial fisher yates then assign random sign
SecretKey SecretKey::sample(std::mt19937_64& rng) {
    SecretKey sk;
    // sk.s residues are zero initialised by Polynomial default ctor

    // partial fisher yates build [0, N) and shuffle the first H slots
    // O(N) memory O(H) rng draws after initial fill total cost dominated by std::iota
    std::vector<std::size_t> indices(POLY_DEGREE);
    std::iota(indices.begin(), indices.end(), 0);
    for (std::size_t i = 0; i < SECRET_HAMMING_WEIGHT; ++i) {
        // uniform j in [i, N)
        // std::uniform_int_distribution is overkill for a 13 bit range modulo bias on a 64 bit mt output is negligible
        const std::size_t span = POLY_DEGREE - i;
        const std::size_t j = i + static_cast<std::size_t>(rng() % span);
        std::swap(indices[i], indices[j]);
    }

    // for each chosen position draw sign and lift into rns
    for (std::size_t k = 0; k < SECRET_HAMMING_WEIGHT; ++k) {
        const std::size_t pos = indices[k];
        // uniform sign read low bit of fresh rng draw
        const bool positive = (rng() & 1ULL) == 1ULL;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            sk.s.residues[i][pos] = positive ? 1ULL : (q - 1ULL);
        }
    }
    return sk;
}

}  // namespace ssns::ckks
