// Plaintext — implementation.  See header for the storage convention.
#include <ssns/ckks/plaintext.hpp>

namespace ssns::ckks {

Plaintext Plaintext::from_polynomial(Polynomial coeff_form,
                                     double scale,
                                     const std::array<NTT, NUM_PRIMES>& ntts,
                                     std::size_t level) {
    // Forward NTT per prime — mutates `coeff_form` in place, then we
    // move it into the result.
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ntts[i].forward(coeff_form.residues[i].data());
    }
    return Plaintext{std::move(coeff_form), scale, level};
}

}  // namespace ssns::ckks
