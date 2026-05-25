// ckks plaintext encoded message ready to be added to or multiplied by ciphertexts
//
// poly stored in ntt form
// cipher plain add and mul work elementwise in freq domain so pre transformed avoids ntt per op
//
// encoder produces coef form
// from_polynomial converts in place via supplied ntt instances
// keeping encoder ntt free is intentional so callers wanting just a polynomial form (tests debug) still get one
//
// scale and level
//   scale used when encoding needed by rescale to match plaintext scale
//   level number of active rns primes fresh pt uses NUM_PRIMES rescale drops one
#pragma once

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>

#include <array>
#include <cstddef>

namespace ssns::ckks {
    struct Plaintext {
        Polynomial poly; // ntt form
        double scale; // encoder scale
        std::size_t level; // active rns primes

        // wrap a coef form polynomial into a Plaintext
        // applies forward ntt per prime captures scale tags level (default NUM_PRIMES)
        static Plaintext from_polynomial(Polynomial coeff_form, double scale,
                                         const std::array<NTT, NUM_PRIMES> &ntts,
                                         std::size_t level = NUM_PRIMES);
    };
} // namespace ssns::ckks
