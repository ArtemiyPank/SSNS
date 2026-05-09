// ckks ciphertext
// holds c0 c1 polys forming an rlwe pair
//
// both stored in ntt form
// decrypt does c0 + c1*s (mod q)
// keeping pre transformed avoids ntt per op
//
// scale and level
//   scale is the float scaling carried by the message
//   add needs both ciphertexts to agree on scale
//   cipher x cipher squares it rescale divides by one prime
//   level is number of active rns primes
//   fresh ct uses all NUM_PRIMES rescale drops one
//   ops should ignore residue slots above level
//   we do NOT truncate inside Polynomial that is the rescale layer job
#pragma once

#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>

#include <cstddef>

namespace ssns::ckks {

struct Ciphertext {
    Polynomial c0;       // ntt form
    Polynomial c1;       // ntt form
    double scale{0.0};   // message scaling
    std::size_t level{NUM_PRIMES};  // active rns primes
};

}  // namespace ssns::ckks
