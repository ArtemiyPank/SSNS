// ckks rlwe encrypt public key form
//
// given pt and pk = (b, a) with b ~ -a*s + e produces
//     c0 =  b*v + e0 + pt    (mod q)
//     c1 =  a*v + e1         (mod q)
//
// where
//     v   sparse ternary {-1, 0, +1} density 1/2
//     e0 e1   rounded gaussian sigma = KEYGEN_NOISE_SIGMA = 3.2
//
// decrypt gives
//     c0 + c1*s = (b + a*s)*v + e0 + e1*s + pt = pt + small noise
//
// проверка: b = -a*s + e значит (b + a*s)*v = e*v
// noise малый так что round off на decode уносит его
//
// both c0 and c1 stored in ntt form so cipher arithmetic is pointwise
//
// determinism: output depends only on rng state at call time
#pragma once

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/public_key.hpp>

#include <array>
#include <random>

namespace ssns::ckks {

// encrypt pt under pk
// draws v e0 e1 from rng returns ct in ntt form
Ciphertext encrypt(const Plaintext& pt,
                   const PublicKey& pk,
                   const std::array<NTT, NUM_PRIMES>& ntts,
                   std::mt19937_64& rng);

}  // namespace ssns::ckks
