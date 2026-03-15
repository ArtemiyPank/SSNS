// CKKS depth-0 linear operations.
//
// "Depth-0" here means no cipher×cipher multiplication — these ops do not
// consume a multiplicative level and require no relinearization.  The set:
//
//   add(ct1, ct2)            cipher + cipher       (pointwise NTT add)
//   sub(ct1, ct2)            cipher - cipher       (pointwise NTT sub)
//   add_plain(ct, pt)        cipher + plaintext    (c0 += pt.poly)
//   sub_plain(ct, pt)        cipher - plaintext    (c0 -= pt.poly)
//
// All inputs are assumed to be in NTT form (the storage convention used
// throughout the CKKS pipeline) — so pointwise add/sub in the frequency
// domain are equivalent to coefficient-domain polynomial add/sub.
//
// Preconditions
// -------------
// Both operands must agree on `scale` (encoder scaling factor) and `level`
// (active RNS primes).  Mismatch on either field throws `std::invalid_argument`.
// Tests use exact equality: in practice the encoder produces deterministic
// scale values so floating-point fuzz is not a concern.
#pragma once

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/plaintext.hpp>

namespace ssns::ckks {

Ciphertext add(const Ciphertext& a, const Ciphertext& b);
Ciphertext sub(const Ciphertext& a, const Ciphertext& b);
Ciphertext add_plain(const Ciphertext& ct, const Plaintext& pt);
Ciphertext sub_plain(const Ciphertext& ct, const Plaintext& pt);

}  // namespace ssns::ckks
