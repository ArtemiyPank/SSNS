// ckks depth 0 linear ops
// no cipher x cipher mul here so no relin
//
//   add(ct1, ct2)            cipher + cipher       (pointwise ntt add)
//   sub(ct1, ct2)            cipher - cipher       (pointwise ntt sub)
//   add_plain(ct, pt)        cipher + plaintext    (c0 += pt.poly)
//   sub_plain(ct, pt)        cipher - plaintext    (c0 -= pt.poly)
//   mul_scalar(ct, s)        cipher * real scalar  (per prime mod mul)
//
// all inputs ntt form so pointwise is equivalent to coef domain poly ops
//
// preconditions
// add/sub/add_plain/sub_plain need scale and level to match
// mismatch throws std::invalid_argument
//
// scale arithmetic
// add/sub/add_plain/sub_plain preserve scale and level
//
// mul_scalar bumps scale
// real scalar is encoded as 60 bit int k = round(scalar * 2^60)
// so the multiply is exact mod each prime
//   out.scale = ct.scale * 2^60
//   out.level = ct.level
//
// the bump factor is exactly 2^60 (not the prime itself)
// rescale fixes the mismatch by dividing by the dropped prime
#pragma once

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/eval_key.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>

#include <array>

namespace ssns::ckks {
    // cipher + cipher needs scale and level to match
    Ciphertext add(const Ciphertext &a, const Ciphertext &b);

    // cipher - cipher needs scale and level to match
    Ciphertext sub(const Ciphertext &a, const Ciphertext &b);

    // cipher + plaintext needs scale and level to match
    Ciphertext add_plain(const Ciphertext &ct, const Plaintext &pt);

    // cipher - plaintext needs scale and level to match
    Ciphertext sub_plain(const Ciphertext &ct, const Plaintext &pt);

    // multiply ciphertext by real scalar
    // output scale is always ct.scale * 2^60
    // must be rescaled before adding to a non bumped ciphertext
    Ciphertext mul_scalar(const Ciphertext &ct, double scalar);

    // depth 1 multiplications
    //
    // mul_plain(ct, pt) cipher x plaintext
    // pointwise mul on c0 and c1
    //   out.c0    = pointwise_mul_ntt(ct.c0, pt.poly)
    //   out.c1    = pointwise_mul_ntt(ct.c1, pt.poly)
    //   out.scale = ct.scale * pt.scale
    //   out.level = min(ct.level, pt.level)
    //
    // precondition ct.level == pt.level
    // scale does NOT need to match
    Ciphertext mul_plain(const Ciphertext &ct, const Plaintext &pt);

    // mul_cipher(a, b, evk, ntts) cipher x cipher with rns gadget relin
    //
    // tensor expansion in ntt form
    //     d0 = a.c0 * b.c0
    //     d1 = a.c0 * b.c1 + a.c1 * b.c0
    //     d2 = a.c1 * b.c1
    //
    // relin back to degree 1 using rns gadget eval key
    //     for i in 0..NUM_PRIMES
    //         d2_at_i = lift d2 slot i residue forward ntt
    //         c0_relin += d2_at_i * sub_keys[i].b
    //         c1_relin += d2_at_i * sub_keys[i].a
    //     c0_relin += d0
    //     c1_relin += d1
    //
    //     out.scale = a.scale * b.scale
    //     out.level = a.level
    //
    // preconditions
    //   a.scale ~ b.scale (within 1e-6 relative)
    //   a.level == b.level
    //   a.level >= 1
    //
    // result is at the SAME level as inputs
    // call rescale to bring scale down and drop a prime
    Ciphertext mul_cipher(const Ciphertext &a,
                          const Ciphertext &b,
                          const EvalKey &evk,
                          const std::array<NTT, NUM_PRIMES> &ntts);

    // drop the highest active prime from the chain
    // brings scale from 2^100 back down to ~2^40 after a mul_scalar
    //
    // with q_drop = COEFF_MODULI[ct.level - 1]
    //     out.c0[i][j] = (ct.c0[i][j] - lift(ct.c0[L-1][j])) * inv(q_drop, q_i)   (mod q_i) for i < L-1
    //     out.c1 same
    //     out.scale = ct.scale / q_drop
    //     out.level = ct.level - 1
    //
    // throws if ct.level < 2
    Ciphertext rescale(const Ciphertext &ct,
                       const std::array<NTT, NUM_PRIMES> &ntts);
} // namespace ssns::ckks
