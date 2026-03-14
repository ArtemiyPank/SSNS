// CKKS RLWE decryption — secret-key form.
//
// Given a ciphertext `ct = (c0, c1)` produced by `encrypt` and the
// secret key `sk.s`, decrypt computes
//
//     pt.poly = c0 + c1·s   (mod q)
//
// in NTT form (the same form encryption stored c0 and c1 in).  Decrypting
// yields the plaintext message plus a small additive noise term — the
// decoder downstream divides by `pt.scale` and absorbs that noise as
// floating-point round-off in the recovered slots.
//
// The returned Plaintext carries `ct.scale` and `ct.level` unchanged so
// the decoder can scale slots correctly and downstream code can know
// which RNS levels are still active.
#pragma once

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>

namespace ssns::ckks {

Plaintext decrypt(const Ciphertext& ct,
                  const SecretKey& sk,
                  const std::array<NTT, NUM_PRIMES>& ntts);

}  // namespace ssns::ckks
