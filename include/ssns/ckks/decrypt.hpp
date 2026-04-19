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

// Fast path for hot loops: take a pre-computed NTT-form `s_ntt` instead
// of re-running 4 forward NTTs on sk.s every call.  Saves ~20 % of FHE
// training wall time at the SSNS preset (153k decrypts × 4 NTTs each).
//
// The caller is responsible for `s_ntt = forward_ntt(sk.s)` once.
Plaintext decrypt_with_ntt_sk(const Ciphertext& ct,
                               const Polynomial& s_ntt);

}  // namespace ssns::ckks
