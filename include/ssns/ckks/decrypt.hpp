// ckks rlwe decrypt secret key form
//
// given ct = (c0, c1) and sk.s do
//     pt.poly = c0 + c1*s (mod q)
// in ntt form same as encrypt stored them
// gives plaintext plus small noise
// decoder absorbs that as fp roundoff
//
// returned pt carries ct.scale and ct.level unchanged
#pragma once

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>

namespace ssns::ckks {

// decrypt ct with sk
// converts sk.s to ntt form on every call
// for hot loops use decrypt_with_ntt_sk
Plaintext decrypt(const Ciphertext& ct,
                  const SecretKey& sk,
                  const std::array<NTT, NUM_PRIMES>& ntts);

// fast path takes pre computed ntt form sk
// caller does s_ntt = forward_ntt(sk.s) once
Plaintext decrypt_with_ntt_sk(const Ciphertext& ct,
                               const Polynomial& s_ntt);

}  // namespace ssns::ckks
