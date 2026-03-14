// decrypt — implementation.  See header for the RLWE math.
#include <ssns/ckks/decrypt.hpp>

#include <ssns/ckks/ntt_ops.hpp>
#include <ssns/ckks/poly.hpp>

namespace ssns::ckks {

Plaintext decrypt(const Ciphertext& ct,
                  const SecretKey& sk,
                  const std::array<NTT, NUM_PRIMES>& ntts) {
    // 1. Bring sk.s (stored in coefficient form) into NTT form.  We
    //    must NOT mutate sk, so work on a local copy.
    Polynomial s_ntt = sk.s;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ntts[i].forward(s_ntt.residues[i].data());
    }

    // 2. c1 · s in NTT form (pointwise multiplication).
    Polynomial c1s = pointwise_mul_ntt(ct.c1, s_ntt);

    // 3. pt.poly = c0 + c1·s   (mod q), still in NTT form.
    Polynomial pt_poly = pointwise_add(ct.c0, c1s);

    Plaintext pt;
    pt.poly = std::move(pt_poly);
    pt.scale = ct.scale;
    pt.level = ct.level;
    return pt;
}

}  // namespace ssns::ckks
