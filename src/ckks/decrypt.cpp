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
    return decrypt_with_ntt_sk(ct, s_ntt);
}

Plaintext decrypt_with_ntt_sk(const Ciphertext& ct, const Polynomial& s_ntt) {
    // c1 · s_ntt pointwise, then add c0.  Both sides already NTT form.
    Polynomial c1s = pointwise_mul_ntt(ct.c1, s_ntt);
    Polynomial pt_poly = pointwise_add(ct.c0, c1s);

    Plaintext pt;
    pt.poly = std::move(pt_poly);
    pt.scale = ct.scale;
    pt.level = ct.level;
    return pt;
}

}  // namespace ssns::ckks
