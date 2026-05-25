// decrypt impl see header
#include <ssns/ckks/decrypt.hpp>

#include <ssns/ckks/ntt_ops.hpp>
#include <ssns/ckks/poly.hpp>

namespace ssns::ckks {
    // decrypt converts sk.s to ntt form on each call
    // hot loops should use decrypt_with_ntt_sk
    Plaintext decrypt(const Ciphertext &ct, const SecretKey &sk,
                      const std::array<NTT, NUM_PRIMES> &ntts) {
        // bring sk.s (coef form) to ntt
        // must NOT mutate sk so work on a copy
        Polynomial s_ntt = sk.s;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            ntts[i].forward(s_ntt.residues[i].data());
        }
        return decrypt_with_ntt_sk(ct, s_ntt);
    }

    // decrypt with pre computed ntt form sk
    // c1 * s_ntt pointwise then add c0
    Plaintext decrypt_with_ntt_sk(const Ciphertext &ct, const Polynomial &s_ntt) {
        // both sides ntt form
        Polynomial c1s = pointwise_mul_ntt(ct.c1, s_ntt);
        Polynomial pt_poly = pointwise_add(ct.c0, c1s);

        Plaintext pt;
        pt.poly = std::move(pt_poly);
        pt.scale = ct.scale;
        pt.level = ct.level;
        return pt;
    }
} // namespace ssns::ckks
