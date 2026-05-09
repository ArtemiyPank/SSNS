// ckks rlwe public key pk = (b, a) with b = -a*s + e (mod q)
//
// security reduces to rlwe
// distinguishing (a, -a*s + e) from uniform (a, b) is hard if rlwe holds
//
// encrypt: ct = (c0, c1) = (b*u + e0 + delta*m, a*u + e1) for fresh small (u, e0, e1)
// decrypt gives c0 + c1*s = delta*m + small noise
//
// storage
// both a and b in ntt form
//   encrypt does a*u and b*u so doing them in ntt domain saves two forward transforms per encrypt
//   stored polys never need coef view at runtime
//
// secret key by contrast stays in coef form (sparse ternary)
#pragma once

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <random>

namespace ssns::ckks {

// gaussian sigma for noise polynomial e
// mirrors seal lattigo defaults gives ~128 bit security at N=8192
inline constexpr double KEYGEN_NOISE_SIGMA = 3.2;

struct PublicKey {
    Polynomial b;  // = -a*s + e (mod q) ntt form
    Polynomial a;  // uniform random in Z_q ntt form
};

// generate fresh public key from sk
// uses ntts for coef <-> ntt conversions during a*s computation
// rng samples a (uniform per prime) and e (rounded gaussian sigma=3.2)
//
// output a and b stored in ntt form
PublicKey gen_public_key(const SecretKey& sk,
                         const std::array<NTT, NUM_PRIMES>& ntts,
                         std::mt19937_64& rng);

}  // namespace ssns::ckks
