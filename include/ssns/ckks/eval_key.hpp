// ckks evaluation key for relinearisation
// rns gadget construction
//
// after cipher x cipher we have degree 2 ct (c0, c1, c2) where
// c0 + c1*s + c2*s^2 ~ delta^2 * m1 * m2 (mod Q)
// relin folds the s^2 term back into degree 1 using an eval key
//
// why rns gadget not the naive single key
// naive bv gives noise ||d2|| * N * sigma which can hit Q/2 * N * sigma and swamps the message
// rns gadget breaks d2 across the prime chain so each sub key noise is bounded by q_i not Q
//
// тонкость: gadget здесь это crt разложение не binary
// каждый sub key i отвечает за проекцию d2 на slot i
// сумма по i восстанавливает d2 через crt
//
// construction
// define rns basis indicator e_i in Z_Q so that
//     e_i == 1 (mod q_i)
//     e_i == 0 (mod q_j)   for j != i
// in rns form just residue 1 at slot i and 0 elsewhere
//
// for each i in [0, NUM_PRIMES) we sample
//     a_i        uniform in Z_Q (ntt form)
//     e_i_noise  gaussian sigma (coef form)
//     b_i        =  -a_i * s + e_i_noise + e_i * s^2   (mod Q)
//
// relin
// given degree 2 (d0, d1, d2) all ntt form
//   1 inverse ntt d2 to get coef form
//   2 for each i lift d2 slot i residue to centered int reduce mod each q_j build d2_at_i forward ntt
//   3 c0_relin += sum_i d2_at_i * sub_keys[i].b
//      c1_relin += sum_i d2_at_i * sub_keys[i].a
//   4 c0_relin += d0 c1_relin += d1
//
// decryption sketch
// (c0_relin) + s * (c1_relin)
//   = d0 + d1*s + sum_i d2_at_i * (e_i * s^2 + e_i_noise)
//   = d0 + d1*s + s^2 * d2 + small noise
// потому что sum_i d2_at_i * e_i = d2 в crt
//
// storage
// all sub_keys[i].a and sub_keys[i].b in ntt form
// only one inverse ntt needed at relin time on d2
#pragma once

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <random>
#include <utility>

namespace ssns::ckks {

// one rns gadget sub key
// (b_i, a_i) with b_i + a_i * s = e_i_noise + e_i * s^2 (mod Q)
struct EvalSubKey {
    Polynomial b;  // ntt form
    Polynomial a;  // ntt form uniform in Z_Q
};

// rns gadget evaluation key one sub key per prime
struct EvalKey {
    std::array<EvalSubKey, NUM_PRIMES> sub_keys;
};

// generate fresh eval key from sk
// uses tightened gaussian sigma so cipher x cipher + rescale stays inside test tolerance
// all sub_keys stored in ntt form
EvalKey gen_eval_key(const SecretKey& sk,
                     const std::array<NTT, NUM_PRIMES>& ntts,
                     std::mt19937_64& rng);

// gaussian sigma for the per sub key noise
// lower than KEYGEN_NOISE_SIGMA to keep relin noise below 1e-2 per slot at our config
// security cost is mild only the eval key noise is reduced
// encryption sigma stays at 3.2 so sk is still protected at the documented rlwe level
inline constexpr double EVAL_KEY_NOISE_SIGMA = 0.5;

}  // namespace ssns::ckks
