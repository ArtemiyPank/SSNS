// CKKS evaluation key (relinearisation) — BV-style construction.
//
// After cipher × cipher we end up with a degree-2 ciphertext (c0, c1, c2)
// satisfying  c0 + c1·s + c2·s² ≈ Δ²·m1·m2  (mod q).  Relinearisation
// folds the s² term back into a degree-1 ciphertext using an evaluation
// key (b_evk, a_evk) such that
//
//     b_evk + a_evk·s ≈ s²  (mod q)            (1)
//
// after which the relinearisation step rewrites
//
//     c0 + c1·s + c2·s² = c0 + c1·s + c2·(b_evk + a_evk·s) + small noise
//                        = (c0 + c2·b_evk) + (c1 + c2·a_evk)·s + small.
//
// Construction (BV / "non-extended" — no extra modulus P):
//
//     a_evk  ← uniform in Z_q (NTT form)
//     e_evk  ← Gaussian σ=3.2  (coefficient form, lifted)
//     b_evk  =  -a_evk·s + e_evk + s²    (mod q)
//
// Trade-off: this is simpler than the SEAL "extended" key-switch (which
// scales by an auxiliary prime P to reduce noise).  It produces more
// noise per relin step, which is acceptable for the SSNS workload of at
// most 3 multiplications per gradient update.  If noise becomes the
// limiting factor, switch to extended keys later.
//
// Storage: both `a` and `b` are in NTT form, identical convention to
// PublicKey, so relinearisation can perform pointwise multiplications
// in the frequency domain.
#pragma once

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <random>

namespace ssns::ckks {

struct EvalKey {
    Polynomial b;  // = -a·s + e + s² (mod q), NTT form.
    Polynomial a;  // uniform random in Z_q, NTT form.
};

// Generate a fresh BV-style evaluation key from `sk`.  Reuses the
// PublicKey noise σ (3.2) — see public_key.hpp for the underlying
// constants.
//
// The output `evk.a` and `evk.b` are stored in NTT form.
EvalKey gen_eval_key(const SecretKey& sk,
                     const std::array<NTT, NUM_PRIMES>& ntts,
                     std::mt19937_64& rng);

}  // namespace ssns::ckks
