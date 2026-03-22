// CKKS depth-0 linear operations.
//
// "Depth-0" here means no cipher×cipher multiplication — these ops do not
// require relinearization.  The set:
//
//   add(ct1, ct2)            cipher + cipher       (pointwise NTT add)
//   sub(ct1, ct2)            cipher - cipher       (pointwise NTT sub)
//   add_plain(ct, pt)        cipher + plaintext    (c0 += pt.poly)
//   sub_plain(ct, pt)        cipher - plaintext    (c0 -= pt.poly)
//   mul_scalar(ct, s)        cipher * real-scalar  (per-prime mod-mul)
//
// All inputs are assumed to be in NTT form (the storage convention used
// throughout the CKKS pipeline) — so pointwise add/sub/mul in the
// frequency domain are equivalent to coefficient-domain polynomial ops.
//
// Preconditions
// -------------
// add/sub/add_plain/sub_plain: both operands must agree on `scale` and
// `level`.  Mismatch throws `std::invalid_argument`.  Tests use exact
// equality: the encoder produces deterministic scale values so floating-
// point fuzz is not a concern.
//
// Scale arithmetic
// ----------------
// add/sub/add_plain/sub_plain preserve scale and level.
//
// mul_scalar bumps scale: the real `scalar` is encoded as a 60-bit integer
// `k = round(scalar * 2^60)` so that the multiplication is exact mod each
// prime.  The output ciphertext therefore carries
//
//     out.scale = ct.scale * 2^60     (= ct.scale * COEFF_MODULI[NUM_PRIMES-1] approx)
//     out.level = ct.level
//
// Strictly the bump factor is exactly 2^60 (a power of two), not the prime
// itself; the mismatch is corrected by `rescale` which divides by the prime
// dropped from the chain.  See `rescale` (Phase 6.5) and the matched test
// `mul_scalar then rescale: decrypt approx scalar*m within 1e-3` for the
// canonical pairing 2^40 → 2^100 → 2^40 with level 4 → 4 → 3.
#pragma once

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/eval_key.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>

#include <array>

namespace ssns::ckks {

Ciphertext add(const Ciphertext& a, const Ciphertext& b);
Ciphertext sub(const Ciphertext& a, const Ciphertext& b);
Ciphertext add_plain(const Ciphertext& ct, const Plaintext& pt);
Ciphertext sub_plain(const Ciphertext& ct, const Plaintext& pt);

// Multiply a ciphertext by a real scalar.  See "Scale arithmetic" above
// for how the output scale is computed (always ct.scale * 2^60, regardless
// of `scalar`).  The result must be `rescale`-d before being added to a
// ciphertext that hasn't been bumped.
Ciphertext mul_scalar(const Ciphertext& ct, double scalar);

// ---------------------------------------------------------------------------
// Depth-1 multiplications (Phase 6.5)
// ---------------------------------------------------------------------------
//
// mul_plain(ct, pt) — cipher × plaintext.  Both operands stored in NTT form,
// so the operation is pointwise multiplication on each (c0, c1) component.
// Result carries the bumped scale and the lower of the two levels:
//
//     out.c0    = pointwise_mul_ntt(ct.c0, pt.poly)
//     out.c1    = pointwise_mul_ntt(ct.c1, pt.poly)
//     out.scale = ct.scale * pt.scale
//     out.level = min(ct.level, pt.level)
//
// Precondition: ct.level == pt.level — the active modulus chains must agree.
// Mismatch throws std::invalid_argument.  Scale is NOT required to match
// (this op is multiplicative — bumping is the whole point).
Ciphertext mul_plain(const Ciphertext& ct, const Plaintext& pt);

// mul_cipher(a, b, evk) — cipher × cipher with BV-style relinearisation.
//
// Tensor expansion (in NTT form):
//     d0 = a.c0 * b.c0
//     d1 = a.c0 * b.c1 + a.c1 * b.c0
//     d2 = a.c1 * b.c1
//
// Relinearisation back to a degree-1 ciphertext via the EvalKey identity
// b_evk + a_evk·s ≈ s² (mod q):
//     out.c0 = d0 + d2 * evk.b
//     out.c1 = d1 + d2 * evk.a
//     out.scale = a.scale * b.scale
//     out.level = a.level
//
// Preconditions:
//   * a.scale ≈ b.scale (within ~1e-6 relative — scales come from a
//     deterministic chain, so exact equality is the common case)
//   * a.level == b.level
//   * a.level >= 1
// Mismatch throws std::invalid_argument.
//
// The EvalKey is generated at full level NUM_PRIMES; ops at level L < NUM_PRIMES
// only read the first L RNS slots, which is fine because all four were filled
// at keygen.
//
// The result is at the SAME level as the inputs (mul_cipher does not drop a
// level by itself).  Call `rescale` to bring the bumped scale back down and
// drop one prime.
Ciphertext mul_cipher(const Ciphertext& a,
                      const Ciphertext& b,
                      const EvalKey& evk);

// Drop the highest-indexed active prime from the modulus chain.  This is
// the CKKS "rescale" / "mod-down" operation: it brings the scale back from
// e.g. 2^100 down to ~2^40 after a `mul_scalar` (or, in later phases, a
// cipher×cipher multiplication).
//
// Concretely, with q_drop = COEFF_MODULI[ct.level - 1]:
//
//     out.c0[i][j] = (ct.c0[i][j] - lift(ct.c0[L-1][j])) * inv(q_drop, q_i)   (mod q_i)
//                                                                  for i < L-1
//     out.c1 analogously
//     out.scale = ct.scale / q_drop
//     out.level = ct.level - 1
//
// where lift(.) centers the residue mod q_drop into a signed integer.  All
// of this is done by inverse-NTT-ing into coefficient form, applying the
// per-prime correction, then re-NTT-ing the surviving residues — the dropped
// residue slot is zeroed for hygiene but downstream ops should not read it
// anyway.
//
// Throws std::invalid_argument if ct.level < 2 (cannot rescale to zero
// active primes).
Ciphertext rescale(const Ciphertext& ct,
                   const std::array<NTT, NUM_PRIMES>& ntts);

}  // namespace ssns::ckks
