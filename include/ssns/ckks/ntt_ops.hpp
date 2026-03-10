// Pointwise polynomial operations shared by encrypt / decrypt / arithmetic.
//
// All three helpers consume two `const Polynomial&` of the canonical
// NUM_PRIMES × POLY_DEGREE shape and return a fresh Polynomial.  They
// operate slot-by-slot per RNS prime, modulo q_i.
//
// `pointwise_mul_ntt` is meaningful only when BOTH operands are in NTT
// (frequency-domain) form — that is the convention under which slot-
// wise multiplication corresponds to polynomial multiplication in
// Z_q[X]/(X^N+1).  The function itself does not care about the form;
// the contract is the caller's responsibility.
//
// `pointwise_add` and `pointwise_sub` work in either form (NTT or
// coefficient) since modular addition is form-agnostic.  Pre-existing
// code uses `Polynomial::add_inplace` / `sub_inplace`; these wrappers
// give a non-mutating alternative that returns a new value, which keeps
// encrypt / decrypt expression-level rather than statement-level.
#pragma once

#include <ssns/ckks/poly.hpp>

namespace ssns::ckks {

// out[i][k] = a[i][k] * b[i][k] mod q_i, for each prime i and slot k.
// Inputs must already be in NTT form.
Polynomial pointwise_mul_ntt(const Polynomial& a, const Polynomial& b);

// out[i][k] = (a[i][k] + b[i][k]) mod q_i.  Form-agnostic.
Polynomial pointwise_add(const Polynomial& a, const Polynomial& b);

// out[i][k] = (a[i][k] - b[i][k]) mod q_i.  Form-agnostic.
Polynomial pointwise_sub(const Polynomial& a, const Polynomial& b);

}  // namespace ssns::ckks
