// pointwise polynomial ops shared by encrypt decrypt and arithmetic
//
// all three helpers take const Polynomial& return fresh Polynomial
// operate slot by slot per prime mod q_i
//
// pointwise_mul_ntt is meaningful only when BOTH operands are in ntt form
// fn does not check the form caller responsibility
//
// pointwise_add and pointwise_sub work in either form
// these wrappers just give a non mutating alternative to add_inplace and sub_inplace
#pragma once

#include <ssns/ckks/poly.hpp>

namespace ssns::ckks {

// out[i][k] = a[i][k] * b[i][k] mod q_i
// inputs must be ntt form
Polynomial pointwise_mul_ntt(const Polynomial& a, const Polynomial& b);

// out[i][k] = (a[i][k] + b[i][k]) mod q_i form agnostic
Polynomial pointwise_add(const Polynomial& a, const Polynomial& b);

// out[i][k] = (a[i][k] - b[i][k]) mod q_i form agnostic
Polynomial pointwise_sub(const Polynomial& a, const Polynomial& b);

}  // namespace ssns::ckks
