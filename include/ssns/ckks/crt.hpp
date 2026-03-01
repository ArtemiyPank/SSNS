// CRT (Chinese-Remainder) lift / center helpers for CKKS RNS coefficients.
//
// A CKKS polynomial coefficient lives mod Q = Π q_i, but is stored as a
// tuple of residues (one per prime).  Reconstructing the signed integer
// requires Garner's algorithm to assemble the residues, plus a centering
// step (values > Q/2 represent negative integers x − Q).
//
// Q is on the order of 2^200, which doesn't fit in any native type, so we
// use a 256-bit unsigned integer (`U256`) made of 4 uint64_t limbs.  Both
// the encoder and the keygen tests need this, so we expose it here.
//
// API:
//   U256 crt_lift(residues)        — assemble residues into x ∈ [0, Q).
//   double crt_center_to_double(x) — center mod Q, return as signed double.
//
// The U256 type and Garner context are also exposed for tests that want
// to inspect lifted values directly.
#pragma once

#include <ssns/ckks/params.hpp>

#include <array>
#include <cstdint>

namespace ssns::ckks {

// 256-bit unsigned integer, little-endian limb layout.  Used to hold the
// CRT-lifted coefficient before centering — Q ≈ 2^200, so 4 limbs are
// plenty.
struct U256 {
    std::uint64_t lo{0}, mid_lo{0}, mid_hi{0}, hi{0};
};

// In-place 256-bit add: a += b (wrapping).
void u256_add(U256& a, const U256& b);

// In-place 256-bit multiply by a 64-bit scalar: a *= m (wrapping).
void u256_mul_u64(U256& a, std::uint64_t m);

// Strict less-than comparison.
bool u256_lt(const U256& a, const U256& b);

// In-place 256-bit subtract: a -= b.  Caller must ensure a >= b.
void u256_sub(U256& a, const U256& b);

// Convert U256 to double (treating value as unsigned).
double u256_to_double(const U256& a);

// Lift a tuple of RNS residues r[i] = x mod q_i into x ∈ [0, Q).
U256 crt_lift(const std::array<std::uint64_t, NUM_PRIMES>& r);

// Center a lifted U256 around 0 (mod Q) and return as signed double.
// If x > Q/2 the value represents the negative integer x − Q.
double crt_center_to_double(const U256& x);

}  // namespace ssns::ckks
