// crt lift and center helpers for rns coefficients
//
// a ckks coefficient lives mod Q = product of q_i
// stored as residues one per prime
// to get the signed integer back we do garner mixed radix lift then center
// values above Q/2 mean negative integer x - Q
//
// Q is roughly 2^200 so we use 256 bit unsigned (4 u64 limbs)
//
// центрирование: ckks хранит signed как [0, Q) минус это всё что > Q/2
// без центра negative coefs восстановятся как Q - tiny вместо -tiny
//
// api
//   U256 crt_lift(residues)        assemble residues into x in [0, Q)
//   double crt_center_to_double(x) center mod Q return signed double
#pragma once

#include <ssns/ckks/params.hpp>

#include <array>
#include <cstdint>

namespace ssns::ckks {

// 256 bit unsigned little endian limbs
// holds the lifted coefficient before centering
struct U256 {
    std::uint64_t lo{0}, mid_lo{0}, mid_hi{0}, hi{0};
};

// lift residues into x in [0, Q)
U256 crt_lift(const std::array<std::uint64_t, NUM_PRIMES>& r);

// level aware lift only first `level` primes participate
// reconstructed value lives in [0, Q_level)
// residues above level are ignored
// 1 <= level <= NUM_PRIMES
U256 crt_lift(const std::array<std::uint64_t, NUM_PRIMES>& r, std::size_t level);

// center around 0 mod Q return signed double
// if x > Q/2 then value is x - Q
double crt_center_to_double(const U256& x);

// level aware centering x lives in [0, Q_level)
double crt_center_to_double(const U256& x, std::size_t level);

}  // namespace ssns::ckks
