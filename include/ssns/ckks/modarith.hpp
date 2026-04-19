// Modular arithmetic primitives over uint64 used by the CKKS RNS / NTT
// machinery.  Header-only to give the compiler full inlining freedom on
// the hot NTT inner loop.
//
// All routines assume the modulus p fits in 63 bits so that the product
// of two reduced operands stays within __uint128_t and `prod % p` is
// well-defined.  This matches the bit-sizes used by the four primes in
// `params.hpp` (60-bit max).
//
// `add_mod` / `sub_mod` are constexpr; `mul_mod`, `pow_mod`, and
// `inv_mod` use __uint128_t so they are not constexpr in C++20 (compiler
// extensions vary), but they are pure inline functions safe to use in
// hot paths.
//
// Miller-Rabin is provided as a deterministic 64-bit primality test for
// the validation suite — not used at runtime by NTT itself.
#pragma once

#include <cstdint>

namespace ssns::ckks {

// (a + b) mod p.  Branch on overflow and on s >= p; both checks must
// happen because a + b can wrap around 2^64 while still being < p.
constexpr std::uint64_t add_mod(std::uint64_t a, std::uint64_t b, std::uint64_t p) noexcept {
    std::uint64_t s = a + b;
    // s < a iff the unsigned addition wrapped past 2^64.
    return (s >= p || s < a) ? s - p : s;
}

// (a - b) mod p.  Both operands assumed already reduced (< p).
constexpr std::uint64_t sub_mod(std::uint64_t a, std::uint64_t b, std::uint64_t p) noexcept {
    return (a >= b) ? (a - b) : (a + p - b);
}

// (a * b) mod p via 128-bit intermediate.  Operands need not be reduced;
// the modulo at the end normalises the result.  Safe for p < 2^63.
//
// `__int128` is a GCC/Clang extension; suppress -Wpedantic locally so
// the rest of the project can build with -Wpedantic clean.
inline std::uint64_t mul_mod(std::uint64_t a, std::uint64_t b, std::uint64_t p) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using u128 = unsigned __int128;
    u128 prod = static_cast<u128>(a) * b;
#pragma GCC diagnostic pop
    return static_cast<std::uint64_t>(prod % p);
}

// Pseudo-Mersenne reduction for primes of the form p = 2^K - C with K=60
// and C < 2^21.  Replaces the 128/64 hardware divide in mul_mod with a
// few shift-mask-mul-add iterations using the identity 2^K ≡ C (mod p).
// Empirically ~2× faster than the generic mul_mod on modern x86-64.
//
// Caller invariants: 0 ≤ a, b < 2^60.  All four CKKS primes in params.hpp
// satisfy the precondition (q0/q3 are 60-bit pseudo-Mersenne; q1/q2 are
// 40-bit which we handle separately).
template <std::uint64_t P, std::uint64_t C>
inline std::uint64_t mul_mod_psm60(std::uint64_t a, std::uint64_t b) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using u128 = unsigned __int128;
    constexpr std::uint64_t MASK60 = (std::uint64_t{1} << 60) - 1;
    u128 prod = static_cast<u128>(a) * b;
    // prod < 2^120.  Iteratively replace 2^60 with C.
    std::uint64_t lo  = static_cast<std::uint64_t>(prod) & MASK60;
    std::uint64_t hi  = static_cast<std::uint64_t>(prod >> 60);   // < 2^60
    u128 r1 = static_cast<u128>(lo) + static_cast<u128>(hi) * C;  // < 2^81
    std::uint64_t r1_lo = static_cast<std::uint64_t>(r1) & MASK60;
    std::uint64_t r1_hi = static_cast<std::uint64_t>(r1 >> 60);    // < 2^21
    std::uint64_t r2    = r1_lo + r1_hi * C;                       // < 2^60 + 2^41
    std::uint64_t r2_lo = r2 & MASK60;
    std::uint64_t r2_hi = r2 >> 60;                                // 0 or 1
    std::uint64_t r3    = r2_lo + r2_hi * C;                       // < 2^60 + 2^21 < 2p
    if (r3 >= P) r3 -= P;
#pragma GCC diagnostic pop
    return r3;
}

// Same idea for K=40 primes (q1, q2).
template <std::uint64_t P, std::uint64_t C>
inline std::uint64_t mul_mod_psm40(std::uint64_t a, std::uint64_t b) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using u128 = unsigned __int128;
    constexpr std::uint64_t MASK40 = (std::uint64_t{1} << 40) - 1;
    u128 prod = static_cast<u128>(a) * b;                          // < 2^80
    std::uint64_t lo = static_cast<std::uint64_t>(prod) & MASK40;
    std::uint64_t hi = static_cast<std::uint64_t>(prod >> 40);     // < 2^40
    // r1 = lo + hi*C, hi < 2^40, C < 2^20 → hi*C < 2^60, fits u64.
    std::uint64_t r1   = lo + hi * C;                              // < 2^60+2^40
    std::uint64_t r1_lo = r1 & MASK40;
    std::uint64_t r1_hi = r1 >> 40;                                // < 2^20
    std::uint64_t r2 = r1_lo + r1_hi * C;                          // < 2^40 + 2^40
    std::uint64_t r2_lo = r2 & MASK40;
    std::uint64_t r2_hi = r2 >> 40;                                // 0 or 1
    std::uint64_t r3 = r2_lo + r2_hi * C;
    if (r3 >= P) r3 -= P;
#pragma GCC diagnostic pop
    return r3;
}

// base^exp mod p.  Standard square-and-multiply.
inline std::uint64_t pow_mod(std::uint64_t base, std::uint64_t exp, std::uint64_t p) noexcept {
    std::uint64_t r = 1 % p;
    base %= p;
    while (exp) {
        if (exp & 1ULL) r = mul_mod(r, base, p);
        base = mul_mod(base, base, p);
        exp >>= 1;
    }
    return r;
}

// Modular inverse via Fermat's little theorem: a^(p-2) ≡ a^-1 (mod p).
// Requires p to be prime and a != 0 (mod p).
inline std::uint64_t inv_mod(std::uint64_t a, std::uint64_t p) noexcept {
    return pow_mod(a, p - 2, p);
}

// Find a primitive 2N-th root of unity ψ in F_p.  Required:
//   ψ^N    ≢ 1 (mod p)   (otherwise ψ is only an N-th root, not 2N-th)
//   ψ^(2N) ≡ 1 (mod p)   (definition of 2N-th root)
//
// Strategy: start with the smallest 2N-th root (any g^((p-1)/(2N)) for a
// generator g of F_p^*).  Rather than searching for a generator, we test
// each candidate g = 2, 3, 4, ... and check both conditions directly —
// the smallest one that passes is a valid primitive 2N-th root.  In
// practice this terminates within a handful of iterations.
//
// `two_n` must divide p-1 (otherwise no 2N-th root exists at all).
std::uint64_t primitive_2n_root(std::uint64_t p, std::uint64_t two_n);

// Deterministic Miller-Rabin primality test for any uint64_t.  Witnesses
// {2,3,5,7,11,13,17,19,23,29,31,37} are provably sufficient for n < 2^64.
// O(k * log^3 n).
bool is_prime(std::uint64_t n) noexcept;

}  // namespace ssns::ckks
