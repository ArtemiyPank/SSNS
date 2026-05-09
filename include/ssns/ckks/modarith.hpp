// modular arithmetic over uint64 used by rns and ntt
// header only so the compiler can inline everything in the ntt inner loop
//
// all routines assume p fits in 63 bits so prod of two reduced operands stays in __uint128_t
// matches the four primes in params.hpp (60 bit max)
//
// add_mod sub_mod are constexpr
// mul_mod pow_mod inv_mod use __uint128_t so not constexpr but inline and safe in hot paths
//
// Miller-Rabin is provided as a deterministic 64 bit primality test for tests not used at runtime
#pragma once

#include <cstdint>

namespace ssns::ckks {

// (a + b) mod p
// branch on overflow and on s >= p
// both checks needed because a + b can wrap past 2^64 while still < p
constexpr std::uint64_t add_mod(std::uint64_t a, std::uint64_t b, std::uint64_t p) noexcept {
    std::uint64_t s = a + b;
    // s < a iff unsigned addition wrapped
    return (s >= p || s < a) ? s - p : s;
}

// (a - b) mod p
// both operands assumed reduced
constexpr std::uint64_t sub_mod(std::uint64_t a, std::uint64_t b, std::uint64_t p) noexcept {
    return (a >= b) ? (a - b) : (a + p - b);
}

// (a * b) mod p via 128 bit intermediate
// operands need not be reduced
// safe for p < 2^63
//
// __int128 is gcc clang extension suppress -Wpedantic locally
inline std::uint64_t mul_mod(std::uint64_t a, std::uint64_t b, std::uint64_t p) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using u128 = unsigned __int128;
    u128 prod = static_cast<u128>(a) * b;
#pragma GCC diagnostic pop
    return static_cast<std::uint64_t>(prod % p);
}

// pseudo mersenne reduction for p = 2^K - C with K=60 and C < 2^21
// uses 2^K == C (mod p) so prod mod p = (lo + hi*C) mod p
// avoids the 128/64 hardware divide
// about 2x faster than mul_mod on x86 64
//
// идея: prod = hi * 2^K + lo тогда prod mod p = (lo + hi*C) mod p
// итерируем 2-3 раза до тех пор пока остаток < p
//
// caller invariants 0 <= a, b < 2^60
template <std::uint64_t P, std::uint64_t C>
inline std::uint64_t mul_mod_psm60(std::uint64_t a, std::uint64_t b) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using u128 = unsigned __int128;
    constexpr std::uint64_t MASK60 = (std::uint64_t{1} << 60) - 1;
    u128 prod = static_cast<u128>(a) * b;
    // тут именно итеративно заменяем 2^60 на C, prod < 2^120
    std::uint64_t lo  = static_cast<std::uint64_t>(prod) & MASK60;
    std::uint64_t hi  = static_cast<std::uint64_t>(prod >> 60);   // < 2^60
    // первая итерация снижает диапазон с 2^120 до 2^81
    u128 r1 = static_cast<u128>(lo) + static_cast<u128>(hi) * C;  // < 2^81
    std::uint64_t r1_lo = static_cast<std::uint64_t>(r1) & MASK60;
    std::uint64_t r1_hi = static_cast<std::uint64_t>(r1 >> 60);    // < 2^21
    // вторая итерация: 2^81 -> 2^60 + 2^41
    std::uint64_t r2    = r1_lo + r1_hi * C;                       // < 2^60 + 2^41
    std::uint64_t r2_lo = r2 & MASK60;
    std::uint64_t r2_hi = r2 >> 60;                                // 0 or 1
    // третья итерация: r2_hi уже 0 или 1 поэтому результат < 2p
    std::uint64_t r3    = r2_lo + r2_hi * C;                       // < 2^60 + 2^21 < 2p
    // финальный if вместо modulo: достаточно одного вычитания
    if (r3 >= P) r3 -= P;
#pragma GCC diagnostic pop
    return r3;
}

// same idea for K=40 primes
template <std::uint64_t P, std::uint64_t C>
inline std::uint64_t mul_mod_psm40(std::uint64_t a, std::uint64_t b) noexcept {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using u128 = unsigned __int128;
    constexpr std::uint64_t MASK40 = (std::uint64_t{1} << 40) - 1;
    u128 prod = static_cast<u128>(a) * b;                          // < 2^80
    std::uint64_t lo = static_cast<std::uint64_t>(prod) & MASK40;
    std::uint64_t hi = static_cast<std::uint64_t>(prod >> 40);     // < 2^40
    // r1 = lo + hi*C, hi*C < 2^60 fits u64
    std::uint64_t r1   = lo + hi * C;                              // < 2^60 + 2^40
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

// base^exp mod p square and multiply
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

// modular inverse via fermat a^(p-2) == a^-1 mod p
// requires p prime and a != 0 mod p
inline std::uint64_t inv_mod(std::uint64_t a, std::uint64_t p) noexcept {
    return pow_mod(a, p - 2, p);
}

// find primitive 2N-th root of unity psi in F_p
//   psi^N    != 1 (mod p)
//   psi^(2N) == 1 (mod p)
//
// strategy try g = 2, 3, ... and check both conditions
// terminates in a handful of iterations
//
// two_n must divide p-1
std::uint64_t primitive_2n_root(std::uint64_t p, std::uint64_t two_n);

// deterministic miller rabin for any uint64
// witnesses {2,3,5,7,11,13,17,19,23,29,31,37} are sufficient for n < 2^64
bool is_prime(std::uint64_t n) noexcept;

}  // namespace ssns::ckks
