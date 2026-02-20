// Implementations of the non-inline modarith helpers.  See the header
// for contracts and bit-size constraints.
#include <ssns/ckks/modarith.hpp>

#include <stdexcept>

namespace ssns::ckks {

bool is_prime(std::uint64_t n) noexcept {
    if (n < 2) return false;
    // Small-prime trial division catches the witnesses themselves and
    // short-circuits common composites.
    constexpr std::uint64_t small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (std::uint64_t sp : small) {
        if (n == sp) return true;
        if (n % sp == 0) return false;
    }
    // Write n - 1 = d · 2^s with d odd.
    std::uint64_t d = n - 1;
    int s = 0;
    while ((d & 1ULL) == 0) { d >>= 1; ++s; }
    for (std::uint64_t a : small) {
        std::uint64_t x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int r = 0; r < s - 1; ++r) {
            x = mul_mod(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

std::uint64_t primitive_2n_root(std::uint64_t p, std::uint64_t two_n) {
    // Sanity: 2N must divide p - 1, else no primitive 2N-th root exists.
    if ((p - 1) % two_n != 0) {
        throw std::invalid_argument("primitive_2n_root: 2N does not divide p-1");
    }
    const std::uint64_t exponent = (p - 1) / two_n;
    const std::uint64_t n = two_n / 2;
    // Iterate candidate generators g of F_p^*.  ψ = g^((p-1)/(2N)) is
    // *some* 2N-th root.  Verify both that ψ^N ≢ 1 (so ψ has true order
    // 2N, not a proper divisor of 2N) and ψ^(2N) ≡ 1.
    for (std::uint64_t g = 2; g < p; ++g) {
        std::uint64_t psi = pow_mod(g, exponent, p);
        if (psi <= 1) continue;                       // trivial roots
        if (pow_mod(psi, n, p) == 1) continue;        // order divides N — not primitive
        // ψ^(2N) ≡ 1 holds by construction (Fermat: g^(p-1) ≡ 1, and
        // (p-1)/(2N) * 2N = p-1).  Sanity-check it anyway.
        if (pow_mod(psi, two_n, p) != 1) continue;
        return psi;
    }
    throw std::runtime_error("primitive_2n_root: search failed (this should be impossible)");
}

}  // namespace ssns::ckks
