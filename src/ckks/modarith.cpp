// non inline modarith helpers see header
#include <ssns/ckks/modarith.hpp>

#include <stdexcept>

namespace ssns::ckks {
    // deterministic miller rabin for any uint64
    // witness set {2,...,37} is provably sufficient for n < 2^64
    bool is_prime(std::uint64_t n) noexcept {
        if (n < 2) return false;
        // small prime trial division catches witnesses themselves and short circuits common composites
        constexpr std::uint64_t small[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
        for (std::uint64_t sp: small) {
            if (n == sp) return true;
            if (n % sp == 0) return false;
        }
        // write n - 1 = d * 2^s with d odd
        // тут именно факторизация двойки нужна для quadratic residue теста
        std::uint64_t d = n - 1;
        int s = 0;
        while ((d & 1ULL) == 0) {
            d >>= 1;
            ++s;
        }
        for (std::uint64_t a: small) {
            std::uint64_t x = pow_mod(a, d, n);
            // a^d == ±1 значит свидетель не уличает n
            if (x == 1 || x == n - 1) continue;
            bool composite = true;
            // s-1 квадратов: ищем x == -1 на любой стадии
            for (int r = 0; r < s - 1; ++r) {
                x = mul_mod(x, x, n);
                if (x == n - 1) {
                    composite = false;
                    break;
                }
            }
            if (composite) return false;
        }
        return true;
    }

    // find primitive 2N-th root of unity in F_p by trying small g
    // F_p^* is cyclic of order p-1
    // 2N-th roots exist iff 2N | (p-1) which we enforce on prime selection see params.hpp
    // // for NTT
    std::uint64_t primitive_2n_root(std::uint64_t p, std::uint64_t two_n) {
        // 2N must divide p - 1 else no primitive 2N-th root
        if ((p - 1) % two_n != 0) {
            throw std::invalid_argument("primitive_2n_root: 2N does not divide p-1");
        }
        // exponent поднимает g до 2N-го корня: g^((p-1)/2N) имеет порядок dividing 2N
        const std::uint64_t exponent = (p - 1) / two_n;
        const std::uint64_t n = two_n / 2;
        // try candidates g of F_p^*
        // psi = g^((p-1)/(2N)) is some 2N-th root
        // verify psi^N != 1 (true order is 2N not a divisor) and psi^(2N) == 1
        //
        // почему не search for full generator: full generator search требует факторизации p-1
        // нам достаточно ЛЮБОЙ 2N-й корень primitive order не нужен generator всей F_p^*
        for (std::uint64_t g = 2; g < p; ++g) {
            std::uint64_t psi = pow_mod(g, exponent, p);
            if (psi <= 1) continue; // trivial roots
            if (pow_mod(psi, n, p) == 1) continue; // order divides N not primitive
            // psi^(2N) == 1 holds by construction sanity check anyway
            if (pow_mod(psi, two_n, p) != 1) continue;
            return psi;
        }
        throw std::runtime_error("primitive_2n_root: search failed (this should be impossible)");
    }
} // namespace ssns::ckks
