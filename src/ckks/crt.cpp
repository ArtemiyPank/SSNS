// CRT helpers — implementation.  See header for the math.
#include <ssns/ckks/crt.hpp>

#include <ssns/ckks/modarith.hpp>

namespace ssns::ckks {

void u256_add(U256& a, const U256& b) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using u128 = unsigned __int128;
    u128 carry = 0;
    u128 t;
    t = u128{a.lo} + b.lo + carry;
    a.lo = static_cast<std::uint64_t>(t);
    carry = t >> 64;
    t = u128{a.mid_lo} + b.mid_lo + carry;
    a.mid_lo = static_cast<std::uint64_t>(t);
    carry = t >> 64;
    t = u128{a.mid_hi} + b.mid_hi + carry;
    a.mid_hi = static_cast<std::uint64_t>(t);
    carry = t >> 64;
    t = u128{a.hi} + b.hi + carry;
    a.hi = static_cast<std::uint64_t>(t);
#pragma GCC diagnostic pop
}

void u256_mul_u64(U256& a, std::uint64_t m) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using u128 = unsigned __int128;
    u128 carry = 0;
    u128 t;
    t = u128{a.lo} * m + carry;
    a.lo = static_cast<std::uint64_t>(t);
    carry = t >> 64;
    t = u128{a.mid_lo} * m + carry;
    a.mid_lo = static_cast<std::uint64_t>(t);
    carry = t >> 64;
    t = u128{a.mid_hi} * m + carry;
    a.mid_hi = static_cast<std::uint64_t>(t);
    carry = t >> 64;
    t = u128{a.hi} * m + carry;
    a.hi = static_cast<std::uint64_t>(t);
#pragma GCC diagnostic pop
}

bool u256_lt(const U256& a, const U256& b) {
    if (a.hi != b.hi)         return a.hi < b.hi;
    if (a.mid_hi != b.mid_hi) return a.mid_hi < b.mid_hi;
    if (a.mid_lo != b.mid_lo) return a.mid_lo < b.mid_lo;
    return a.lo < b.lo;
}

void u256_sub(U256& a, const U256& b) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using i128 = __int128;
    i128 t;
    i128 borrow = 0;
    t = i128{a.lo} - b.lo - borrow;
    a.lo = static_cast<std::uint64_t>(static_cast<unsigned __int128>(t));
    borrow = (t < 0) ? 1 : 0;
    t = i128{a.mid_lo} - b.mid_lo - borrow;
    a.mid_lo = static_cast<std::uint64_t>(static_cast<unsigned __int128>(t));
    borrow = (t < 0) ? 1 : 0;
    t = i128{a.mid_hi} - b.mid_hi - borrow;
    a.mid_hi = static_cast<std::uint64_t>(static_cast<unsigned __int128>(t));
    borrow = (t < 0) ? 1 : 0;
    t = i128{a.hi} - b.hi - borrow;
    a.hi = static_cast<std::uint64_t>(static_cast<unsigned __int128>(t));
#pragma GCC diagnostic pop
}

double u256_to_double(const U256& a) {
    constexpr double TWO64  = 18446744073709551616.0;          // 2^64
    constexpr double TWO128 = TWO64 * TWO64;                    // 2^128
    constexpr double TWO192 = TWO128 * TWO64;                   // 2^192
    return static_cast<double>(a.lo)
         + TWO64  * static_cast<double>(a.mid_lo)
         + TWO128 * static_cast<double>(a.mid_hi)
         + TWO192 * static_cast<double>(a.hi);
}

namespace {

// Garner CRT setup: precomputes Π_{i<L} q_i for every level L ∈ [1..NUM_PRIMES]
// and the inverses needed for the mixed-radix lift.  We store one entry per
// level so the level-aware lift can pick the right Q (and the right Q/2 for
// centering) at zero runtime cost.
struct GarnerCtx {
    // partial[i] = Π_{j<i} q_j.  partial[NUM_PRIMES] = full Q.
    std::array<U256, NUM_PRIMES + 1> partial;
    // Q_at_level[L] = Π_{i<L} q_i (= partial[L]); Q_half_at_level[L] = ⌊Q/2⌋.
    std::array<U256, NUM_PRIMES + 1> Q_half_at_level;
    std::array<std::uint64_t, NUM_PRIMES * NUM_PRIMES> inv_table;
    // inv_table[i * NUM_PRIMES + j] = (q_j)^-1 mod q_i, used by Garner
    // when reducing the running sum.
};

void rshift1(U256& x) {
    const std::uint64_t hi_lsb = x.hi & 1ULL;
    const std::uint64_t mhi_lsb = x.mid_hi & 1ULL;
    const std::uint64_t mlo_lsb = x.mid_lo & 1ULL;
    x.hi     = x.hi >> 1;
    x.mid_hi = (x.mid_hi >> 1) | (hi_lsb << 63);
    x.mid_lo = (x.mid_lo >> 1) | (mhi_lsb << 63);
    x.lo     = (x.lo >> 1) | (mlo_lsb << 63);
}

GarnerCtx make_garner_ctx() {
    GarnerCtx ctx;
    ctx.partial[0] = U256{1, 0, 0, 0};
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ctx.partial[i + 1] = ctx.partial[i];
        u256_mul_u64(ctx.partial[i + 1], COEFF_MODULI[i]);
    }
    // Q_half at every level (Q_at_level[0] is degenerate but we never query
    // level 0; treat the entry as zero for safety).
    ctx.Q_half_at_level[0] = U256{0, 0, 0, 0};
    for (std::size_t L = 1; L <= NUM_PRIMES; ++L) {
        ctx.Q_half_at_level[L] = ctx.partial[L];
        rshift1(ctx.Q_half_at_level[L]);
    }
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            ctx.inv_table[i * NUM_PRIMES + j] =
                inv_mod(COEFF_MODULI[j] % COEFF_MODULI[i], COEFF_MODULI[i]);
        }
    }
    return ctx;
}

const GarnerCtx& garner_ctx() {
    static const GarnerCtx ctx = make_garner_ctx();
    return ctx;
}

}  // namespace

U256 crt_lift(const std::array<std::uint64_t, NUM_PRIMES>& r, std::size_t level) {
    const auto& ctx = garner_ctx();
    std::array<std::uint64_t, NUM_PRIMES> c{};
    c[0] = r[0] % COEFF_MODULI[0];
    for (std::size_t i = 1; i < level; ++i) {
        std::uint64_t partial = 0;
        std::uint64_t prod_mod_qi = 1;
        for (std::size_t j = 0; j < i; ++j) {
            partial = add_mod(partial, mul_mod(c[j], prod_mod_qi, COEFF_MODULI[i]),
                              COEFF_MODULI[i]);
            prod_mod_qi = mul_mod(prod_mod_qi, COEFF_MODULI[j] % COEFF_MODULI[i],
                                  COEFF_MODULI[i]);
        }
        std::uint64_t diff = sub_mod(r[i] % COEFF_MODULI[i], partial, COEFF_MODULI[i]);
        std::uint64_t inv_prod = 1;
        for (std::size_t j = 0; j < i; ++j) {
            inv_prod = mul_mod(inv_prod, ctx.inv_table[i * NUM_PRIMES + j],
                               COEFF_MODULI[i]);
        }
        c[i] = mul_mod(diff, inv_prod, COEFF_MODULI[i]);
    }
    U256 x{0, 0, 0, 0};
    for (std::size_t i = 0; i < level; ++i) {
        U256 term = ctx.partial[i];
        u256_mul_u64(term, c[i]);
        u256_add(x, term);
    }
    return x;
}

U256 crt_lift(const std::array<std::uint64_t, NUM_PRIMES>& r) {
    return crt_lift(r, NUM_PRIMES);
}

double crt_center_to_double(const U256& x, std::size_t level) {
    const auto& ctx = garner_ctx();
    const U256& Q_half = ctx.Q_half_at_level[level];
    const U256& Q = ctx.partial[level];
    if (u256_lt(Q_half, x)) {
        // x > Q/2 → negative branch.  Return (x − Q) as double.
        U256 negated = Q;
        u256_sub(negated, x);
        return -u256_to_double(negated);
    }
    return u256_to_double(x);
}

double crt_center_to_double(const U256& x) {
    return crt_center_to_double(x, NUM_PRIMES);
}

}  // namespace ssns::ckks
