// CKKS canonical embedding — implementation.  See header for the math.
//
// Encode pipeline (slots z ∈ C^{N/2}  →  polynomial m ∈ Z_q[X]/(X^N+1)):
//
//   z_full[k]       = z[k]         for k < N/2          ┐
//   z_full[N-1-k]   = conj(z[k])   for k < N/2          ┘  conjugate-mirror
//   m̃ = IFFT_N(z_full)                                  // length-N inverse DFT
//   m_j = real(m̃_j · ζ^{-j})        ζ = exp(πi/N)         // un-twist
//   coeff_j = round(scale · m_j)                          // quantise
//   lift coeff_j into RNS form per CKKS prime              // Polynomial
//
// The forward (decode) direction is the matched inverse:
//
//   coeff_j ← signed integer obtained by Garner CRT on residues
//   m_j     = coeff_j / scale
//   m̃_j     = m_j · ζ^j                                    // twist
//   z_full  = FFT_N(m̃)
//   z[k]    = z_full[k]   for k < N/2                      // output slots
//
// The conjugate symmetry σ_{N-1-k} = conj(σ_k) is automatic when m has
// real coefficients, so the decoder ignores the upper half.
#include <ssns/ckks/encoder.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/params.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>

namespace ssns::ckks {

namespace {

// CRT product Q = Π q_i and per-prime constants used by Garner's
// algorithm to lift an RNS residue tuple back to a signed integer mod Q.
// Q is on the order of 2^200, so we need a small bigint helper.  We
// implement the lift in floating-point using "double-double" tricks?
// Simpler: keep partial sum as double; for our scale 2^40 the encoded
// integer fits in ~60 bits, well within double-precision once we recover
// the centered residue mod Q.
//
// Actually we don't need full Q: by CKKS construction the encoded
// coefficient `round(scale · m)` has magnitude ≤ scale · max|slots| · N
// — at scale ≈ 2^40 and unit slots that's ≈ 2^53.  We can read it from
// just the smallest CKKS prime (which is 40 bits) iff we recover the
// centered representation mod q1 — but that fails as soon as |coeff| >
// q1/2.  Solution: use Garner's algorithm to combine all primes into the
// full integer, then center mod Q.  We carry the value as long double /
// __float128 / a tiny bigint.  Here we use a lightweight approach: hold
// the running sum as a small array of 64-bit limbs, then convert to
// double at the end (with sign).
//
// Magnitude bound: log2(scale) + log2(N) + log2(max|slot|) ≈ 40 + 13 + 0
// ≈ 53 bits, plus a few bits margin from Δ² in mul scale.  At up to ~120
// bits we still fit in 2 × uint64.  Phase-6 cipher mul will push us
// further; we provide enough headroom for 4 limbs (240+ bits) which
// covers the full Q regardless.
struct U256 {
    std::uint64_t lo{0}, mid_lo{0}, mid_hi{0}, hi{0};
};

constexpr std::size_t LIMBS = 4;

inline void u256_add(U256& a, const U256& b) {
    unsigned __int128 carry = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    using u128 = unsigned __int128;
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

inline void u256_mul_u64(U256& a, std::uint64_t m) {
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

inline bool u256_lt(const U256& a, const U256& b) {
    if (a.hi != b.hi)         return a.hi < b.hi;
    if (a.mid_hi != b.mid_hi) return a.mid_hi < b.mid_hi;
    if (a.mid_lo != b.mid_lo) return a.mid_lo < b.mid_lo;
    return a.lo < b.lo;
}

inline void u256_sub(U256& a, const U256& b) {
    // a -= b, assuming a >= b.
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

// Convert a U256 into a double, treating it as unsigned.  Used by the
// CRT lift after centering.
inline double u256_to_double(const U256& a) {
    constexpr double TWO64  = 18446744073709551616.0;          // 2^64
    constexpr double TWO128 = TWO64 * TWO64;                    // 2^128
    constexpr double TWO192 = TWO128 * TWO64;                   // 2^192
    return static_cast<double>(a.lo)
         + TWO64  * static_cast<double>(a.mid_lo)
         + TWO128 * static_cast<double>(a.mid_hi)
         + TWO192 * static_cast<double>(a.hi);
}

// Garner CRT setup: precomputes Π q_i and the inverses needed for the
// mixed-radix lift.
struct GarnerCtx {
    U256 Q;                                                      // product
    U256 Q_half;                                                 // Q/2 — center boundary
    std::array<U256, NUM_PRIMES> partial;                        // Π_{j<i} q_j
    std::array<std::uint64_t, NUM_PRIMES * NUM_PRIMES> inv_table;
    // inv_table[i * NUM_PRIMES + j] = (q_j)^-1 mod q_i, used by Garner
    // when reducing the running sum.
};

GarnerCtx make_garner_ctx() {
    GarnerCtx ctx;
    // Q = Π q_i, partial[i] = Π_{j<i} q_j.
    ctx.partial[0] = U256{1, 0, 0, 0};
    for (std::size_t i = 1; i < NUM_PRIMES; ++i) {
        ctx.partial[i] = ctx.partial[i - 1];
        u256_mul_u64(ctx.partial[i], COEFF_MODULI[i - 1]);
    }
    ctx.Q = ctx.partial[NUM_PRIMES - 1];
    u256_mul_u64(ctx.Q, COEFF_MODULI[NUM_PRIMES - 1]);
    ctx.Q_half = ctx.Q;
    // Q / 2: shift right by 1.  Q is even iff one q_i is even, but all
    // CKKS primes are odd, so Q is odd → integer division floors.
    {
        std::uint64_t carry = 0;
        std::uint64_t newhi = (ctx.Q_half.hi >> 1) | (carry << 63);
        std::uint64_t hi_lsb = ctx.Q_half.hi & 1ULL;
        std::uint64_t new_mid_hi = (ctx.Q_half.mid_hi >> 1) | (hi_lsb << 63);
        std::uint64_t mhi_lsb = ctx.Q_half.mid_hi & 1ULL;
        std::uint64_t new_mid_lo = (ctx.Q_half.mid_lo >> 1) | (mhi_lsb << 63);
        std::uint64_t mlo_lsb = ctx.Q_half.mid_lo & 1ULL;
        std::uint64_t new_lo = (ctx.Q_half.lo >> 1) | (mlo_lsb << 63);
        ctx.Q_half.hi = newhi;
        ctx.Q_half.mid_hi = new_mid_hi;
        ctx.Q_half.mid_lo = new_mid_lo;
        ctx.Q_half.lo = new_lo;
    }
    // inv_table[i, j] = (q_j)^-1 mod q_i for j < i.  Other entries unused.
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

// Mixed-radix Garner lift: given residues r[i] = x mod q_i, recover the
// integer x ∈ [0, Q).  Output is in U256.
//
// Garner's algorithm:
//   c_0 = r_0
//   c_i = ((r_i - x_{i-1}) · (Π_{j<i} q_j)^-1) mod q_i      // mixed-radix digit
//   x_i = x_{i-1} + c_i · (Π_{j<i} q_j)
//
// We expand on each level and accumulate into U256.
U256 crt_lift(const std::array<std::uint64_t, NUM_PRIMES>& r) {
    const auto& ctx = garner_ctx();
    // mixed-radix digits c_i
    std::array<std::uint64_t, NUM_PRIMES> c{};
    c[0] = r[0] % COEFF_MODULI[0];
    for (std::size_t i = 1; i < NUM_PRIMES; ++i) {
        // x_{i-1} mod q_i = Σ_{j<i} c_j · (Π_{k<j} q_k)  mod q_i
        std::uint64_t partial = 0;
        std::uint64_t prod_mod_qi = 1;
        for (std::size_t j = 0; j < i; ++j) {
            // partial += c[j] · prod_mod_qi
            partial = add_mod(partial, mul_mod(c[j], prod_mod_qi, COEFF_MODULI[i]),
                              COEFF_MODULI[i]);
            prod_mod_qi = mul_mod(prod_mod_qi, COEFF_MODULI[j] % COEFF_MODULI[i],
                                  COEFF_MODULI[i]);
        }
        // c_i = (r_i - x_{i-1}) * (Π_{j<i} q_j)^-1  mod q_i
        std::uint64_t diff = sub_mod(r[i] % COEFF_MODULI[i], partial, COEFF_MODULI[i]);
        // (Π_{j<i} q_j)^-1 mod q_i — built from inv_table by iterated multiplication
        std::uint64_t inv_prod = 1;
        for (std::size_t j = 0; j < i; ++j) {
            inv_prod = mul_mod(inv_prod, ctx.inv_table[i * NUM_PRIMES + j],
                               COEFF_MODULI[i]);
        }
        c[i] = mul_mod(diff, inv_prod, COEFF_MODULI[i]);
    }
    // Build x = Σ c_i · (Π_{j<i} q_j) as U256.
    U256 x{0, 0, 0, 0};
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        U256 term = ctx.partial[i];
        u256_mul_u64(term, c[i]);
        u256_add(x, term);
    }
    return x;
}

// Center the lifted integer: if x > Q/2, it represents a negative number
// x - Q.  Returns the centered value as a double (signed).
double crt_center_to_double(const U256& x) {
    const auto& ctx = garner_ctx();
    if (u256_lt(ctx.Q_half, x)) {
        // x > Q/2 → negative branch.  Return (x - Q) as double.
        // Since |x - Q| < Q/2, fits in same magnitude.
        U256 negated = ctx.Q;
        u256_sub(negated, x);
        return -u256_to_double(negated);
    }
    return u256_to_double(x);
}

}  // namespace

Encoder::Encoder() {
    constexpr std::size_t N = POLY_DEGREE;
    const double pi_over_N = std::numbers::pi_v<double> / static_cast<double>(N);

    zeta_pow_.resize(N);
    zeta_pow_conj_.resize(N);
    for (std::size_t k = 0; k < N; ++k) {
        const double angle = pi_over_N * static_cast<double>(k);
        zeta_pow_[k] = std::complex<double>(std::cos(angle), std::sin(angle));
        zeta_pow_conj_[k] = std::conj(zeta_pow_[k]);
    }

    // FFT twiddles: at stage of size m, we need exp(±2πi · k / m) for
    // k = 0..m/2-1.  We pre-compute one entry per (m, k) in a flat array
    // indexed by m + k where m is a power of two — the layout is the
    // same as the standard "iterative Cooley-Tukey" textbook.
    // CKKS canonical embedding uses σ_k = Σ_j m̃_j · ω^(jk) with ω=exp(2πi/N) —
    // the kernel sign is +, opposite the standard "forward DFT" convention.
    // Compose so that fft(_,false) computes the σ-direction (exp(+2πi·k/m)
    // twiddles) and fft(_,true) computes m̃ = DFT_A(σ)/N (exp(-2πi·k/m), with
    // /N normalization). The names "fwd" / "inv" thus refer to encode / decode
    // direction, not signal-processing forward / inverse.
    twiddle_fwd_.resize(N);
    twiddle_inv_.resize(N);
    for (std::size_t m = 2; m <= N; m <<= 1) {
        const double base_angle = 2.0 * std::numbers::pi_v<double> / static_cast<double>(m);
        const std::size_t half = m / 2;
        for (std::size_t k = 0; k < half; ++k) {
            const double a = base_angle * static_cast<double>(k);
            twiddle_fwd_[half + k] = std::complex<double>(std::cos(a), std::sin(a));
            twiddle_inv_[half + k] = std::complex<double>(std::cos(-a), std::sin(-a));
        }
    }
}

void Encoder::bitreverse_permute(std::vector<std::complex<double>>& a) const {
    const std::size_t N = a.size();
    const int log_N = std::bit_width(N) - 1;
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t j = 0;
        std::size_t x = i;
        for (int b = 0; b < log_N; ++b) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        if (j > i) std::swap(a[i], a[j]);
    }
}

void Encoder::fft(std::vector<std::complex<double>>& a, bool inverse) const {
    const std::size_t N = a.size();
    bitreverse_permute(a);
    const auto& tw = inverse ? twiddle_inv_ : twiddle_fwd_;
    for (std::size_t m = 2; m <= N; m <<= 1) {
        const std::size_t half = m / 2;
        for (std::size_t k = 0; k < N; k += m) {
            for (std::size_t j = 0; j < half; ++j) {
                const auto t = tw[half + j] * a[k + j + half];
                const auto u = a[k + j];
                a[k + j]        = u + t;
                a[k + j + half] = u - t;
            }
        }
    }
    if (inverse) {
        const double inv_N = 1.0 / static_cast<double>(N);
        for (auto& x : a) x *= inv_N;
    }
}

Polynomial Encoder::encode(const std::vector<std::complex<double>>& z, double scale) const {
    constexpr std::size_t N = POLY_DEGREE;
    constexpr std::size_t H = N / 2;
    if (z.size() != H) {
        throw std::invalid_argument("Encoder::encode: slot vector must have length POLY_DEGREE/2");
    }
    if (!(scale > 0.0)) {
        throw std::invalid_argument("Encoder::encode: scale must be positive");
    }

    // Mirror to length-N with conjugate symmetry: z_full[k] = z[k] for
    // k < N/2 and z_full[N-1-k] = conj(z[k]).
    std::vector<std::complex<double>> z_full(N);
    for (std::size_t k = 0; k < H; ++k) {
        z_full[k] = z[k];
        z_full[N - 1 - k] = std::conj(z[k]);
    }

    // Inverse N-point DFT — gives the twisted coefficients m̃.
    fft(z_full, /*inverse=*/true);

    // Un-twist: m_j = real(m̃_j · ζ^{-j}).  Because z_full was conjugate-
    // symmetric the result is (numerically) real.
    Polynomial out;
    for (std::size_t j = 0; j < N; ++j) {
        const double m_real = (z_full[j] * zeta_pow_conj_[j]).real();
        const double scaled = std::round(scale * m_real);

        const bool negative = (scaled < 0.0);
        const double mag = negative ? -scaled : scaled;
        // mag fits comfortably in 64 bits for scale ≤ 2^53 and bounded slots.
        const std::uint64_t mag_u = static_cast<std::uint64_t>(mag);
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            const std::uint64_t r = mag_u % q;
            out.residues[i][j] = negative ? (r == 0 ? 0 : q - r) : r;
        }
    }
    return out;
}

std::vector<std::complex<double>> Encoder::decode(const Polynomial& p, double scale) const {
    constexpr std::size_t N = POLY_DEGREE;
    constexpr std::size_t H = N / 2;
    if (!(scale > 0.0)) {
        throw std::invalid_argument("Encoder::decode: scale must be positive");
    }

    const double inv_scale = 1.0 / scale;

    // Lift each coefficient via Garner CRT, center mod Q, divide by scale.
    std::vector<std::complex<double>> z_full(N);
    for (std::size_t j = 0; j < N; ++j) {
        std::array<std::uint64_t, NUM_PRIMES> r;
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            r[i] = p.residues[i][j];
        }
        const U256 lifted = crt_lift(r);
        const double centered = crt_center_to_double(lifted);
        const double m_real = centered * inv_scale;
        // Twist: m̃_j = m_j · ζ^j (input is real, so imaginary part is zero).
        z_full[j] = std::complex<double>(m_real, 0.0) * zeta_pow_[j];
    }

    // Forward N-point DFT — recovers slot values σ_k = m(ζ^{2k+1}).
    fft(z_full, /*inverse=*/false);

    // First N/2 entries are the user-visible slots; the upper half is the
    // conjugate mirror.
    std::vector<std::complex<double>> out(H);
    for (std::size_t k = 0; k < H; ++k) out[k] = z_full[k];
    return out;
}

}  // namespace ssns::ckks
