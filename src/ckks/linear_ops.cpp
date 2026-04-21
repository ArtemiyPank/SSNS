// CKKS depth-0 linear operations — implementation.  See header for the
// op set and preconditions.
#include <ssns/ckks/linear_ops.hpp>

#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/ntt_ops.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/poly.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ssns::ckks {

namespace {

void check_compatible(const Ciphertext& a, const Ciphertext& b, const char* op) {
    if (a.scale != b.scale) {
        throw std::invalid_argument(
            std::string(op) + ": scale mismatch (" +
            std::to_string(a.scale) + " vs " + std::to_string(b.scale) + ")");
    }
    if (a.level != b.level) {
        throw std::invalid_argument(
            std::string(op) + ": level mismatch (" +
            std::to_string(a.level) + " vs " + std::to_string(b.level) + ")");
    }
}

void check_compatible(const Ciphertext& ct, const Plaintext& pt, const char* op) {
    if (ct.scale != pt.scale) {
        throw std::invalid_argument(
            std::string(op) + ": scale mismatch (" +
            std::to_string(ct.scale) + " vs " + std::to_string(pt.scale) + ")");
    }
    if (ct.level != pt.level) {
        throw std::invalid_argument(
            std::string(op) + ": level mismatch (" +
            std::to_string(ct.level) + " vs " + std::to_string(pt.level) + ")");
    }
}

}  // namespace

Ciphertext add(const Ciphertext& a, const Ciphertext& b) {
    check_compatible(a, b, "ckks::add");
    Ciphertext out;
    out.c0 = pointwise_add(a.c0, b.c0);
    out.c1 = pointwise_add(a.c1, b.c1);
    out.scale = a.scale;
    out.level = a.level;
    return out;
}

Ciphertext sub(const Ciphertext& a, const Ciphertext& b) {
    check_compatible(a, b, "ckks::sub");
    Ciphertext out;
    out.c0 = pointwise_sub(a.c0, b.c0);
    out.c1 = pointwise_sub(a.c1, b.c1);
    out.scale = a.scale;
    out.level = a.level;
    return out;
}

Ciphertext add_plain(const Ciphertext& ct, const Plaintext& pt) {
    check_compatible(ct, pt, "ckks::add_plain");
    Ciphertext out;
    out.c0 = pointwise_add(ct.c0, pt.poly);
    out.c1 = ct.c1;
    out.scale = ct.scale;
    out.level = ct.level;
    return out;
}

Ciphertext sub_plain(const Ciphertext& ct, const Plaintext& pt) {
    check_compatible(ct, pt, "ckks::sub_plain");
    Ciphertext out;
    out.c0 = pointwise_sub(ct.c0, pt.poly);
    out.c1 = ct.c1;
    out.scale = ct.scale;
    out.level = ct.level;
    return out;
}

namespace {

// Bump factor used by mul_scalar — see header "Scale arithmetic".  We pick
// 2^60 (a power of two) because:
//   * |scalar| ≤ 1 → |k| ≤ 2^60, well within int64 range with one bit to spare;
//   * 2^60 is close enough to the 60-bit prime COEFF_MODULI[NUM_PRIMES-1]
//     that a single rescale brings the scale back to ≈ 2^40.
constexpr int MUL_SCALAR_BUMP_BITS = 60;

// Reduce a signed 64-bit integer mod p.
std::uint64_t signed_mod(std::int64_t k, std::uint64_t p) noexcept {
    if (k >= 0) {
        return static_cast<std::uint64_t>(k) % p;
    }
    // k < 0: compute |k| mod p, then negate in the field.
    const std::uint64_t mag = static_cast<std::uint64_t>(-(k + 1)) + 1ULL;  // |k|, even when k = INT64_MIN
    const std::uint64_t r = mag % p;
    return r == 0 ? 0 : p - r;
}

// Modulus-drop helper for rescale: given a coefficient-form polynomial
// across all primes [0, level), drop residue index `level - 1` and adjust
// the surviving residues so they represent the same logical value scaled
// by inv(q_drop) mod q_i.  After the call, residues [0, level-1) hold the
// rescaled coefficients in coefficient form (still needs NTT-forward to
// be ready for arithmetic), and the dropped slot is zeroed.
void apply_modulus_drop(Polynomial& poly, std::size_t level) {
    const std::size_t drop_idx = level - 1;
    const std::uint64_t q_drop = COEFF_MODULI[drop_idx];
    const std::uint64_t half_drop = q_drop >> 1;

    // Pre-compute inv(q_drop, q_i) per surviving prime.
    std::array<std::uint64_t, NUM_PRIMES> inv_q_drop{};
    for (std::size_t i = 0; i < drop_idx; ++i) {
        inv_q_drop[i] = inv_mod(q_drop % COEFF_MODULI[i], COEFF_MODULI[i]);
    }

    auto& dropped = poly.residues[drop_idx];
    for (std::size_t j = 0; j < POLY_DEGREE; ++j) {
        const std::uint64_t r_drop = dropped[j];
        // Centered representation: if r_drop > q_drop/2, the value is
        // r_drop - q_drop (negative); we encode the negative as (q_i - mag)
        // mod q_i for each surviving prime.
        const bool negative = (r_drop > half_drop);
        const std::uint64_t mag = negative ? (q_drop - r_drop) : r_drop;
        for (std::size_t i = 0; i < drop_idx; ++i) {
            const std::uint64_t q_i = COEFF_MODULI[i];
            const std::uint64_t mag_i = mag % q_i;
            const std::uint64_t signed_i = negative
                ? (mag_i == 0 ? 0 : q_i - mag_i)  // represents -mag mod q_i
                : mag_i;
            // r_i' = (r_i - signed_i) * inv(q_drop) mod q_i
            const std::uint64_t diff = sub_mod(poly.residues[i][j], signed_i, q_i);
            poly.residues[i][j] = mul_mod(diff, inv_q_drop[i], q_i);
        }
        dropped[j] = 0;
    }
}

}  // namespace

Ciphertext mul_scalar(const Ciphertext& ct, double scalar) {
    // Encode scalar as a 60-bit signed integer k = round(scalar * 2^60).
    // |scalar| ≤ 1 keeps |k| ≤ 2^60, fitting comfortably in int64_t.
    const double bump = std::ldexp(1.0, MUL_SCALAR_BUMP_BITS);  // 2^60
    const double k_real = std::round(scalar * bump);
    const std::int64_t k = static_cast<std::int64_t>(k_real);

    Ciphertext out;
    out.scale = ct.scale * bump;
    out.level = ct.level;

    // PSM hoist: dispatch per prime so the inner mul_mod is constant-folded.
    constexpr std::uint64_t LO_C0 = (std::uint64_t{1} << 60) - COEFF_MODULI[0];
    constexpr std::uint64_t LO_C1 = (std::uint64_t{1} << 40) - COEFF_MODULI[1];
    constexpr std::uint64_t LO_C2 = (std::uint64_t{1} << 40) - COEFF_MODULI[2];
    constexpr std::uint64_t LO_C3 = (std::uint64_t{1} << 60) - COEFF_MODULI[3];
    auto run_prime = [&](auto P_v, auto C_v, auto IS60_v, std::size_t i) {
        constexpr std::uint64_t P  = decltype(P_v)::value;
        constexpr std::uint64_t C  = decltype(C_v)::value;
        constexpr bool         IS = decltype(IS60_v)::value;
        const std::uint64_t k_i = signed_mod(k, P);
        const auto& c0_i = ct.c0.residues[i];
        const auto& c1_i = ct.c1.residues[i];
        auto& out0 = out.c0.residues[i];
        auto& out1 = out.c1.residues[i];
        for (std::size_t j = 0; j < POLY_DEGREE; ++j) {
            if constexpr (IS) {
                out0[j] = mul_mod_psm60<P, C>(c0_i[j], k_i);
                out1[j] = mul_mod_psm60<P, C>(c1_i[j], k_i);
            } else {
                out0[j] = mul_mod_psm40<P, C>(c0_i[j], k_i);
                out1[j] = mul_mod_psm40<P, C>(c1_i[j], k_i);
            }
        }
    };
    run_prime(std::integral_constant<std::uint64_t, COEFF_MODULI[0]>{},
              std::integral_constant<std::uint64_t, LO_C0>{},
              std::true_type{},  0);
    run_prime(std::integral_constant<std::uint64_t, COEFF_MODULI[1]>{},
              std::integral_constant<std::uint64_t, LO_C1>{},
              std::false_type{}, 1);
    run_prime(std::integral_constant<std::uint64_t, COEFF_MODULI[2]>{},
              std::integral_constant<std::uint64_t, LO_C2>{},
              std::false_type{}, 2);
    run_prime(std::integral_constant<std::uint64_t, COEFF_MODULI[3]>{},
              std::integral_constant<std::uint64_t, LO_C3>{},
              std::true_type{},  3);
    return out;
}

Ciphertext mul_cipher(const Ciphertext& a,
                      const Ciphertext& b,
                      const EvalKey& evk,
                      const std::array<NTT, NUM_PRIMES>& ntts) {
    // Scale check — tolerate a tiny relative drift (post-rescale scales are
    // not powers of two, but they ARE deterministic, so any pair produced by
    // the same chain agrees to within machine epsilon).
    const double scale_diff = std::abs(a.scale - b.scale);
    if (scale_diff > 1e-6 * std::max(a.scale, 1.0)) {
        throw std::invalid_argument(
            std::string("ckks::mul_cipher: scale mismatch (") +
            std::to_string(a.scale) + " vs " + std::to_string(b.scale) + ")");
    }
    if (a.level != b.level) {
        throw std::invalid_argument(
            std::string("ckks::mul_cipher: level mismatch (") +
            std::to_string(a.level) + " vs " + std::to_string(b.level) + ")");
    }
    if (a.level < 1) {
        throw std::invalid_argument(
            "ckks::mul_cipher: level must be >= 1");
    }

    // Tensor expansion — all in NTT (frequency-domain) form.
    Polynomial d0 = pointwise_mul_ntt(a.c0, b.c0);
    Polynomial d1a = pointwise_mul_ntt(a.c0, b.c1);
    Polynomial d1b = pointwise_mul_ntt(a.c1, b.c0);
    Polynomial d1 = pointwise_add(d1a, d1b);
    Polynomial d2 = pointwise_mul_ntt(a.c1, b.c1);

    // ----- RNS-gadget relinearisation -----------------------------------
    // The naive BV-style relin (single (b_evk, a_evk)) leaves a noise term
    // d2 · e_evk mod Q with magnitude ~Q/2 — swamps the message.
    // The RNS-gadget version decomposes d2 across the prime chain so each
    // sub-key contributes a noise term bounded by ||d2_at_i||·N·σ ≈ q_i·N·σ
    // instead of Q·N·σ, recovering the message.  See eval_key.hpp.
    //
    // Step 1: bring d2 to coefficient form (so we can lift its slot-i
    //         residue to a centred integer per coefficient).
    Polynomial d2_coeff = d2;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ntts[i].inverse(d2_coeff.residues[i].data());
    }

    Polynomial c0_relin = d0;
    Polynomial c1_relin = d1;

    // Step 2: per gadget index i, build d2_at_i — the polynomial whose
    //         coefficient k equals d2_coeff.residues[i][k] lifted to a
    //         centred integer in (-q_i/2, q_i/2], then reduced mod each q_j.
    //         Forward-NTT and accumulate d2_at_i · sub_keys[i].b into c0_relin
    //         and d2_at_i · sub_keys[i].a into c1_relin.
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const std::uint64_t q_i = COEFF_MODULI[i];
        const std::uint64_t half_qi = q_i >> 1;

        Polynomial d2_at_i;  // RNS poly to be built across all NUM_PRIMES slots.
        const auto& src = d2_coeff.residues[i];
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            const std::uint64_t r = src[k];
            const bool negative = (r > half_qi);
            const std::uint64_t mag = negative ? (q_i - r) : r;
            for (std::size_t j = 0; j < NUM_PRIMES; ++j) {
                const std::uint64_t q_j = COEFF_MODULI[j];
                const std::uint64_t mag_j = mag % q_j;
                d2_at_i.residues[j][k] = negative
                    ? (mag_j == 0 ? 0 : q_j - mag_j)
                    : mag_j;
            }
        }
        // Forward-NTT each slot of d2_at_i.
        for (std::size_t j = 0; j < NUM_PRIMES; ++j) {
            ntts[j].forward(d2_at_i.residues[j].data());
        }

        // Accumulate.
        Polynomial term0 = pointwise_mul_ntt(d2_at_i, evk.sub_keys[i].b);
        Polynomial term1 = pointwise_mul_ntt(d2_at_i, evk.sub_keys[i].a);
        c0_relin.add_inplace(term0);
        c1_relin.add_inplace(term1);
    }

    Ciphertext out;
    out.c0 = std::move(c0_relin);
    out.c1 = std::move(c1_relin);
    out.scale = a.scale * b.scale;
    out.level = a.level;
    return out;
}

Ciphertext mul_plain(const Ciphertext& ct, const Plaintext& pt) {
    if (ct.level != pt.level) {
        throw std::invalid_argument(
            std::string("ckks::mul_plain: level mismatch (") +
            std::to_string(ct.level) + " vs " + std::to_string(pt.level) + ")");
    }
    Ciphertext out;
    out.c0 = pointwise_mul_ntt(ct.c0, pt.poly);
    out.c1 = pointwise_mul_ntt(ct.c1, pt.poly);
    out.scale = ct.scale * pt.scale;
    out.level = ct.level;  // == pt.level
    return out;
}

Ciphertext rescale(const Ciphertext& ct,
                   const std::array<NTT, NUM_PRIMES>& ntts) {
    if (ct.level < 2) {
        throw std::invalid_argument(
            "ckks::rescale: cannot rescale below 1 active prime (level=" +
            std::to_string(ct.level) + ")");
    }
    const std::size_t L = ct.level;
    const std::size_t drop_idx = L - 1;
    const std::uint64_t q_drop = COEFF_MODULI[drop_idx];

    Ciphertext out = ct;

    // 1. Inverse-NTT the active residues so we can reduce coefficient-wise
    //    in natural order.  Inactive residues [L, NUM_PRIMES) are not touched.
    for (std::size_t i = 0; i < L; ++i) {
        ntts[i].inverse(out.c0.residues[i].data());
        ntts[i].inverse(out.c1.residues[i].data());
    }

    // 2. Apply the modulus drop in coefficient form: subtract the
    //    centered residue mod q_drop and multiply by inv(q_drop).
    apply_modulus_drop(out.c0, L);
    apply_modulus_drop(out.c1, L);

    // 3. Re-NTT the surviving residues so subsequent ops see NTT form.
    for (std::size_t i = 0; i < drop_idx; ++i) {
        ntts[i].forward(out.c0.residues[i].data());
        ntts[i].forward(out.c1.residues[i].data());
    }

    // 4. Update bookkeeping: scale shrinks by q_drop, level drops by one.
    out.scale = ct.scale / static_cast<double>(q_drop);
    out.level = L - 1;
    return out;
}

}  // namespace ssns::ckks
