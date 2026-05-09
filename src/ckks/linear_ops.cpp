// ckks depth 0 linear ops impl see header
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

// throw if two ciphertexts disagree on scale or level
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

// throw if cipher and plaintext disagree on scale or level
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

// cipher + cipher
Ciphertext add(const Ciphertext& a, const Ciphertext& b) {
    check_compatible(a, b, "ckks::add");
    Ciphertext out;
    out.c0 = pointwise_add(a.c0, b.c0);
    out.c1 = pointwise_add(a.c1, b.c1);
    out.scale = a.scale;
    out.level = a.level;
    return out;
}

// cipher - cipher
Ciphertext sub(const Ciphertext& a, const Ciphertext& b) {
    check_compatible(a, b, "ckks::sub");
    Ciphertext out;
    out.c0 = pointwise_sub(a.c0, b.c0);
    out.c1 = pointwise_sub(a.c1, b.c1);
    out.scale = a.scale;
    out.level = a.level;
    return out;
}

// cipher + plaintext only c0 is touched
Ciphertext add_plain(const Ciphertext& ct, const Plaintext& pt) {
    check_compatible(ct, pt, "ckks::add_plain");
    Ciphertext out;
    out.c0 = pointwise_add(ct.c0, pt.poly);
    out.c1 = ct.c1;
    out.scale = ct.scale;
    out.level = ct.level;
    return out;
}

// cipher - plaintext only c0 is touched
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

// bump factor used by mul_scalar
// we pick 2^60 because
//   |scalar| <= 1 -> |k| <= 2^60 fits int64 with one bit to spare
//   2^60 is close to the 60 bit prime so one rescale brings scale back to ~2^40
constexpr int MUL_SCALAR_BUMP_BITS = 60;

// reduce signed 64 bit int mod p
std::uint64_t signed_mod(std::int64_t k, std::uint64_t p) noexcept {
    if (k >= 0) {
        return static_cast<std::uint64_t>(k) % p;
    }
    // k < 0 compute |k| mod p then negate in field
    // |k| even when k = INT64_MIN
    const std::uint64_t mag = static_cast<std::uint64_t>(-(k + 1)) + 1ULL;
    const std::uint64_t r = mag % p;
    return r == 0 ? 0 : p - r;
}

// modulus drop helper for rescale
// given coef form poly across all primes [0, level) drop residue index level - 1
// adjust surviving residues so they represent same value scaled by inv(q_drop) mod q_i
// after the call residues [0, level-1) hold rescaled coefs in coef form (still need ntt forward)
// dropped slot is zeroed
//
// идея rescale: x mod Q -> floor(x / q_drop) mod (Q / q_drop)
// напрямую делить нельзя поэтому subtract centered residue mod q_drop сначала
// чтобы получить число делящееся на q_drop
// потом умножаем на inv(q_drop) в каждом surviving prime
// math: (x - x mod q_drop) ≡ 0 mod q_drop значит y такой что (x - x mod q_drop) = q_drop * y
// и y mod q_i = (x - x mod q_drop) * inv(q_drop) mod q_i
void apply_modulus_drop(Polynomial& poly, std::size_t level) {
    const std::size_t drop_idx = level - 1;
    const std::uint64_t q_drop = COEFF_MODULI[drop_idx];
    const std::uint64_t half_drop = q_drop >> 1;

    // pre compute inv(q_drop, q_i) per surviving prime
    std::array<std::uint64_t, NUM_PRIMES> inv_q_drop{};
    for (std::size_t i = 0; i < drop_idx; ++i) {
        inv_q_drop[i] = inv_mod(q_drop % COEFF_MODULI[i], COEFF_MODULI[i]);
    }

    auto& dropped = poly.residues[drop_idx];
    for (std::size_t j = 0; j < POLY_DEGREE; ++j) {
        const std::uint64_t r_drop = dropped[j];
        // centered repr if r_drop > q_drop/2 the value is r_drop - q_drop (negative)
        // encode negative as (q_i - mag) mod q_i for each surviving prime
        const bool negative = (r_drop > half_drop);
        const std::uint64_t mag = negative ? (q_drop - r_drop) : r_drop;
        for (std::size_t i = 0; i < drop_idx; ++i) {
            const std::uint64_t q_i = COEFF_MODULI[i];
            const std::uint64_t mag_i = mag % q_i;
            const std::uint64_t signed_i = negative
                ? (mag_i == 0 ? 0 : q_i - mag_i)  // represents -mag mod q_i
                : mag_i;
            // r_i' = (r_i - signed_i) * inv(q_drop) mod q_i
            // вычитание делает coef делящимся на q_drop потом mul на inv эквивалентно делению
            const std::uint64_t diff = sub_mod(poly.residues[i][j], signed_i, q_i);
            poly.residues[i][j] = mul_mod(diff, inv_q_drop[i], q_i);
        }
        dropped[j] = 0;
    }
}

}  // namespace

// multiply ct by real scalar encoding scalar as 60 bit signed int for exact per prime arithmetic
Ciphertext mul_scalar(const Ciphertext& ct, double scalar) {
    // encode scalar as 60 bit signed int k = round(scalar * 2^60)
    // |scalar| <= 1 keeps |k| <= 2^60 fits int64
    const double bump = std::ldexp(1.0, MUL_SCALAR_BUMP_BITS);  // 2^60
    const double k_real = std::round(scalar * bump);
    const std::int64_t k = static_cast<std::int64_t>(k_real);

    Ciphertext out;
    out.scale = ct.scale * bump;
    out.level = ct.level;

    // per prime psm reduced multiply by scalar
    // templated so mul_mod_psm sees P and C as compile time constants
    auto run_prime = [&]<std::uint64_t P, std::uint64_t C, bool IS60>(std::size_t i) {
        const std::uint64_t k_i = signed_mod(k, P);
        const auto& c0_i = ct.c0.residues[i];
        const auto& c1_i = ct.c1.residues[i];
        auto& out0 = out.c0.residues[i];
        auto& out1 = out.c1.residues[i];
        for (std::size_t j = 0; j < POLY_DEGREE; ++j) {
            if constexpr (IS60) {
                out0[j] = mul_mod_psm60<P, C>(c0_i[j], k_i);
                out1[j] = mul_mod_psm60<P, C>(c1_i[j], k_i);
            } else {
                out0[j] = mul_mod_psm40<P, C>(c0_i[j], k_i);
                out1[j] = mul_mod_psm40<P, C>(c1_i[j], k_i);
            }
        }
    };
    run_prime.template operator()<COEFF_MODULI[0], PSM_C[0], true >(0);
    run_prime.template operator()<COEFF_MODULI[1], PSM_C[1], false>(1);
    run_prime.template operator()<COEFF_MODULI[2], PSM_C[2], false>(2);
    run_prime.template operator()<COEFF_MODULI[3], PSM_C[3], true >(3);
    return out;
}

// cipher x cipher with rns gadget relin
Ciphertext mul_cipher(const Ciphertext& a,
                      const Ciphertext& b,
                      const EvalKey& evk,
                      const std::array<NTT, NUM_PRIMES>& ntts) {
    // scale check tolerates tiny relative drift
    // post rescale scales are not powers of two but ARE deterministic so any pair from same chain agrees within machine eps
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

    // tensor expansion all in ntt form
    // (c0 + c1*s) * (c0' + c1'*s) даёт 3-полиномиальный ciphertext (d0, d1, d2)
    // d0 = c0*c0', d1 = c0*c1' + c1*c0', d2 = c1*c1' (коэффициент при s^2)
    Polynomial d0 = pointwise_mul_ntt(a.c0, b.c0);
    Polynomial d1a = pointwise_mul_ntt(a.c0, b.c1);
    Polynomial d1b = pointwise_mul_ntt(a.c1, b.c0);
    Polynomial d1 = pointwise_add(d1a, d1b);
    Polynomial d2 = pointwise_mul_ntt(a.c1, b.c1);

    // rns gadget relin
    // naive bv style relin leaves noise term d2 * e_evk mod Q with magnitude ~Q/2 swamping the message
    // gadget version decomposes d2 across prime chain so each sub key noise is bounded by q_i not Q
    //
    // тут critical: d2_at_i имеет coefs ограниченные q_i/2 не Q/2 потому что лифтим только slot i
    // sum_i d2_at_i * e_i = d2 в crt (где e_i это rns basis indicator)
    //
    // step 1 bring d2 to coef form so we can lift slot i residue to centered int
    Polynomial d2_coeff = d2;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        ntts[i].inverse(d2_coeff.residues[i].data());
    }

    Polynomial c0_relin = d0;
    Polynomial c1_relin = d1;

    // step 2 per gadget index i build d2_at_i
    // coef k of d2_at_i equals d2_coeff.residues[i][k] lifted to centered int reduced mod each q_j
    // forward ntt and accumulate d2_at_i * sub_keys[i].b into c0_relin
    //                            d2_at_i * sub_keys[i].a into c1_relin
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        const std::uint64_t q_i = COEFF_MODULI[i];
        const std::uint64_t half_qi = q_i >> 1;

        // rns poly across all NUM_PRIMES slots
        // тут берём ОДИН slot d2 (mod q_i) и распихиваем в ВСЕ q_j через signed lift
        // чтобы он был корректно представим как маленькое число mod каждого prime
        Polynomial d2_at_i;
        const auto& src = d2_coeff.residues[i];
        for (std::size_t k = 0; k < POLY_DEGREE; ++k) {
            const std::uint64_t r = src[k];
            // centered: r > q_i/2 значит представляли отрицательное r-q_i
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
        // forward ntt each slot of d2_at_i
        for (std::size_t j = 0; j < NUM_PRIMES; ++j) {
            ntts[j].forward(d2_at_i.residues[j].data());
        }

        // accumulate into c0_relin and c1_relin
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

// cipher x plaintext pointwise on c0 and c1
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

// drop highest active prime divide scale by it reduce level by 1
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

    // 1 inverse ntt active residues so we can reduce coef wise in natural order
    //   inactive residues [L, NUM_PRIMES) not touched
    for (std::size_t i = 0; i < L; ++i) {
        ntts[i].inverse(out.c0.residues[i].data());
        ntts[i].inverse(out.c1.residues[i].data());
    }

    // 2 apply modulus drop in coef form
    apply_modulus_drop(out.c0, L);
    apply_modulus_drop(out.c1, L);

    // 3 re ntt surviving residues so subsequent ops see ntt form
    for (std::size_t i = 0; i < drop_idx; ++i) {
        ntts[i].forward(out.c0.residues[i].data());
        ntts[i].forward(out.c1.residues[i].data());
    }

    // 4 update bookkeeping scale shrinks by q_drop level drops by one
    // именно поэтому primes выбраны близко к 2^40: scale после rescale ~ 2^scale_bits / 2^40
    out.scale = ct.scale / static_cast<double>(q_drop);
    out.level = L - 1;
    return out;
}

}  // namespace ssns::ckks
