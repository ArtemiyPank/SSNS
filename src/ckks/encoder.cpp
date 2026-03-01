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

#include <ssns/ckks/crt.hpp>
#include <ssns/ckks/modarith.hpp>
#include <ssns/ckks/params.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>

namespace ssns::ckks {

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
