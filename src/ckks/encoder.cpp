// ckks canonical embedding impl see header
//
// encode pipeline (slots z in C^{N/2} -> poly m in Z_q[X]/(X^N+1))
//   z_full[k]       = z[k]         for k < N/2          (conjugate mirror)
//   z_full[N-1-k]   = conj(z[k])   for k < N/2
//   m_tilde = IFFT_N(z_full)
//   m_j = real(m_tilde_j * zeta^{-j})    zeta = exp(pi*i/N)
//   coeff_j = round(scale * m_j)
//   lift coeff_j into rns per prime
//
// decode is the matched inverse
//
// conjugate symmetry sigma_{N-1-k} = conj(sigma_k) is automatic when m has real coefs
// decoder ignores upper half
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

// pre compute zeta tables and fft twiddles
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

    // fft twiddles
    // at stage size m we need exp(+/- 2*pi*i * k / m) for k = 0..m/2-1
    // pre compute one entry per (m, k) in a flat array indexed by m + k
    //
    // ckks uses sigma_k = sum_j m_tilde_j * omega^(jk) with omega = exp(2*pi*i/N)
    // kernel sign is + opposite the standard forward dft
    // so fft(_, false) computes sigma direction (exp(+) twiddles)
    //    fft(_, true) computes m_tilde = DFT_A(sigma) / N (exp(-) with /N)
    // names fwd inv refer to encode decode direction not signal forward inverse
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

// in place bit reversal permutation used by both fft directions
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

// iterative radix 2 fft in place divides by N when inverse
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

// encode slot vector into coef form polynomial
Polynomial Encoder::encode(const std::vector<std::complex<double>>& z, double scale) const {
    constexpr std::size_t N = POLY_DEGREE;
    constexpr std::size_t H = N / 2;
    if (z.size() != H) {
        throw std::invalid_argument("Encoder::encode: slot vector must have length POLY_DEGREE/2");
    }
    if (!(scale > 0.0)) {
        throw std::invalid_argument("Encoder::encode: scale must be positive");
    }

    // mirror to length N with conjugate symmetry
    // conjugate mirror гарантирует что после ifft получим вещественные коэффициенты
    // иначе encoded poly содержал бы мнимую часть и крипто бы сломалось
    std::vector<std::complex<double>> z_full(N);
    for (std::size_t k = 0; k < H; ++k) {
        z_full[k] = z[k];
        z_full[N - 1 - k] = std::conj(z[k]);
    }

    // inverse N point dft gives twisted coefs m_tilde
    fft(z_full, /*inverse=*/true);

    // un twist m_j = real(m_tilde_j * zeta^{-j})
    // result is real because z_full was conjugate symmetric
    //
    // именно тут negacyclic structure появляется: умножение на zeta^{-j} = (e^{i*pi/N})^{-j}
    // это и есть twist для X^N+1 ring (X^N == -1)
    // без twist получили бы encoding для X^N-1 ring бесполезный для ckks
    Polynomial out;
    for (std::size_t j = 0; j < N; ++j) {
        const double m_real = (z_full[j] * zeta_pow_conj_[j]).real();
        const double scaled = std::round(scale * m_real);

        const bool negative = (scaled < 0.0);
        const double mag = negative ? -scaled : scaled;
        // mag fits in 64 bits for scale <= 2^53 and bounded slots
        const std::uint64_t mag_u = static_cast<std::uint64_t>(mag);
        for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
            const std::uint64_t q = COEFF_MODULI[i];
            const std::uint64_t r = mag_u % q;
            out.residues[i][j] = negative ? (r == 0 ? 0 : q - r) : r;
        }
    }
    return out;
}

// decode poly back into slots
std::vector<std::complex<double>> Encoder::decode(const Polynomial& p,
                                                  double scale,
                                                  std::size_t level) const {
    constexpr std::size_t N = POLY_DEGREE;
    constexpr std::size_t H = N / 2;
    if (!(scale > 0.0)) {
        throw std::invalid_argument("Encoder::decode: scale must be positive");
    }
    if (level == 0 || level > NUM_PRIMES) {
        throw std::invalid_argument("Encoder::decode: level must be in [1, NUM_PRIMES]");
    }

    const double inv_scale = 1.0 / scale;

    // lift each coef via garner crt center mod Q_level divide by scale
    // only first `level` residues participate
    std::vector<std::complex<double>> z_full(N);
    for (std::size_t j = 0; j < N; ++j) {
        std::array<std::uint64_t, NUM_PRIMES> r{};
        for (std::size_t i = 0; i < level; ++i) {
            r[i] = p.residues[i][j];
        }
        const U256 lifted = crt_lift(r, level);
        const double centered = crt_center_to_double(lifted, level);
        const double m_real = centered * inv_scale;
        // twist m_tilde_j = m_j * zeta^j (input real so imag is zero)
        // обратный twist для encode: тут zeta^j а в encode было zeta^{-j}
        z_full[j] = std::complex<double>(m_real, 0.0) * zeta_pow_[j];
    }

    // forward N point dft recovers slot values sigma_k = m(zeta^{2k+1})
    fft(z_full, /*inverse=*/false);

    // first N/2 entries are user visible slots upper half is conjugate mirror
    // верхнюю половину игнорируем потому что она просто conj первой
    std::vector<std::complex<double>> out(H);
    for (std::size_t k = 0; k < H; ++k) out[k] = z_full[k];
    return out;
}

}  // namespace ssns::ckks
