// In-tree double-precision matmul kernel.  Selected when the project is
// built with -DSSNS_USE_BLAS=OFF.  Replaces OpenBLAS cblas_dgemm on the
// algorithmic core (academic deliverable constraint).
//
// Layout
//   * Outer driver tiles (M, N, K) at (M_BLOCK, N_BLOCK, K_BLOCK) for L2.
//   * Inner microkernel: 6 rows x 8 cols, AVX2 + FMA, 12 ymm accumulators.
//     Reads one A scalar broadcast + two B ymm vectors per FMA pair, so the
//     k-loop body issues 12 FMAs against 3 loads — close to peak FMA
//     throughput on Intel client cores.
//   * Edge cases (M%6, N%8, missing AVX2 support) fall back to a scalar
//     triple-loop computed against the same partial-tile geometry.
//
// Numerical contract: bit-for-bit it is NOT identical to the reference loop
// (different summation order changes rounding).  Up to 1e-12 absolute
// difference is acceptable for double precision; tests assert this bound.
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define SSNS_HAS_X86_INTRIN 1
#else
#define SSNS_HAS_X86_INTRIN 0
#endif

namespace ssns::linalg {

namespace {

// L2-tile parameters.  Working set per K-panel:
//   A_tile: M_BLOCK * K_BLOCK doubles  (64 * 128 * 8 = 64 KiB)
//   B_tile: K_BLOCK * N_BLOCK doubles  (128 * 128 * 8 = 128 KiB)
//   C_tile: M_BLOCK * N_BLOCK doubles  (64 * 128 * 8 = 64 KiB)
// Sums to ~256 KiB, comfortably resident in the 1.25 MiB L2 of recent Intel
// client cores while leaving room for the rest of the working set.
constexpr std::size_t M_BLOCK = 96;
constexpr std::size_t N_BLOCK = 128;
constexpr std::size_t K_BLOCK = 128;

constexpr std::size_t MR = 6;   // microkernel rows
constexpr std::size_t NR = 8;   // microkernel cols (= 2 ymm * 4 doubles)

#if SSNS_HAS_X86_INTRIN
// AVX2 6x8 microkernel.  Computes a single MR x NR tile of C += A_panel * B_panel
// where A_panel is MR x kc (row-major, stride K) and B_panel is kc x NR
// (row-major, stride N).  Accumulators stay in registers across the whole
// kc-loop; C is loaded once at entry, stored once at exit.
__attribute__((target("avx2,fma")))
inline void microkernel_6x8(const double* A, std::size_t lda,
                            const double* B, std::size_t ldb,
                            double* C, std::size_t ldc, std::size_t kc) {
    __m256d c00 = _mm256_loadu_pd(C + 0 * ldc + 0);
    __m256d c01 = _mm256_loadu_pd(C + 0 * ldc + 4);
    __m256d c10 = _mm256_loadu_pd(C + 1 * ldc + 0);
    __m256d c11 = _mm256_loadu_pd(C + 1 * ldc + 4);
    __m256d c20 = _mm256_loadu_pd(C + 2 * ldc + 0);
    __m256d c21 = _mm256_loadu_pd(C + 2 * ldc + 4);
    __m256d c30 = _mm256_loadu_pd(C + 3 * ldc + 0);
    __m256d c31 = _mm256_loadu_pd(C + 3 * ldc + 4);
    __m256d c40 = _mm256_loadu_pd(C + 4 * ldc + 0);
    __m256d c41 = _mm256_loadu_pd(C + 4 * ldc + 4);
    __m256d c50 = _mm256_loadu_pd(C + 5 * ldc + 0);
    __m256d c51 = _mm256_loadu_pd(C + 5 * ldc + 4);

    for (std::size_t k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_loadu_pd(B + k * ldb + 0);
        const __m256d b1 = _mm256_loadu_pd(B + k * ldb + 4);

        const __m256d a0 = _mm256_broadcast_sd(A + 0 * lda + k);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);

        const __m256d a1 = _mm256_broadcast_sd(A + 1 * lda + k);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);

        const __m256d a2 = _mm256_broadcast_sd(A + 2 * lda + k);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c21 = _mm256_fmadd_pd(a2, b1, c21);

        const __m256d a3 = _mm256_broadcast_sd(A + 3 * lda + k);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c31 = _mm256_fmadd_pd(a3, b1, c31);

        const __m256d a4 = _mm256_broadcast_sd(A + 4 * lda + k);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c41 = _mm256_fmadd_pd(a4, b1, c41);

        const __m256d a5 = _mm256_broadcast_sd(A + 5 * lda + k);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
    }

    _mm256_storeu_pd(C + 0 * ldc + 0, c00);
    _mm256_storeu_pd(C + 0 * ldc + 4, c01);
    _mm256_storeu_pd(C + 1 * ldc + 0, c10);
    _mm256_storeu_pd(C + 1 * ldc + 4, c11);
    _mm256_storeu_pd(C + 2 * ldc + 0, c20);
    _mm256_storeu_pd(C + 2 * ldc + 4, c21);
    _mm256_storeu_pd(C + 3 * ldc + 0, c30);
    _mm256_storeu_pd(C + 3 * ldc + 4, c31);
    _mm256_storeu_pd(C + 4 * ldc + 0, c40);
    _mm256_storeu_pd(C + 4 * ldc + 4, c41);
    _mm256_storeu_pd(C + 5 * ldc + 0, c50);
    _mm256_storeu_pd(C + 5 * ldc + 4, c51);
}
#endif  // SSNS_HAS_X86_INTRIN

// Scalar fallback for partial tiles (mr < MR or nr < NR) and for the
// non-AVX2 build path.  Same C += A*B contract over an mr x nr region of
// length kc.
inline void microkernel_scalar(const double* A, std::size_t lda,
                               const double* B, std::size_t ldb,
                               double* C, std::size_t ldc, std::size_t kc,
                               std::size_t mr, std::size_t nr) {
    for (std::size_t i = 0; i < mr; ++i) {
        for (std::size_t k = 0; k < kc; ++k) {
            const double a_ik = A[i * lda + k];
            const double* b_row = B + k * ldb;
            double* c_row = C + i * ldc;
            for (std::size_t j = 0; j < nr; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

// Process one (M_BLOCK, K_BLOCK, N_BLOCK) tile of C in-place.  Assumes C is
// already initialised to the previous accumulated value (zero on the first
// call).  The driver below zeroes C up front and then sweeps k-tiles to add.
inline void macro_tile(const double* A, std::size_t lda,
                       const double* B, std::size_t ldb,
                       double* C, std::size_t ldc,
                       std::size_t mc, std::size_t nc, std::size_t kc,
                       bool use_avx2) {
    std::size_t i = 0;
    for (; i + MR <= mc; i += MR) {
        std::size_t j = 0;
        for (; j + NR <= nc; j += NR) {
#if SSNS_HAS_X86_INTRIN
            if (use_avx2) {
                microkernel_6x8(A + i * lda, lda,
                                B + j, ldb,
                                C + i * ldc + j, ldc,
                                kc);
                continue;
            }
#else
            (void)use_avx2;
#endif
            microkernel_scalar(A + i * lda, lda, B + j, ldb,
                               C + i * ldc + j, ldc, kc, MR, NR);
        }
        if (j < nc) {
            microkernel_scalar(A + i * lda, lda, B + j, ldb,
                               C + i * ldc + j, ldc, kc, MR, nc - j);
        }
    }
    if (i < mc) {
        const std::size_t mr = mc - i;
        std::size_t j = 0;
        for (; j + NR <= nc; j += NR) {
            microkernel_scalar(A + i * lda, lda, B + j, ldb,
                               C + i * ldc + j, ldc, kc, mr, NR);
        }
        if (j < nc) {
            microkernel_scalar(A + i * lda, lda, B + j, ldb,
                               C + i * ldc + j, ldc, kc, mr, nc - j);
        }
    }
}

inline bool host_supports_avx2_fma() {
#if SSNS_HAS_X86_INTRIN
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

}  // namespace

void matmul_native(const double* A, const double* B, double* C,
                   std::size_t M, std::size_t N, std::size_t K) {
    // C is already zero on entry (Matrix::zeros is the only call site).
    // Walk K-panels last so each (mc x nc) C-tile is reused across kc-tiles
    // and stays hot in cache.
    const bool use_avx2 = host_supports_avx2_fma();
    for (std::size_t i0 = 0; i0 < M; i0 += M_BLOCK) {
        const std::size_t mc = (i0 + M_BLOCK <= M) ? M_BLOCK : (M - i0);
        for (std::size_t j0 = 0; j0 < N; j0 += N_BLOCK) {
            const std::size_t nc = (j0 + N_BLOCK <= N) ? N_BLOCK : (N - j0);
            for (std::size_t k0 = 0; k0 < K; k0 += K_BLOCK) {
                const std::size_t kc = (k0 + K_BLOCK <= K) ? K_BLOCK : (K - k0);
                macro_tile(A + i0 * K + k0, K,
                           B + k0 * N + j0, N,
                           C + i0 * N + j0, N,
                           mc, nc, kc, use_avx2);
            }
        }
    }
}

}  // namespace ssns::linalg
