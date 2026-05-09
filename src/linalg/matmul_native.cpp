// in tree f64 matmul
// no blas just our own kernel
// avx2+fma microkernel inside cache tiles
// not bit exact vs naive loop diff up to 1e-12 ok
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define SSNS_HAS_X86_INTRIN 1
#else
#define SSNS_HAS_X86_INTRIN 0
#endif

namespace ssns::linalg {

namespace {

// l2 tile sizes picked to fit working set in l2
constexpr std::size_t M_BLOCK = 96;
constexpr std::size_t N_BLOCK = 128;
constexpr std::size_t K_BLOCK = 128;

// register block for microkernel
// 6x8 because avx2 has 16 ymm
// 12 идут под аккумуляторы 4 под загрузки
constexpr std::size_t MR = 6;   // rows
constexpr std::size_t NR = 8;   // cols = 2 ymm * 4 doubles

#if SSNS_HAS_X86_INTRIN
// avx2 6x8 kernel
// C += A_panel * B_panel
// accs stay in regs whole kc loop
// k loop body 12 fmas vs 3 loads
// тут именно 6x8 12 ymm под акк 4 под загрузки
// arithmetic intensity 12 fma на 3 ymm load
__attribute__((target("avx2,fma")))
inline void microkernel_6x8(const double* A, std::size_t lda,
                            const double* B, std::size_t ldb,
                            double* C, std::size_t ldc, std::size_t kc) {
    // load 6x8 of C into 12 ymm
    // loadu unaligned safe
    // C аккумулируется в регистрах не трогаем память в k-loop
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

    // основной k loop
    // берём один срез B 8 doubles
    // broadcast a[i,k] из 6 строк
    // load B один раз потом 6 broadcasts + 12 fma
    for (std::size_t k = 0; k < kc; ++k) {
        // load one k slice of B 8 doubles
        const __m256d b0 = _mm256_loadu_pd(B + k * ldb + 0);
        const __m256d b1 = _mm256_loadu_pd(B + k * ldb + 4);

        // broadcast scalar A[i,k] into all 4 lanes
        // broadcast a_ik в lane всех 4 элементов ymm
        const __m256d a0 = _mm256_broadcast_sd(A + 0 * lda + k);
        // fma is mul+add in one op
        // fma c += a*b за один такт ровно
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

    // single store back to C
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

// scalar fallback
// for partial tiles and no avx2 build
// i k j order so a_ik hoisted out
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

// process one tile mc x nc x kc
// full MR x NR blocks go to avx2
// edges go to scalar fallback
// хвосты по N справа и по M снизу через скаляр иначе UB на loadu
inline void macro_tile(const double* A, std::size_t lda,
                       const double* B, std::size_t ldb,
                       double* C, std::size_t ldc,
                       std::size_t mc, std::size_t nc, std::size_t kc,
                       bool use_avx2) {
    // main pass full MR x NR blocks
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
        // right tail along N
        if (j < nc) {
            microkernel_scalar(A + i * lda, lda, B + j, ldb,
                               C + i * ldc + j, ldc, kc, MR, nc - j);
        }
    }
    // bottom tail along M all scalar
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

// true if cpu has avx2+fma
// cheap cpuid check
inline bool host_supports_avx2_fma() {
#if SSNS_HAS_X86_INTRIN
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

}  // namespace

// sequential matmul over rows [i_lo, i_hi)
// disjoint output range so threads dont sync
// reads of B shared and read only
// порядок M-N-K чтобы B держался в L1 для каждого панели A
inline void matmul_row_slab(const double* A, const double* B, double* C,
                            std::size_t i_lo, std::size_t i_hi,
                            std::size_t N, std::size_t K, bool use_avx2) {
    for (std::size_t i0 = i_lo; i0 < i_hi; i0 += M_BLOCK) {
        const std::size_t mc = (i0 + M_BLOCK <= i_hi) ? M_BLOCK : (i_hi - i0);
        for (std::size_t j0 = 0; j0 < N; j0 += N_BLOCK) {
            const std::size_t nc = (j0 + N_BLOCK <= N) ? N_BLOCK : (N - j0);
            for (std::size_t k0 = 0; k0 < K; k0 += K_BLOCK) {
                const std::size_t kc =
                    (k0 + K_BLOCK <= K) ? K_BLOCK : (K - k0);
                macro_tile(A + i0 * K + k0, K,
                           B + k0 * N + j0, N,
                           C + i0 * N + j0, N,
                           mc, nc, kc, use_avx2);
            }
        }
    }
}

// below this many flops thread spawn not worth it
constexpr std::size_t PARALLEL_FLOP_THRESHOLD = 1ULL << 19;  // 512K mults

// C = A*B row major MxK KxN MxN
// threads rows of C if big enough
// each worker owns disjoint C rows no contention
// std::thread не openmp потому что хочется явный контроль
// строки C disjoint между потоками не нужно sync
void matmul_native(const double* A, const double* B, double* C,
                   std::size_t M, std::size_t N, std::size_t K) {
    // C is zero on entry
    const bool use_avx2 = host_supports_avx2_fma();
    const std::size_t work = M * N * K;

    // small problems just go sequential
    if (work < PARALLEL_FLOP_THRESHOLD || M < 2 * MR) {
        matmul_row_slab(A, B, C, 0, M, N, K, use_avx2);
        return;
    }

    // partition M into chunks aligned to MR
    // last worker takes the tail
    // cap threads at M/MR
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    std::size_t n_threads = static_cast<std::size_t>(hw);
    const std::size_t max_workers_by_rows = (M + MR - 1) / MR;
    if (n_threads > max_workers_by_rows) n_threads = max_workers_by_rows;
    if (n_threads < 1) n_threads = 1;
    if (n_threads == 1) {
        matmul_row_slab(A, B, C, 0, M, N, K, use_avx2);
        return;
    }

    // even partition by MR aligned chunks
    // last worker eats the remainder
    // выравнивание по MR чтобы микрокернел работал на полных 6-строчных панелях
    const std::size_t mr_blocks = (M + MR - 1) / MR;
    const std::size_t blocks_per_worker = mr_blocks / n_threads;
    const std::size_t blocks_remainder  = mr_blocks % n_threads;

    std::vector<std::thread> workers;
    workers.reserve(n_threads - 1);
    std::size_t start_block = 0;
    for (std::size_t t = 0; t < n_threads; ++t) {
        const std::size_t this_blocks =
            blocks_per_worker + (t < blocks_remainder ? 1 : 0);
        const std::size_t i_lo = start_block * MR;
        std::size_t i_hi = (start_block + this_blocks) * MR;
        if (i_hi > M) i_hi = M;
        start_block += this_blocks;
        if (i_lo >= i_hi) continue;
        if (t + 1 == n_threads) {
            // last slab inline saves one spawn+join
            matmul_row_slab(A, B, C, i_lo, i_hi, N, K, use_avx2);
        } else {
            workers.emplace_back(matmul_row_slab,
                                 A, B, C, i_lo, i_hi, N, K, use_avx2);
        }
    }
    for (auto& w : workers) w.join();
}

}  // namespace ssns::linalg
