// In-tree double-precision matmul kernel.  Used when the project is built
// with -DSSNS_USE_BLAS=OFF.  Phase 8 progressively replaces the OpenBLAS
// cblas_dgemm dependency with this hand-written code.
//
// Commit 1 (this file): minimal triple-loop placeholder.  Commits 2-4 expand
// to a register-blocked AVX2 microkernel with cache tiling and std::thread
// parallelism.
#include <cstddef>

namespace ssns::linalg {

void matmul_native(const double* A, const double* B, double* C,
                   std::size_t M, std::size_t N, std::size_t K) {
    // Row-major triple loop.  i-k-j ordering keeps B's row contiguous in the
    // innermost stride, letting the compiler auto-vectorise the j sweep.
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const double a_ik = A[i * K + k];
            const double* b_row = B + k * N;
            double* c_row = C + i * N;
            for (std::size_t j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

}  // namespace ssns::linalg
