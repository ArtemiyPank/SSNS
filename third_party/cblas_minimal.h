/* third_party/cblas_minimal.h
 *
 * Minimal CBLAS prototypes used by the SSNS C++ port.  The Fedora system
 * lacks an openblas-devel package by default, but libopenblaso.so is present
 * with the standard CBLAS ABI.  Vendoring this tiny header avoids the system
 * dependency.  Only declarations the project actually calls are listed.
 *
 * Reference: Netlib CBLAS (http://www.netlib.org/blas/cblas.h)
 */
#ifndef SSNS_CBLAS_MINIMAL_H
#define SSNS_CBLAS_MINIMAL_H

#ifdef __cplusplus
extern "C" {
#endif

enum CBLAS_ORDER     { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans  = 111, CblasTrans    = 112, CblasConjTrans = 113 };

/* C := alpha * op(A) * op(B) + beta * C   (double precision general matmul) */
void cblas_dgemm(
    enum CBLAS_ORDER     Order,
    enum CBLAS_TRANSPOSE TransA,
    enum CBLAS_TRANSPOSE TransB,
    int M, int N, int K,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double beta,
    double *C, int ldc);

/* y := alpha * x + y */
void cblas_daxpy(int N, double alpha, const double *X, int incX,
                 double *Y, int incY);

/* y := alpha * op(A) * x + beta * y */
void cblas_dgemv(
    enum CBLAS_ORDER     Order,
    enum CBLAS_TRANSPOSE TransA,
    int M, int N,
    double alpha,
    const double *A, int lda,
    const double *X, int incX,
    double beta,
    double *Y, int incY);

#ifdef __cplusplus
}
#endif

#endif /* SSNS_CBLAS_MINIMAL_H */
