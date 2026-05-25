// ckks canonical embedding
// complex slot vector <-> Polynomial
//
// encodes N/2 complex numbers into Z[X]/(X^N + 1) by evaluating at primitive 2N-th roots
// poly must have real coefs so N evals come in N/2 conjugate pairs
// gives N/2 slots
//
// encode pipeline
//   1 mirror N/2 slots to length N with z_full[N-1-k] = conj(z_full[k])
//   2 inverse special fft (twist by zeta^-k then ifft then twist again)
//   3 multiply by scale round to int lift into rns
//
// decode is the matched inverse
//
// важно: twist by zeta^-k и conjugate mirror это две части одного special fft
// без twist получился бы стандартный dft не уважающий X^N+1
// без mirror коэффициенты были бы комплексными а нам нужны real
//
// fft is plain radix 2 in place using std::complex<double>
#pragma once

#include <ssns/ckks/poly.hpp>

#include <complex>
#include <cstddef>
#include <vector>

namespace ssns::ckks {
    class Encoder {
    public:
        // builds the encoder
        // pre computes fft twiddles and zeta^k twist tables
        Encoder();

        // encode slot vector of length POLY_DEGREE/2 into a polynomial
        // throws if z.size() != POLY_DEGREE/2
        Polynomial encode(const std::vector<std::complex<double> > &z, double scale) const;

        // decode polynomial back into slots
        // lifts each rns coef to signed int via garner crt divides by scale runs special fft
        //
        // level controls how many rns primes participate in the lift
        // defaults to NUM_PRIMES
        // after rescale active level shrinks pass that smaller level so dropped residues do not perturb the lift
        // 1 <= level <= NUM_PRIMES
        std::vector<std::complex<double> > decode(const Polynomial &p, double scale,
                                                  std::size_t level = NUM_PRIMES) const;

    private:
        // radix 2 fft in place
        // inverse=true divides by N at the end
        void fft(std::vector<std::complex<double> > &a, bool inverse) const;

        // bit reversal permutation used by both directions
        void bitreverse_permute(std::vector<std::complex<double> > &a) const;

        // pre computed twist factors
        // zeta_pow_[k] = zeta^k where zeta = exp(pi*i/N)
        // decode uses the conjugate
        std::vector<std::complex<double> > zeta_pow_;
        std::vector<std::complex<double> > zeta_pow_conj_;

        // pre computed fft twiddles
        std::vector<std::complex<double> > twiddle_fwd_;
        std::vector<std::complex<double> > twiddle_inv_;
    };
} // namespace ssns::ckks
