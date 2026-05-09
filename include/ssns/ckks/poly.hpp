// polynomial element of Z[X]/(X^N+1) in rns form
//
// a coefficient lives mod Q where Q = product of q_i
// rns repr stores one residue vector per prime q_i
// shape is NUM_PRIMES x N uint64
// ops run independently per residue and result is reduced mod Q via crt
//
// from_coeffs lifts a small int polynomial into rns by reducing each coef mod q_i
// for negative values add q_i first to map into [0, q_i)
//
// multiply does poly mul via ntt
// forward ntt both operands per prime pointwise mul mod q_i inverse ntt
// тут convolution theorem: pointwise product в ntt domain == polynomial product в coefficient domain
// модуло X^N+1 благодаря negacyclic ntt
//
// caller supplies pre built ntt instances one per prime so they are cached across many multiplications
#pragma once

#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace ssns::ckks {

class Polynomial {
public:
    // all zero polynomial
    Polynomial();

    // lift small int coefs into rns
    // coeffs.size() must be <= POLY_DEGREE missing high coefs are zero
    static Polynomial from_coeffs(const std::vector<std::int64_t>& coeffs);

    // in place add this += other per prime
    Polynomial& add_inplace(const Polynomial& other);
    // in place sub this -= other per prime
    Polynomial& sub_inplace(const Polynomial& other);

    // poly mul in Z_q[X]/(X^N+1)
    // forward ntt pointwise mul inverse ntt per prime
    // ntts[i].prime() must equal COEFF_MODULI[i] and degree must equal POLY_DEGREE
    //
    // negacyclic ntt уже учитывает X^N == -1 поэтому отдельная reduction по X^N+1 не нужна
    // обычная циклическая convolution дала бы X^N-1 что бесполезно для ckks
    static Polynomial multiply(
        const Polynomial& a,
        const Polynomial& b,
        const std::array<NTT, NUM_PRIMES>& ntts);

    // bitwise equality on every residue
    bool operator==(const Polynomial& other) const noexcept;

    // public storage one residue vector per prime each of length POLY_DEGREE
    // exposed because tests and ops need to read write residues directly
    std::array<std::vector<std::uint64_t>, NUM_PRIMES> residues;
};

}  // namespace ssns::ckks
