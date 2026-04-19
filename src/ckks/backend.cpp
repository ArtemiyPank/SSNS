// Backend — implementation.  See header for rationale.
#include <ssns/ckks/backend.hpp>

#include <random>

namespace ssns::ckks {

namespace {

std::array<NTT, NUM_PRIMES> make_ntts() {
    return std::array<NTT, NUM_PRIMES>{{
        NTT(COEFF_MODULI[0], POLY_DEGREE),
        NTT(COEFF_MODULI[1], POLY_DEGREE),
        NTT(COEFF_MODULI[2], POLY_DEGREE),
        NTT(COEFF_MODULI[3], POLY_DEGREE),
    }};
}

}  // namespace

Backend Backend::create(std::uint64_t seed) {
    // We must initialise members in declaration order (ntts, encoder, sk,
    // pk, evk, scale) so RNG draws are deterministic across compilers.
    // Constructing a fresh PRNG here and consuming it sequentially gives
    // us bit-stable output for a given seed.
    std::mt19937_64 rng(seed);

    Backend b{
        make_ntts(),
        Encoder{},
        SecretKey::sample(rng),
        PublicKey{},   // filled in below to enforce strict draw order
        EvalKey{},     // filled in below
        static_cast<double>(1ULL << SCALE_BITS),
        Polynomial{},  // s_ntt, filled below
    };
    b.pk  = gen_public_key(b.sk, b.ntts, rng);
    b.evk = gen_eval_key(b.sk, b.ntts, rng);
    // Pre-NTT sk so the decrypt hot path doesn't redo 4 forward NTTs per call.
    b.s_ntt = b.sk.s;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        b.ntts[i].forward(b.s_ntt.residues[i].data());
    }
    return b;
}

}  // namespace ssns::ckks
