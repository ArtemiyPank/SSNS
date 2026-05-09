// backend impl see header
#include <ssns/ckks/backend.hpp>

#include <random>

namespace ssns::ckks {

namespace {

// build array of ntt instances one per prime
std::array<NTT, NUM_PRIMES> make_ntts() {
    return std::array<NTT, NUM_PRIMES>{{
        NTT(COEFF_MODULI[0], POLY_DEGREE),
        NTT(COEFF_MODULI[1], POLY_DEGREE),
        NTT(COEFF_MODULI[2], POLY_DEGREE),
        NTT(COEFF_MODULI[3], POLY_DEGREE),
    }};
}

}  // namespace

// build full backend from one seed same seed gives same keys
Backend Backend::create(std::uint64_t seed) {
    // members init in declaration order so rng draws are stable across compilers
    // sk pk evk drawn from one stream in fixed order
    std::mt19937_64 rng(seed);

    Backend b{
        make_ntts(),
        Encoder{},
        SecretKey::sample(rng),
        PublicKey{},   // filled below to keep draw order
        EvalKey{},     // filled below
        static_cast<double>(1ULL << SCALE_BITS),
        Polynomial{},  // s_ntt filled below
    };
    b.pk  = gen_public_key(b.sk, b.ntts, rng);
    b.evk = gen_eval_key(b.sk, b.ntts, rng);
    // pre ntt sk so decrypt hot path skips 4 forward ntts per call
    b.s_ntt = b.sk.s;
    for (std::size_t i = 0; i < NUM_PRIMES; ++i) {
        b.ntts[i].forward(b.s_ntt.residues[i].data());
    }
    return b;
}

}  // namespace ssns::ckks
