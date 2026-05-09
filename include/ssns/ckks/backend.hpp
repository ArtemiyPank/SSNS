// ckks backend bundle
// packs ntts encoder sk pk evk scale into one struct
// so callers pass one thing not five
//
// determinism: same seed gives same keys
// uses one mt19937_64 in fixed order sk pk evk
//
// ntt cache shared across all ops
// encoder reused everywhere
#pragma once

#include <ssns/ckks/encoder.hpp>
#include <ssns/ckks/eval_key.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/public_key.hpp>
#include <ssns/ckks/secret_key.hpp>

#include <array>
#include <cstdint>

namespace ssns::ckks {

struct Backend {
    std::array<NTT, NUM_PRIMES> ntts;
    Encoder    encoder;
    SecretKey  sk;
    PublicKey  pk;
    EvalKey    evk;
    double     scale;

    // ntt form of sk.s cached at create time
    // decrypt hot path uses this so we skip 4 forward ntts per call
    Polynomial s_ntt;

    // build deterministically from seed
    static Backend create(std::uint64_t seed);
};

}  // namespace ssns::ckks
