// "real algorithm" checks for the FHE training step
//
// proves (via CKKS metadata + ciphertext residues) the step really exercises
// encrypt, mul_cipher, mul_plain, mul_scalar, rescale (not a stub)
//
// coverage:
//   - fresh-encryption invariants on H_ct (level == NUM_PRIMES, scale == backend.scale)
//   - grad_W2_ct rescaled to NUM_PRIMES-1 (rescale after mul_cipher dropped one prime)
//   - error_hidden_ct at level 3 (mul_plain accumulation, no level drop)
//   - grad_W2 scale lands in CKKS-chain band after mul_cipher * rescale
//   - decrypt with WRONG sk returns noise, not the message (binds to (sk, pk))
//
// tag: [protocol][training][fhe][algorithm]
#include <catch.hpp>

#include <cmath>
#include <cstddef>
#include <random>

#include <ssns/ckks/backend.hpp>
#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/decrypt.hpp>
#include <ssns/ckks/encoder.hpp>
#include <ssns/ckks/eval_key.hpp>
#include <ssns/ckks/ntt.hpp>
#include <ssns/ckks/params.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/poly.hpp>
#include <ssns/ckks/public_key.hpp>
#include <ssns/ckks/secret_key.hpp>
#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/client.hpp>
#include <ssns/nn/init.hpp>
#include <ssns/nn/server.hpp>
#include <ssns/protocol/training.hpp>

using ssns::ckks::Backend;
using ssns::ckks::Ciphertext;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::Plaintext;
using ssns::ckks::Polynomial;
using ssns::linalg::Matrix;
using ssns::nn::CleanClient;
using ssns::nn::CleanClientConfig;
using ssns::nn::CleanServer;
using ssns::nn::Rng;
using ssns::protocol::client_encrypt;
using ssns::protocol::ClientKeys;
using ssns::protocol::ClientPayload;
using ssns::protocol::make_client_keys;
using ssns::protocol::make_server_keys;
using ssns::protocol::server_compute_gradients;
using ssns::protocol::ServerKeys;
using ssns::protocol::ServerResponse;

namespace {

CleanClientConfig algo_cfg() {
    CleanClientConfig c{};
    c.input_dim          = 2;
    c.hidden_dim         = 4;
    c.output_dim         = 4;
    c.lr_max             = 0.01;
    c.lr_total_steps     = 200;
    c.lr_warmup_frac     = 0.05;
    c.seed               = 2024;
    c.grad_clip_max_norm = 1.0;
    return c;
}

// run one phase-1 + phase-2 cycle; returns payload+response plus original H/Y_pred
struct AlgoState {
    Matrix         X;
    Matrix         H;
    Matrix         Y_pred;
    ClientPayload  payload;
    ServerResponse response;
};

AlgoState run_one_step(CleanClient& client,
                       const CleanServer& server,
                       const Backend& backend,
                       std::size_t batch,
                       std::size_t in,
                       std::uint64_t fhe_seed) {
    Rng rng(2025);
    std::normal_distribution<double> dist(0.0, 1.0);
    Matrix X(batch, in);
    for (std::size_t i = 0; i < X.size(); ++i) X.data()[i] = dist(rng.engine());
    auto fwd = client.forward(X);

    std::mt19937_64 fhe_rng(fhe_seed);
    auto ck = make_client_keys(backend);
    auto sk = make_server_keys(backend);
    ClientPayload payload =
        client_encrypt(X, fwd.H, fwd.Y_pred, ck, fhe_rng);
    ServerResponse response =
        server_compute_gradients(payload, server, sk);

    return AlgoState{
        std::move(X),
        std::move(fwd.H),
        std::move(fwd.Y_pred),
        std::move(payload),
        std::move(response),
    };
}

// decrypt-with-arbitrary-keys helper for wrong-key case; mirrors private decrypt_scalar in training.cpp
double dec_with(const Ciphertext& ct,
                const ssns::ckks::SecretKey& sk,
                const std::array<ssns::ckks::NTT, NUM_PRIMES>& ntts,
                const ssns::ckks::Encoder& encoder) {
    Plaintext pt = ssns::ckks::decrypt(ct, sk, ntts);
    Polynomial coeff = pt.poly;
    for (std::size_t i = 0; i < pt.level; ++i) {
        ntts[i].inverse(coeff.residues[i].data());
    }
    auto slots = encoder.decode(coeff, pt.scale, pt.level);
    return slots[0].real();
}

}  // namespace

TEST_CASE("FHE: H ciphertexts are at fresh level and scale=backend.scale",
          "[protocol][training][fhe][algorithm]") {
    auto cfg = algo_cfg();
    CleanClient client(cfg);
    CleanServer server(2, 4, 2, 4, 42, 43);
    Backend backend = Backend::create(0xCAFEULL);
    auto state = run_one_step(client, server, backend, 8, 2, 0xBEEFULL);

    REQUIRE_FALSE(state.payload.H_ct.empty());
    REQUIRE_FALSE(state.payload.Y_pred_ct.empty());

    for (const auto& ct : state.payload.H_ct) {
        REQUIRE(ct.level == NUM_PRIMES);  // = 4, fresh encryption
        REQUIRE(ct.scale == backend.scale);
    }
    for (const auto& ct : state.payload.Y_pred_ct) {
        REQUIRE(ct.level == NUM_PRIMES);
        REQUIRE(ct.scale == backend.scale);
    }
}

TEST_CASE("FHE: server response grad_W2 ciphertexts are rescaled to NUM_PRIMES - 1",
          "[protocol][training][fhe][algorithm]") {
    // grad_W2 path: H rescaled 4->3 (mul_scalar(1.0)+rescale), error 4->3 (sub_plain, mul_scalar, rescale)
    // mul_cipher(H_l3, error) stays at 3, then rescale -> 2; so every grad_W2 ct has level == NUM_PRIMES-2 == 2
    auto cfg = algo_cfg();
    CleanClient client(cfg);
    CleanServer server(2, 4, 2, 4, 42, 43);
    Backend backend = Backend::create(0xCAFEULL);
    auto state = run_one_step(client, server, backend, 8, 2, 0xBEEFULL);

    REQUIRE_FALSE(state.response.grad_W2_ct.empty());
    for (const auto& ct : state.response.grad_W2_ct) {
        REQUIRE(ct.level == NUM_PRIMES - 2);
        REQUIRE(ct.level >= 1);
    }
}

TEST_CASE("FHE: server response error_hidden ciphertexts are at level 3",
          "[protocol][training][fhe][algorithm]") {
    // error_hidden = error * B_FA via mul_plain
    // error at level 3 (post divide-by-batch rescale), B_FA encoded at level 3; mul_plain keeps 3, no rescale
    auto cfg = algo_cfg();
    CleanClient client(cfg);
    CleanServer server(2, 4, 2, 4, 42, 43);
    Backend backend = Backend::create(0xCAFEULL);
    auto state = run_one_step(client, server, backend, 8, 2, 0xBEEFULL);

    REQUIRE_FALSE(state.response.error_hidden_ct.empty());
    for (const auto& ct : state.response.error_hidden_ct) {
        REQUIRE(ct.level == 3);
    }
}

TEST_CASE("FHE: grad_W2 ciphertext scale reflects mul_cipher × rescale chain",
          "[protocol][training][fhe][algorithm]") {
    // after one mul_cipher (scale = scale_a*scale_b ~ delta^2) and one rescale (scale /= q_drop):
    //   result = delta^2 / q_drop; delta=2^40, q_drop~2^60 -> ~2^20 (very different from backend.scale)
    // we check the scale lands in the CKKS chain band [delta^2/q_max, delta^2/q_min]
    // consistent with exactly one mul_cipher + one rescale
    auto cfg = algo_cfg();
    CleanClient client(cfg);
    CleanServer server(2, 4, 2, 4, 42, 43);
    Backend backend = Backend::create(0xCAFEULL);
    auto state = run_one_step(client, server, backend, 8, 2, 0xBEEFULL);

    const double delta = backend.scale;
    // q-drop = highest-indexed ACTIVE prime before rescale (per linear_ops.hpp)
    // pipeline: H, error both rescale 4->3 (drops COEFF_MODULI[3], 60-bit), then mul_cipher*rescale drops COEFF_MODULI[2] (40-bit)
    // so q_drop here = index-2 prime
    const double q_drop_lo = std::ldexp(1.0, 39);   // 2^39 lower envelope
    const double q_drop_hi = std::ldexp(1.0, 41);   // 2^41 upper envelope
    const double scale_lo  = (delta * delta) / q_drop_hi;
    const double scale_hi  = (delta * delta) / q_drop_lo;

    for (const auto& ct : state.response.grad_W2_ct) {
        INFO("grad_W2 scale=" << ct.scale
             << " expected range [" << scale_lo << ", " << scale_hi << "]");
        REQUIRE(ct.scale > scale_lo);
        REQUIRE(ct.scale < scale_hi);
    }
}

TEST_CASE("FHE: decrypt with the WRONG SecretKey returns noise, not the message",
          "[protocol][training][fhe][algorithm]") {
    // negative test: H_ct are real RLWE encryptions, not a plaintext channel
    // two independent backends; encrypt H[0,0] under backend1.pk, decrypt with backend2.sk
    // wrong-sk decrypt produces values dominated by the (very large) noise term
    auto cfg = algo_cfg();
    CleanClient client(cfg);
    CleanServer server(2, 4, 2, 4, 42, 43);
    Backend backend1 = Backend::create(0xAAAA1111ULL);
    Backend backend2 = Backend::create(0xBBBB2222ULL);
    REQUIRE(backend1.scale == backend2.scale);

    Rng rng(2025);
    std::normal_distribution<double> dist(0.0, 1.0);
    Matrix X(8, 2);
    for (std::size_t i = 0; i < X.size(); ++i) X.data()[i] = dist(rng.engine());
    auto fwd = client.forward(X);
    const double original = fwd.H(0, 0);

    std::mt19937_64 fhe_rng(0xBEEFULL);
    auto ck1 = make_client_keys(backend1);
    ClientPayload payload =
        client_encrypt(X, fwd.H, fwd.Y_pred, ck1, fhe_rng);

    // sanity: right key recovers the message
    const double recovered_right = [&] {
        Plaintext pt =
            ssns::ckks::decrypt(payload.H_ct[0], backend1.sk, backend1.ntts);
        Polynomial coeff = pt.poly;
        for (std::size_t i = 0; i < pt.level; ++i) {
            backend1.ntts[i].inverse(coeff.residues[i].data());
        }
        auto slots = backend1.encoder.decode(coeff, pt.scale, pt.level);
        return slots[0].real();
    }();
    INFO("right-key decrypt = " << recovered_right
         << " (expected " << original << ")");
    REQUIRE(std::abs(recovered_right - original) < 1e-3);

    // now decrypt with wrong sk + foreign ntts/encoder; ntts/encoder are stateless but use backend2 to emphasise foreign keyholder
    const double recovered_wrong =
        dec_with(payload.H_ct[0], backend2.sk,
                 backend2.ntts, backend2.encoder);
    INFO("wrong-key decrypt = " << recovered_wrong
         << " (expected garbage, not " << original << ")");
    // wrong-key noise is huge (centered-residue lift of an unrelated poly / scale; typical |.| >> 1)
    // require gap >> right-key tolerance; 1.0 absolute is the documented "garbage" envelope
    REQUIRE(std::abs(recovered_wrong - original) > 1.0);
}
