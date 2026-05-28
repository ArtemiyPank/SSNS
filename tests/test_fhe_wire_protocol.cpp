// wire-protocol invariants for the FHE training step
//
// honesty checks for the three-phase wire protocol:
//   1. ServerKeys structurally cannot hold a SecretKey (compile-time SFINAE)
//   2. ClientPayload = X plain + cipher(H) + cipher(Y_pred); those decrypt back to originals
//   3. ServerResponse holds ciphertexts only; on decrypt match plain grads within CKKS budget
//   4. server_compute_gradients never touches a sk (verified via has_sk_member)
//   5. composed clean_train_step_fhe == manual three-phase bit-for-bit at same seeds (no side channel)
//
// tag: [protocol][training][fhe][wire]
#include <catch.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

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
using ssns::protocol::clean_train_step;
using ssns::protocol::clean_train_step_fhe;
using ssns::protocol::client_decrypt;
using ssns::protocol::client_encrypt;
using ssns::protocol::ClientKeys;
using ssns::protocol::ClientPayload;
using ssns::protocol::DecryptedGradients;
using ssns::protocol::make_client_keys;
using ssns::protocol::make_server_keys;
using ssns::protocol::server_compute_gradients;
using ssns::protocol::ServerKeys;
using ssns::protocol::ServerResponse;

namespace {

// compile-time check: ServerKeys cannot hold a SecretKey
// SFINAE detector: has_sk_member<T>::value true iff T has a public `sk` decltype can resolve
template <typename, typename = void>
struct has_sk_member : std::false_type {};

template <typename T>
struct has_sk_member<T, std::void_t<decltype(std::declval<T>().sk)>>
    : std::true_type {};

// detectors for the borrowed views we DO expect on ServerKeys
template <typename, typename = void>
struct has_pk_member : std::false_type {};
template <typename T>
struct has_pk_member<T, std::void_t<decltype(std::declval<T>().pk)>>
    : std::true_type {};

template <typename, typename = void>
struct has_evk_member : std::false_type {};
template <typename T>
struct has_evk_member<T, std::void_t<decltype(std::declval<T>().evk)>>
    : std::true_type {};

// tiny config helpers
CleanClientConfig wire_cfg() {
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

double l_inf_diff(const Matrix& a, const Matrix& b) {
    REQUIRE(a.rows() == b.rows());
    REQUIRE(a.cols() == b.cols());
    double m = 0.0;
    const std::size_t n = a.size();
    const double* da = a.data();
    const double* db = b.data();
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::abs(da[i] - db[i]);
        if (d > m) m = d;
    }
    return m;
}

// decrypt + decode helper; mirrors private decrypt_scalar in src/protocol/training.cpp
// re-implemented here so we don't touch prod code from tests
double dec_scalar(const Ciphertext& ct, const ClientKeys& keys) {
    Plaintext pt = ssns::ckks::decrypt(ct, *keys.sk, *keys.ntts);
    Polynomial coeff = pt.poly;
    for (std::size_t i = 0; i < pt.level; ++i) {
        (*keys.ntts)[i].inverse(coeff.residues[i].data());
    }
    auto slots = keys.encoder->decode(coeff, pt.scale, pt.level);
    return slots[0].real();
}

}  // namespace

TEST_CASE("ServerKeys: has no SecretKey member (structural)",
          "[protocol][training][fhe][wire]") {
    // compile-time invariants: ServerKeys MUST NOT expose .sk, ClientKeys MUST
    // these static_asserts reject any future commit that accidentally adds sk to ServerKeys
    static_assert(!has_sk_member<ServerKeys>::value,
                  "ServerKeys must not expose a SecretKey field");
    static_assert(has_sk_member<ClientKeys>::value,
                  "ClientKeys must expose the SecretKey for decryption");
    static_assert(has_pk_member<ServerKeys>::value,
                  "ServerKeys must carry the PublicKey view");
    static_assert(has_evk_member<ServerKeys>::value,
                  "ServerKeys must carry the EvalKey view");

    // pair the type-level check with a runtime build; ServerKeys from a real Backend compiles
    // accessing .sk on it would not compile (verified above)
    Backend backend = Backend::create(0xC0FFEEULL);
    ServerKeys sk_view = make_server_keys(backend);
    REQUIRE(sk_view.pk      != nullptr);
    REQUIRE(sk_view.evk     != nullptr);
    REQUIRE(sk_view.ntts    != nullptr);
    REQUIRE(sk_view.encoder != nullptr);
    REQUIRE(sk_view.scale   == backend.scale);
}

TEST_CASE("ClientPayload: X plaintext + H/Y_pred ciphertexts decrypt to originals",
          "[protocol][training][fhe][wire]") {
    auto cfg = wire_cfg();
    CleanClient client(cfg);
    CleanServer server(2, 4, 2, 4, 42, 43);
    Backend backend = Backend::create(0xCAFEULL);
    ClientKeys client_keys = make_client_keys(backend);

    // sample fresh X, run student forward (same path as FHE step)
    Rng rng(2025);
    std::normal_distribution<double> dist(0.0, 1.0);
    const std::size_t batch_size = 4;
    const std::size_t input_dim  = 2;
    Matrix X(batch_size, input_dim);
    for (std::size_t i = 0; i < X.size(); ++i) X.data()[i] = dist(rng.engine());
    auto fwd = client.forward(X);

    std::mt19937_64 fhe_rng(0xBEEFULL);
    ClientPayload payload =
        client_encrypt(X, fwd.H, fwd.Y_pred, client_keys, fhe_rng);

    // (a) X forwarded as plaintext Matrix, shape preserved
    REQUIRE(payload.X.rows() == batch_size);
    REQUIRE(payload.X.cols() == input_dim);
    for (std::size_t i = 0; i < X.size(); ++i) {
        REQUIRE(payload.X.data()[i] == X.data()[i]);
    }

    // (b) ciphertext counts match (batch * cols) per matrix
    REQUIRE(payload.H_cols       == fwd.H.cols());
    REQUIRE(payload.Y_cols       == fwd.Y_pred.cols());
    REQUIRE(payload.H_ct.size()      == batch_size * payload.H_cols);
    REQUIRE(payload.Y_pred_ct.size() == batch_size * payload.Y_cols);

    // (c) fresh-encryption invariants on every ciphertext
    for (const auto& ct : payload.H_ct) {
        REQUIRE(ct.scale > 0.0);
        REQUIRE(ct.level == NUM_PRIMES);
    }
    for (const auto& ct : payload.Y_pred_ct) {
        REQUIRE(ct.scale > 0.0);
        REQUIRE(ct.level == NUM_PRIMES);
    }

    // (d) decrypt first 3 entries of each; round-trip must match original within CKKS noise floor
    //     1e-3 absolute matches encoder-test tolerance
    for (std::size_t k = 0; k < 3 && k < payload.H_ct.size(); ++k) {
        const std::size_t i = k / payload.H_cols;
        const std::size_t h = k % payload.H_cols;
        const double recovered = dec_scalar(payload.H_ct[k], client_keys);
        const double expected  = fwd.H(i, h);
        INFO("H_ct[" << k << "] decrypts to " << recovered
             << ", expected " << expected);
        REQUIRE(std::abs(recovered - expected) < 1e-3);
    }
    for (std::size_t k = 0; k < 3 && k < payload.Y_pred_ct.size(); ++k) {
        const std::size_t i = k / payload.Y_cols;
        const std::size_t o = k % payload.Y_cols;
        const double recovered = dec_scalar(payload.Y_pred_ct[k], client_keys);
        const double expected  = fwd.Y_pred(i, o);
        INFO("Y_pred_ct[" << k << "] decrypts to " << recovered
             << ", expected " << expected);
        REQUIRE(std::abs(recovered - expected) < 1e-3);
    }
}

TEST_CASE("ServerResponse: only ciphertexts, decrypt matches plaintext gradients",
          "[protocol][training][fhe][wire]") {
    auto cfg = wire_cfg();

    // plaintext reference
    CleanClient client_plain(cfg);
    CleanServer server_plain(2, 4, 2, 4, 42, 43);
    Rng rng_plain(2025);
    auto plain = clean_train_step(client_plain, server_plain,
                                  /*batch_size=*/8, /*input_dim=*/2, rng_plain);

    // manual three-phase FHE
    CleanClient client_fhe(cfg);
    CleanServer server_fhe(2, 4, 2, 4, 42, 43);
    Backend backend = Backend::create(0xC0FFEEULL);
    Rng rng_fhe(2025);

    // mirror the seed sequence inside clean_train_step_fhe so X / forward-pass match plain path exactly
    std::normal_distribution<double> dist(0.0, 1.0);
    const std::size_t batch_size = 8;
    const std::size_t input_dim  = 2;
    Matrix X(batch_size, input_dim);
    for (std::size_t i = 0; i < X.size(); ++i)
        X.data()[i] = dist(rng_fhe.engine());
    auto fwd = client_fhe.forward(X);

    std::mt19937_64 fhe_rng(rng_fhe.engine()());
    ClientKeys client_keys = make_client_keys(backend);
    ServerKeys server_keys = make_server_keys(backend);

    ClientPayload payload =
        client_encrypt(X, fwd.H, fwd.Y_pred, client_keys, fhe_rng);
    ServerResponse response =
        server_compute_gradients(payload, server_fhe, server_keys);

    // (a) response shapes match the protocol header
    REQUIRE(response.H_cols     == fwd.H.cols());
    REQUIRE(response.Y_cols     == fwd.Y_pred.cols());
    REQUIRE(response.batch_size == batch_size);
    REQUIRE(response.grad_W2_ct.size()
            == response.H_cols * response.Y_cols);
    REQUIRE(response.error_hidden_ct.size()
            == response.batch_size * response.H_cols);

    // (b) grad_W2 cts rescaled (level 2: mul_cipher at 3 then rescale)
    //     error_hidden at level 3 (mul_plain doesn't drop level; error was already at 3)
    for (const auto& ct : response.grad_W2_ct) {
        REQUIRE(ct.level >= 1);
        REQUIRE(ct.level < NUM_PRIMES);
    }
    for (const auto& ct : response.error_hidden_ct) {
        REQUIRE(ct.level >= 1);
        REQUIRE(ct.level < NUM_PRIMES);
    }

    // (c) decrypted gradients match plaintext path within 5e-2 L_inf
    DecryptedGradients grads = client_decrypt(response, client_keys);
    REQUIRE(grads.grad_W2.rows()      == plain.grad_W2.rows());
    REQUIRE(grads.grad_W2.cols()      == plain.grad_W2.cols());
    REQUIRE(grads.error_hidden.rows() == plain.error_hidden.rows());
    REQUIRE(grads.error_hidden.cols() == plain.error_hidden.cols());

    const double gw2_diff = l_inf_diff(grads.grad_W2,      plain.grad_W2);
    const double eh_diff  = l_inf_diff(grads.error_hidden, plain.error_hidden);
    INFO("grad_W2 L_inf diff = "      << gw2_diff);
    INFO("error_hidden L_inf diff = " << eh_diff);
    REQUIRE(gw2_diff < 5e-2);
    REQUIRE(eh_diff  < 5e-2);
}

TEST_CASE("server_compute_gradients does not touch SecretKey",
          "[protocol][training][fhe][wire]") {
    // structural invariant first: server fn takes ServerKeys, and ServerKeys::sk does not exist
    // even a malicious caller with a SecretKey in-process has no way to plumb it through this API
    static_assert(!has_sk_member<ServerKeys>::value,
                  "ServerKeys must not expose a SecretKey field");

    auto cfg = wire_cfg();
    CleanClient client(cfg);
    CleanServer server(2, 4, 2, 4, 42, 43);
    Backend backend = Backend::create(0xCAFEULL);

    // build the payload normally (client side)
    Rng rng(2025);
    std::normal_distribution<double> dist(0.0, 1.0);
    Matrix X(4, 2);
    for (std::size_t i = 0; i < X.size(); ++i) X.data()[i] = dist(rng.engine());
    auto fwd = client.forward(X);

    std::mt19937_64 fhe_rng(0xFEEDULL);
    ClientKeys ck = make_client_keys(backend);
    ClientPayload payload = client_encrypt(X, fwd.H, fwd.Y_pred, ck, fhe_rng);

    // server processes payload with ONLY a ServerKeys view; no path to backend.sk through this signature
    ServerKeys sk_view = make_server_keys(backend);
    ServerResponse response =
        server_compute_gradients(payload, server, sk_view);

    REQUIRE_FALSE(response.grad_W2_ct.empty());
    REQUIRE_FALSE(response.error_hidden_ct.empty());
    REQUIRE(response.batch_size == 4);
    REQUIRE(response.H_cols     == fwd.H.cols());
    REQUIRE(response.Y_cols     == fwd.Y_pred.cols());
}

TEST_CASE("composed clean_train_step_fhe matches manual three-phase exactly",
          "[protocol][training][fhe][wire]") {
    auto cfg = wire_cfg();

    // path A: composed clean_train_step_fhe
    CleanClient client_a(cfg);
    CleanServer server_a(2, 4, 2, 4, 42, 43);
    Backend backend_a = Backend::create(0xC0FFEEULL);
    Rng rng_a(2025);
    auto state_a = clean_train_step_fhe(client_a, server_a, backend_a,
                                        /*batch_size=*/8,
                                        /*input_dim=*/2, rng_a);

    // path B: manual three-phase + Adam update
    CleanClient client_b(cfg);
    CleanServer server_b(2, 4, 2, 4, 42, 43);
    Backend backend_b = Backend::create(0xC0FFEEULL);
    Rng rng_b(2025);

    std::normal_distribution<double> dist(0.0, 1.0);
    const std::size_t batch_size = 8;
    const std::size_t input_dim  = 2;
    Matrix X(batch_size, input_dim);
    for (std::size_t i = 0; i < X.size(); ++i)
        X.data()[i] = dist(rng_b.engine());
    auto fwd = client_b.forward(X);
    Matrix Y_true_local = server_b.teacher_forward(X);

    std::mt19937_64 fhe_rng(rng_b.engine()());
    ClientKeys ck = make_client_keys(backend_b);
    ServerKeys sk = make_server_keys(backend_b);

    ClientPayload payload =
        client_encrypt(X, fwd.H, fwd.Y_pred, ck, fhe_rng);
    ServerResponse response =
        server_compute_gradients(payload, server_b, sk);
    DecryptedGradients grads = client_decrypt(response, ck);
    client_b.update(grads.grad_W2, grads.error_hidden);

    // X / forward / Y_true / grads / weights match BIT-EXACTLY
    // seed schedule identical, CKKS noise deterministic; no info slips between paths
    REQUIRE(l_inf_diff(state_a.X,            X)              == 0.0);
    REQUIRE(l_inf_diff(state_a.Y_true,       Y_true_local)   == 0.0);
    REQUIRE(l_inf_diff(state_a.H,            fwd.H)          == 0.0);
    REQUIRE(l_inf_diff(state_a.Y_pred,       fwd.Y_pred)     == 0.0);
    REQUIRE(l_inf_diff(state_a.grad_W2,      grads.grad_W2)  == 0.0);
    REQUIRE(l_inf_diff(state_a.error_hidden, grads.error_hidden) == 0.0);
    REQUIRE(l_inf_diff(client_a.W1(),        client_b.W1())  == 0.0);
    REQUIRE(l_inf_diff(client_a.W2(),        client_b.W2())  == 0.0);
}
