#include <ssns/protocol/training.hpp>

#include <complex>
#include <random>
#include <utility>
#include <vector>

#include <ssns/ckks/ciphertext.hpp>
#include <ssns/ckks/decrypt.hpp>
#include <ssns/ckks/encrypt.hpp>
#include <ssns/ckks/linear_ops.hpp>
#include <ssns/ckks/plaintext.hpp>
#include <ssns/ckks/poly.hpp>

namespace ssns::protocol {

namespace {
// draw N(0, 1) batch of shape [batch, cols]
// fresh inputs every epoch
linalg::Matrix sample_normal_batch(std::size_t batch, std::size_t cols,
                                   nn::Rng& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    auto& g = rng.engine();
    linalg::Matrix X(batch, cols);
    double* d = X.data();
    const std::size_t n = X.size();
    for (std::size_t i = 0; i < n; ++i) d[i] = dist(g);
    return X;
}

// elementwise mse only used for logging
// gradients flow through compute_gradients not this scalar
double mse_loss(const linalg::Matrix& A, const linalg::Matrix& B) {
    double acc = 0.0;
    const double* a = A.data();
    const double* b = B.data();
    const std::size_t n = A.size();
    for (std::size_t i = 0; i < n; ++i) {
        const double d = a[i] - b[i];
        acc += d * d;
    }
    return acc / static_cast<double>(n);
}
}  // namespace

// три варианта step:
//   clean_train_step       - чистый baseline для tests и grid search
//   clean_train_step_noisy - быстрый прокси для FHE
//   clean_train_step_fhe   - полный CKKS pipeline ~100x медленнее

// one plain step sample X teacher target student forward FA grad client update
StepResult clean_train_step(
    nn::CleanClient& client,
    const nn::CleanServer& server,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng,
    double bimodality_alpha)
{
    // X гауссова свежая каждый шаг student учится функции а не точкам
    linalg::Matrix X = sample_normal_batch(batch_size, input_dim, rng);
    // Y_true живёт только на сервере teacher веса никогда не покидают сервер
    linalg::Matrix Y_true = server.teacher_forward(X);

    // H и Y_pred живут у клиента forward кэширует X и H_pre для последующего update
    auto fwd = client.forward(X);
    auto& H = fwd.H;
    auto& Y_pred = fwd.Y_pred;

    // порядок: forward -> compute_gradients -> update иначе update упадёт без кэша
    auto grads = server.compute_gradients(H, Y_pred, Y_true, bimodality_alpha);
    auto& grad_W2 = grads.grad_W2;
    auto& error_hidden = grads.error_hidden;

    // loss до update чтобы видеть state перед шагом
    const double loss = mse_loss(Y_pred, Y_true);

    client.update(grad_W2, error_hidden);

    return StepResult{
        std::move(X),
        std::move(Y_true),
        std::move(H),
        std::move(Y_pred),
        std::move(grad_W2),
        std::move(error_hidden),
        loss,
    };
}

// plain step plus gaussian noise on grads to fake FHE error
StepResult clean_train_step_noisy(
    nn::CleanClient& client,
    const nn::CleanServer& server,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng,
    double bimodality_alpha,
    double noise_std)
{
    linalg::Matrix X = sample_normal_batch(batch_size, input_dim, rng);
    linalg::Matrix Y_true = server.teacher_forward(X);

    auto fwd = client.forward(X);
    auto& H = fwd.H;
    auto& Y_pred = fwd.Y_pred;

    auto grads = server.compute_gradients(H, Y_pred, Y_true, bimodality_alpha);
    auto& grad_W2 = grads.grad_W2;
    auto& error_hidden = grads.error_hidden;

    // add noise to grads to imitate CKKS error
    // noise_std=0 means same as plain
    if (noise_std > 0.0) {
        std::normal_distribution<double> dist(0.0, noise_std);
        auto& g = rng.engine();
        double* g2 = grad_W2.data();
        for (std::size_t i = 0; i < grad_W2.size(); ++i) g2[i] += dist(g);
        double* eh = error_hidden.data();
        for (std::size_t i = 0; i < error_hidden.size(); ++i) eh[i] += dist(g);
    }

    const double loss = mse_loss(Y_pred, Y_true);

    client.update(grad_W2, error_hidden);

    return StepResult{
        std::move(X),
        std::move(Y_true),
        std::move(H),
        std::move(Y_pred),
        std::move(grad_W2),
        std::move(error_hidden),
        loss,
    };
}

// FHE training step three phases
// each scalar of H and Y_pred is its own ciphertext value broadcast to all slots
// trades packing for simpler matrix arithmetic
//
// server side pipeline all on ciphertexts:
//   1 error = (Y_pred - Y_true) / batch then rescale
//   2 H_l3 dummy mul and rescale to drop level
//   3 grad_W2 = sum H * error then rescale
//   4 error_hidden = sum error * B_FA cipher x plain
namespace {

using ssns::ckks::Ciphertext;
using ssns::ckks::Encoder;
using ssns::ckks::NTT;
using ssns::ckks::NUM_PRIMES;
using ssns::ckks::Plaintext;
using ssns::ckks::Polynomial;
using ssns::ckks::POLY_DEGREE;
using ssns::ckks::PublicKey;
using ssns::ckks::SecretKey;
using ssns::ckks::add;
using ssns::ckks::decrypt;
using ssns::ckks::encrypt;
using ssns::ckks::mul_cipher;
using ssns::ckks::mul_plain;
using ssns::ckks::mul_scalar;
using ssns::ckks::rescale;
using ssns::ckks::sub_plain;

constexpr std::size_t SLOT_COUNT = POLY_DEGREE / 2;

// encode one scalar broadcast across slots
Plaintext encode_scalar(const Encoder& encoder,
                        const std::array<NTT, NUM_PRIMES>& ntts,
                        double value, double scale, std::size_t level) {
    std::vector<std::complex<double>> slots(SLOT_COUNT, std::complex<double>(value, 0.0));
    Polynomial coeff = encoder.encode(slots, scale);
    return Plaintext::from_polynomial(std::move(coeff), scale, ntts, level);
}

// encrypt one scalar broadcast to all slots
// uses caller rng so output is deterministic
Ciphertext encrypt_scalar(const Encoder& encoder,
                          const std::array<NTT, NUM_PRIMES>& ntts,
                          const PublicKey& pk,
                          double value, double scale,
                          std::mt19937_64& rng) {
    Plaintext pt = encode_scalar(encoder, ntts, value, scale, NUM_PRIMES);
    return encrypt(pt, pk, ntts, rng);
}

// decrypt broadcast scalar return slot 0 real part
// uses precomputed NTT sk to skip 4 forward NTTs per call
// hot path 153k calls per preset config
double decrypt_scalar(const Encoder& encoder,
                      const std::array<NTT, NUM_PRIMES>& ntts,
                      const Polynomial& s_ntt,
                      const Ciphertext& ct) {
    Plaintext pt = decrypt_with_ntt_sk(ct, s_ntt);
    Polynomial coeff = pt.poly;
    for (std::size_t i = 0; i < pt.level; ++i) {
        ntts[i].inverse(coeff.residues[i].data());
    }
    auto slots = encoder.decode(coeff, pt.scale, pt.level);
    return slots[0].real();
}

}  // namespace

// build client view over backend includes SecretKey
ClientKeys make_client_keys(const ckks::Backend& backend) {
    return ClientKeys{
        /*sk=*/      &backend.sk,
        /*pk=*/      &backend.pk,
        /*ntts=*/    &backend.ntts,
        /*encoder=*/ &backend.encoder,
        /*s_ntt=*/   &backend.s_ntt,
        /*scale=*/   backend.scale,
    };
}

// build server view over backend no SecretKey
ServerKeys make_server_keys(const ckks::Backend& backend) {
    // SecretKey left out on purpose server can only consume ciphertexts
    return ServerKeys{
        /*pk=*/      &backend.pk,
        /*evk=*/     &backend.evk,
        /*ntts=*/    &backend.ntts,
        /*encoder=*/ &backend.encoder,
        /*scale=*/   backend.scale,
    };
}

// phase 1 client side encryption

// encrypt H and Y_pred per scalar bundle with plain X
ClientPayload client_encrypt(
    const linalg::Matrix& X,
    const linalg::Matrix& H,
    const linalg::Matrix& Y_pred,
    const ClientKeys& keys,
    std::mt19937_64& fhe_rng)
{
    const std::size_t batch_size = H.rows();
    const std::size_t H_cols = H.cols();
    const std::size_t Y_cols = Y_pred.cols();

    // derive seeds up front so encryption noise is deterministic regardless of thread order
    const std::size_t H_count = batch_size * H_cols;
    const std::size_t Y_count = batch_size * Y_cols;
    std::vector<std::uint64_t> H_seeds(H_count);
    std::vector<std::uint64_t> Y_seeds(Y_count);
    for (auto& s : H_seeds) s = fhe_rng();
    for (auto& s : Y_seeds) s = fhe_rng();

    std::vector<Ciphertext> H_ct(H_count);
    std::vector<Ciphertext> Y_pred_ct(Y_count);

    #pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < H_count; ++k) {
        const std::size_t i = k / H_cols;
        const std::size_t h = k % H_cols;
        std::mt19937_64 rk(H_seeds[k]);
        H_ct[k] = encrypt_scalar(*keys.encoder, *keys.ntts, *keys.pk,
                                  H(i, h), keys.scale, rk);
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < Y_count; ++k) {
        const std::size_t i = k / Y_cols;
        const std::size_t o = k % Y_cols;
        std::mt19937_64 rk(Y_seeds[k]);
        Y_pred_ct[k] = encrypt_scalar(*keys.encoder, *keys.ntts, *keys.pk,
                                       Y_pred(i, o), keys.scale, rk);
    }

    return ClientPayload{
        /*X=*/         X,
        /*H_ct=*/      std::move(H_ct),
        /*Y_pred_ct=*/ std::move(Y_pred_ct),
        /*H_cols=*/    H_cols,
        /*Y_cols=*/    Y_cols,
    };
}

// phase 2 server homomorphic compute no SecretKey

// FA grads on ciphertexts only
// server never sees plain H or Y_pred
ServerResponse server_compute_gradients(
    const ClientPayload& payload,
    const nn::CleanServer& server,
    const ServerKeys& keys,
    double bimodality_alpha)
{
    const std::size_t batch_size = payload.X.rows();
    const std::size_t H_cols = payload.H_cols;
    const std::size_t Y_cols = payload.Y_cols;

    // server computes Y_true locally teacher weights stay on server
    linalg::Matrix Y_true = server.teacher_forward(payload.X);
    const linalg::Matrix& B_FA = server.b_fa();

    // error = (Y_pred - Y_true) * (1/batch)
    // деление на batch заменено на mul_scalar в FHE деление невозможно
    const double inv_batch = 1.0 / static_cast<double>(batch_size);
    const std::size_t E_count = batch_size * Y_cols;
    std::vector<Ciphertext> error_ct(E_count);
    #pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < E_count; ++k) {
        const std::size_t i = k / Y_cols;
        const std::size_t o = k % Y_cols;
        double yt_val = Y_true(i, o);
        if (bimodality_alpha != 0.0) {
            // bimodality push толкаем target дальше от 0.5
            // conf in [0, 1] is how far sigmoid is from 0.5
            const double sig = 1.0 / (1.0 + std::exp(-yt_val));
            const double conf = 2.0 * std::abs(sig - 0.5);
            yt_val *= (1.0 + bimodality_alpha * conf);
        }
        Plaintext yt_pt = encode_scalar(*keys.encoder, *keys.ntts,
                                         yt_val, keys.scale, NUM_PRIMES);
        // sub_plain бесплатен по level mul_scalar потребляет один level
        Ciphertext diff = sub_plain(payload.Y_pred_ct[k], yt_pt);
        Ciphertext scaled = mul_scalar(diff, inv_batch);
        // rescale обязателен после mul иначе scale^2 переполнит modulus chain
        error_ct[k] = rescale(scaled, *keys.ntts);
    }

    // bring H to level 3 to match error chain
    // фиктивное mul_scalar(*,1.0) только чтобы согласовать уровни перед mul_cipher
    const std::size_t H_count = payload.H_ct.size();
    std::vector<Ciphertext> H_l3(H_count);
    #pragma omp parallel for schedule(static)
    for (std::size_t k = 0; k < H_count; ++k) {
        Ciphertext bumped = mul_scalar(payload.H_ct[k], 1.0);
        H_l3[k] = rescale(bumped, *keys.ntts);
    }

    // grad_W2[h,o] = sum_i H[i,h] * error[i,o] cipher x cipher
    // H_cols * Y_cols independent reductions parallel friendly
    // mul_cipher самая дорогая операция съедает level и нужен evk для relinearize
    std::vector<Ciphertext> grad_W2_ct(H_cols * Y_cols);
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t h = 0; h < H_cols; ++h) {
        for (std::size_t o = 0; o < Y_cols; ++o) {
            Ciphertext acc = mul_cipher(H_l3[0 * H_cols + h],
                                         error_ct[0 * Y_cols + o],
                                         *keys.evk, *keys.ntts);
            for (std::size_t i = 1; i < batch_size; ++i) {
                Ciphertext term = mul_cipher(H_l3[i * H_cols + h],
                                              error_ct[i * Y_cols + o],
                                              *keys.evk, *keys.ntts);
                // add не съедает level аккумулируем все term-ы потом один rescale
                acc = add(acc, term);
            }
            grad_W2_ct[h * Y_cols + o] = rescale(acc, *keys.ntts);
        }
    }

    // error_hidden[i,h] = sum_o error[i,o] * B_FA[o,h] cipher x plain
    // mul_plain дешевле mul_cipher не нужен evk и съедает меньше шумового бюджета
    std::vector<Plaintext> bfa_pt;
    bfa_pt.reserve(Y_cols * H_cols);
    for (std::size_t o = 0; o < Y_cols; ++o) {
        for (std::size_t h = 0; h < H_cols; ++h) {
            bfa_pt.push_back(encode_scalar(*keys.encoder, *keys.ntts,
                                           B_FA(o, h), keys.scale, /*level=*/3));
        }
    }
    std::vector<Ciphertext> error_hidden_ct(batch_size * H_cols);
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t i = 0; i < batch_size; ++i) {
        for (std::size_t h = 0; h < H_cols; ++h) {
            Ciphertext acc = mul_plain(error_ct[i * Y_cols + 0],
                                        bfa_pt[0 * H_cols + h]);
            for (std::size_t o = 1; o < Y_cols; ++o) {
                Ciphertext term = mul_plain(error_ct[i * Y_cols + o],
                                             bfa_pt[o * H_cols + h]);
                acc = add(acc, term);
            }
            error_hidden_ct[i * H_cols + h] = std::move(acc);
        }
    }

    return ServerResponse{
        /*grad_W2_ct=*/      std::move(grad_W2_ct),
        /*error_hidden_ct=*/ std::move(error_hidden_ct),
        /*H_cols=*/          H_cols,
        /*Y_cols=*/          Y_cols,
        /*batch_size=*/      batch_size,
    };
}

// phase 3 client side decryption

// decrypt server response back to plain matrices
DecryptedGradients client_decrypt(
    const ServerResponse& response,
    const ClientKeys& keys)
{
    DecryptedGradients out{
        linalg::Matrix(response.H_cols, response.Y_cols),
        linalg::Matrix(response.batch_size, response.H_cols),
    };
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t h = 0; h < response.H_cols; ++h) {
        for (std::size_t o = 0; o < response.Y_cols; ++o) {
            out.grad_W2(h, o) = decrypt_scalar(*keys.encoder, *keys.ntts, *keys.s_ntt,
                                                response.grad_W2_ct[h * response.Y_cols + o]);
        }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t i = 0; i < response.batch_size; ++i) {
        for (std::size_t h = 0; h < response.H_cols; ++h) {
            out.error_hidden(i, h) = decrypt_scalar(*keys.encoder, *keys.ntts, *keys.s_ntt,
                                                     response.error_hidden_ct[i * response.H_cols + h]);
        }
    }
    return out;
}

// one full FHE step three phases plus update
StepResult clean_train_step_fhe(
    nn::CleanClient& client,
    const nn::CleanServer& server,
    ckks::Backend& backend,
    std::size_t batch_size,
    std::size_t input_dim,
    nn::Rng& rng,
    double bimodality_alpha)
{
    // sample X and run client forward all plain client side
    // forward всегда plain клиент держит W1 W2 у себя FHE начинается только на wire
    linalg::Matrix X = sample_normal_batch(batch_size, input_dim, rng);
    auto fwd = client.forward(X);
    linalg::Matrix& H = fwd.H;
    linalg::Matrix& Y_pred = fwd.Y_pred;

    // compute Y_true here just for the returned MSE
    // in real deployment this happens on server
    linalg::Matrix Y_true = server.teacher_forward(X);
    const double loss = mse_loss(Y_pred, Y_true);

    // три фазы FHE encrypt -> compute -> decrypt SecretKey есть только у клиента
    // phase 1 client encrypts
    // зашифровано: H Y_pred plain: X (свежий гаусс не секрет)
    std::mt19937_64 fhe_rng(rng.engine()());
    auto client_keys = make_client_keys(backend);
    ClientPayload payload = client_encrypt(X, H, Y_pred, client_keys, fhe_rng);

    // phase 2 server compute no SecretKey by construction
    // сервер видит только ciphertexts град FA полностью на encrypted данных
    auto server_keys = make_server_keys(backend);
    ServerResponse response = server_compute_gradients(payload, server, server_keys, bimodality_alpha);

    // phase 3 client decrypts and updates
    // расшифровка только у клиента update идёт на plain градиентах
    DecryptedGradients grads = client_decrypt(response, client_keys);
    client.update(grads.grad_W2, grads.error_hidden);

    return StepResult{
        std::move(X),
        std::move(Y_true),
        std::move(H),
        std::move(Y_pred),
        std::move(grads.grad_W2),
        std::move(grads.error_hidden),
        loss,
    };
}

}  // namespace ssns::protocol
