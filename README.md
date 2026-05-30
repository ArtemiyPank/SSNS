# SSNS: Secure Split-Neural Synchronization

A two-party key-exchange protocol where the **shared secret is a synchronized pair of neural networks**, established once over a homomorphically-encrypted channel, then reused to derive unlimited symmetric session keys at near-zero cost.

> A Teacher-Student key exchange with a homomorphically encrypted channel and SHA-256 verification.

---

## The Idea

Classical asymmetric cryptography (RSA, ECDH) pays a heavy per-message cost. Every session key you want to negotiate costs you another full asymmetric handshake. If a system needs thousands of session keys (short-lived tokens, per-message keys, rotating credentials), the cumulative cost is high.

**SSNS pays the asymmetric cost once.**

The expensive thing transmitted is not a session key, it is the **weights of a neural network**, established under FHE so the server never sees them in plaintext. Once Teacher and Student networks are synchronized, both parties hold the *same function*. From that point on, session keys are free:

1. **Server picks a fresh public seed `X_key`** (no encryption needed, since `X_key` itself is not secret).
2. **Server runs Teacher(`X_key`)** locally; **Client runs Student(`X_key`)** locally.
3. Both networks produce **the same bit-vector** (by construction, they were trained to agree).
4. SHA-256 of the extracted bits is exchanged; if it matches, the key is accepted.

No ciphertext per session. No handshake per session. The forward pass is a few matmuls, orders of magnitude cheaper than an RSA decryption.

The "private key" is the weights. The "key generator" is the network itself.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md): process topology, module map, training data flow, where-to-start-reading guide.
- [`docs/api.md`](docs/api.md): HTTP API reference (all endpoints, request/response shapes, CLI flags).
- [`docs/SSNS-presentation.pdf`](docs/SSNS-presentation.pdf): full mathematical write-up of the protocol and CKKS internals.
- [`LICENSE`](LICENSE): MIT.

---

## Protocol: Three Phases

### 1. Initialization
- **Teacher (Server)** samples random weights `W₁_T, W₂_T`, the target function the Student must learn.
- **Student (Client)** initializes its own weights `W₁_S, W₂_S`, plus a Feedback-Alignment matrix `B_FA` (used in place of the Student's `W₂_Sᵀ` during the backward pass, so the server never needs to know Student's weights).
- **Client generates CKKS keys**: sparse-ternary secret key `sk`, public key `pk`, evaluation key `evk`. Server receives only `pk` and `evk`, so it can compute on ciphertexts but cannot decrypt.

### 2. Synchronization (training, over FHE)
For each epoch:
1. A fresh input `X` is sampled; both parties see it in plaintext.
2. Server computes `Y_true = Teacher(X)` locally.
3. Client computes `H = ReLU(X·W₁_S)` and `Y_pred = H·W₂_S`, **encrypts both under CKKS**, sends to the server.
4. Server, working entirely on ciphertexts, computes:
   - `error = (Y_pred − Y_true) / batch_size`
   - `grad_W₂ = Hᵀ · error` *(cipher × cipher)*
   - `error_hidden = error · B_FA` *(cipher × plaintext, Feedback Alignment instead of true backprop, because the server doesn't know `W₂_S`)*
5. Server returns encrypted gradients; client decrypts, applies Adam + L2-norm-clipped weight updates.

After convergence: `Student(X) ≈ Teacher(X)` for any fresh `X`, *and neither party's weights ever crossed the wire in plaintext*.

### 3. Key Derivation
- Server picks `X_key` (public) and sends it.
- Both parties run their own network and apply sigmoid.
- The output layer is sized as `key_bits × cluster_size`. Cluster-mean + a dead-zone of `±dz` around `0.5` converts each cluster of neurons into one confident bit (or discards it as ambiguous). The dead-zone is what guarantees agreement under residual numerical noise.
- SHA-256 of the bit sequence is exchanged. Match ⇒ key accepted on both sides.

---

## Cryptographic Stack: Hand-Written CKKS

CKKS is implemented from scratch in this repo. No SEAL, OpenFHE, HElib, TenSEAL.

| Component | Implementation |
|---|---|
| Polynomial degree `N` | 4096 (power of 2, required by NTT) |
| RNS modulus chain | `[60, 40, 40, 60]` bits, ≈ 200 bits total |
| Scaling factor `Δ` | 2⁴⁰ |
| Secret key | Sparse ternary, Hamming weight ≈ 64 |
| Public key | BV-style `(b, a) = (−a·s + e, a)` |
| Evaluation key | RNS-gadget decomposed encryptions of `s²` |
| Encoding | IFFT + scaling + RNS lift |
| Ciphertext ops | Add, sub, mul-scalar, mul-plain, mul-cipher (tensor expansion + relinearization), rescale |
| Decryption | RNS decrypt + Garner's CRT reconstruction + FFT decode |

The full mathematical write-up of every step is in the project presentation: [`docs/SSNS-presentation.pdf`](docs/SSNS-presentation.pdf).

---

## Two Hard Rules

1. **All neural-net math is hand-written.** No PyTorch, no Eigen, no external BLAS. Matmul uses an in-tree register-blocked AVX2 microkernel with `std::thread` parallelism (`src/linalg/matmul_native.cpp`).
2. **CKKS is implemented from scratch.** Real polynomial-ring cryptography end-to-end.

These rules exist because the project is also a teaching artifact.

---

## Numerical Recipe (current champion)

| Knob | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| LR schedule | Warmup + cosine, peak `lr = 0.01` |
| Gradient clip | L2-norm ≤ 1.0 per matrix |
| Epochs | 30 000 |
| Batch size | 64 |
| Student hidden width | 768 |
| Teacher hidden width | 128 (asymmetric: Student is wider, absorbs FA noise) |
| Cluster size | 5 neurons per bit |
| Dead-zone half-width | 0.09 |
| Hidden activation | ReLU (required by FA, since sigmoid kills the FA gradient) |

Adam earns its keep here: the cryptographic + Feedback-Alignment noise is high-variance, and Adam's per-parameter adaptive step neutralises it.

---

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requirements: C++20 compiler, CMake ≥ 3.20, x86-64 with AVX2 + FMA. No external numerical libraries.

## Run the Visual IDE

```sh
./build/ssns-server --host 127.0.0.1 --port 8765 --ui-dir ui
```

Open <http://127.0.0.1:8765/> in a browser. The UI lets you configure a run, watch loss curves and weight previews live, perform the keygen exchange, and run stress tests (up to 200 000 trials).

---

## Layout

| Path | Role |
|---|---|
| `include/ssns/` | Public headers |
| `src/ckks/` | CKKS: NTT, RNS, encoder, keygen, encrypt/decrypt, linear ops, eval-key relinearization |
| `src/nn/` | Teacher, Client (Student), Adam, initializers, activations, LR schedule |
| `src/linalg/` | Hand-written matrix + AVX2 matmul microkernel |
| `src/protocol/` | Training loop and FHE-wired variant |
| `src/crypto/` | SHA-256 |
| `src/http/` | `cpp-httplib`-based server and REST handlers |
| `src/io/` | Status-file writer + logger |
| `src/server_main.cpp` | `ssns-server` entry point |
| `src/benchmark.cpp` | `ssns-benchmark` CLI subprocess spawned by `/api/run_training` |
| `tests/` | Catch2 unit + integration tests |
| `third_party/` | Vendored single-header deps: cpp-httplib, nlohmann/json, Catch2 v2 |
| `ui/` | Frontend SPA (HTML/CSS/JS) |
| `docs/` | Presentation, design notes, project book |

---

## Status / Scope

This is a **research / educational prototype**. The protocol's *idea*, synchronized neural networks as a session-key generator, is the contribution. The CKKS layer is implemented for correctness and pedagogical clarity, not constant-time hardened production crypto. Don't ship it without a security review.
