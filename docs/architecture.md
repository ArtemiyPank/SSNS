# SSNS Architecture

This is a navigation guide for the codebase. For the high-level protocol idea see [README.md](../README.md); for the mathematics see [SSNS-presentation.pdf](SSNS-presentation.pdf); for HTTP endpoints see [api.md](api.md).

---

## Process Topology

```
                                  ┌──────────────────────────────┐
                                  │  Web UI (ui/index.html + JS) │
                                  │  served from ui/ at GET /    │
                                  └──────────────┬───────────────┘
                                                 │ JSON over HTTP
                                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ssns-server  (long-lived HTTP service, src/server_main.cpp)           │
│                                                                        │
│   src/http/server.cpp  (cpp-httplib + nlohmann/json)                   │
│      ├── GET   /api/training_data    ──┐                               │
│      ├── GET   /api/training_status   ─┤  reads ui_data/*.json         │
│      ├── POST  /api/manual_test      ──┘  (written by benchmark child) │
│      ├── POST  /api/stress_test      ──── runs locally in-process      │
│      ├── POST  /api/run_training     ──── fork() + execvp() ──┐        │
│      └── POST  /api/stop_training    ──── kill(pid, SIGTERM)  │        │
└───────────────────────────────────────────────────────────────┼────────┘
                                                                │
                                                                ▼
                                ┌────────────────────────────────────────┐
                                │  ssns-benchmark  (one-shot subprocess) │
                                │  src/benchmark.cpp                     │
                                │                                        │
                                │  Calls into src/protocol/training.cpp  │
                                │  Writes ui_data/training_data.json     │
                                │  Writes ui_data/training_status.json   │
                                └────────────────────────────────────────┘
```

The server stays alive serving HTTP and reading status files; the actual training is a separate process so it can be killed cleanly without taking the server down with it.

---

## Module Map

```
include/ssns/                  public headers, one subdir per module
src/
├── server_main.cpp            entry: ssns-server (HTTP service)
├── benchmark.cpp              entry: ssns-benchmark (CLI training runner)
│
├── http/                      cpp-httplib glue, route handlers, subprocess spawn
│   └── server.cpp
│
├── protocol/                  the SSNS training loop itself
│   └── training.cpp           train_step (plaintext + FHE-wired variants)
│
├── nn/                        neural-net building blocks (hand-written, no PyTorch)
│   ├── teacher.cpp            fixed-weight reference MLP
│   ├── server.cpp             ServerNode: holds Teacher + B_FA + gradient helpers
│   ├── client.cpp             ClientNode: trainable Student MLP
│   ├── adam.cpp               Adam optimizer state
│   ├── init.cpp               He / Xavier initializers
│   ├── activations.cpp        ReLU, sigmoid, derivatives
│   └── lr_schedule.cpp        warmup + cosine
│
├── ckks/                      hand-written CKKS implementation
│   ├── backend.cpp            top-level Backend struct
│   ├── params.hpp             N, modulus chain, scale
│   ├── ntt.cpp                Number Theoretic Transform (forward/inverse)
│   ├── crt.cpp                Chinese Remainder Theorem (Garner)
│   ├── modarith.cpp           modular arithmetic primitives
│   ├── poly.cpp               polynomial ops in evaluation form
│   ├── ntt_ops.cpp            pointwise operations in NTT form
│   ├── encoder.cpp            IFFT + RNS lift / inverse
│   ├── plaintext.cpp          Plaintext data type
│   ├── secret_key.cpp         sparse-ternary sk sampling
│   ├── public_key.cpp         BV-style (b, a) pk generation
│   ├── eval_key.cpp           RNS-gadget evk for relinearization
│   ├── encrypt.cpp            encryption of polynomials
│   ├── decrypt.cpp            decryption + CRT reconstruction
│   └── linear_ops.cpp         add, sub, mul-scalar/plain/cipher, rescale, relin
│
├── linalg/                    hand-written matrix math
│   ├── matrix.cpp             Matrix type, allocation, shape utilities
│   └── matmul_native.cpp      register-blocked AVX2+FMA microkernel, std::thread pool
│
├── crypto/
│   └── sha256.cpp             FIPS-180-4 SHA-256 (for key-confirmation exchange)
│
└── io/
    ├── logger.cpp             append-only JSON snapshot writer
    └── status_file.cpp        atomic write of training_status.json

tests/                         Catch2 unit + integration tests
third_party/                   vendored single-header deps (cpp-httplib, json, Catch2)
ui/                            web SPA (HTML/CSS/JS)
```

---

## Training Loop Data Flow

`src/protocol/training.cpp::train_step` is the heart of the protocol. Per epoch:

```
                       CLIENT side                         SERVER side
                       (src/nn/client.cpp)                 (src/nn/server.cpp)

  X (fresh batch) ─────────────────────────────────────────►
                                                           │
                                                           ▼
                                                Y_true = Teacher(X)
                                                (src/nn/teacher.cpp)

  H = ReLU(X · W1_S)
  Y_pred = H · W2_S

  ┌──── if use_fhe ────┐
  │                    │
  │  encrypt(H, pk)    │      H_enc, Y_pred_enc
  │  encrypt(Y_pred,pk)│────────────────────────►
  │  (src/ckks/        │                          error = (Y_pred − Y_true) / batch
  │   encrypt.cpp)     │                          grad_W2 = Hᵀ · error
  │                    │                          err_hidden = error · B_FA
  │                    │                          (src/ckks/linear_ops.cpp:
  │                    │                              cipher × cipher → relinearize → rescale)
  │                    │     grad_W2_enc, err_hidden_enc
  │                    │◄─────────────────────────
  │  decrypt(grad_W2)  │
  │  decrypt(err_hid)  │
  │  (src/ckks/        │
  │   decrypt.cpp)     │
  └────────────────────┘
  ┌──── else (plain) ──┐
  │                    │      H, Y_pred (plaintext)
  │                    │─────────────────────────►
  │                    │                          grad_W2 = Hᵀ · error
  │                    │                          err_hidden = error · B_FA
  │                    │◄─────────────────────────
  └────────────────────┘

  err_hidden *= ReLU'(X · W1_S)           ← client-local, server cannot
                                            compute this without W1_S
  grad_W1 = Xᵀ · err_hidden

  L2-norm-clip(grad_W1), L2-norm-clip(grad_W2)
  Adam update of W1_S, W2_S
  (src/nn/adam.cpp)
```

Loss is recorded each epoch; periodic snapshots (W1_S, W2_S, plus Teacher's W1, W2) are appended to `training_data.json` by `src/io/logger.cpp`.

---

## Feedback Alignment: Why

The server cannot compute the standard backprop signal `error · W2_Sᵀ` because it does not know `W2_S` (it lives only on the client). Standard backprop would require the server to learn the student's weights, defeating the protocol.

Feedback Alignment replaces `W2_Sᵀ` with a **fixed random matrix `B_FA`** held only by the server. The gradient direction it produces is biased but consistent enough that the network still converges, given enough capacity (the wide Student hidden layer in the champion recipe) and enough epochs.

Cost: a noisier gradient, mitigated by Adam's per-parameter adaptive step. Gain: the server never needs the student's weights.

---

## Key Derivation: Why Clusters + Dead-Zone

After training, sigmoid outputs are clustered into groups of `cluster_size` neurons. Each cluster's mean is mapped to a single bit using a dead-zone:

```
mean ≥ 0.5 + dz   →  bit 1 (confident high)
mean ≤ 0.5 − dz   →  bit 0 (confident low)
0.5 − dz < mean < 0.5 + dz  →  discarded (ambiguous, position skipped)
```

The dead-zone width `dz` is what guarantees agreement under residual numerical noise: if Teacher's and Student's cluster means land on the same side of the dead-zone (which they do for sufficiently confident outputs), they emit the same bit. Ambiguous clusters are dropped on both sides simultaneously, so neither party ends up with bits the other lacks. After bit extraction, both sides exchange SHA-256 hashes; only matching keys are accepted.

---

## Matmul Hot Path

`src/linalg/matmul_native.cpp` is the only floating-point matmul in the project. It uses:

- **Register-blocked microkernel** with AVX2 + FMA intrinsics. Inner kernel keeps a small tile of the accumulator in SIMD registers across the K loop.
- **Cache blocking** at the outer loops to keep operand tiles in L1/L2.
- **`std::thread` pool** for the outermost loop, partitioning the M dimension across cores.

There is no BLAS, no Eigen, no PyTorch. This is intentional (see the "Two Hard Rules" section of the README).

---

## Where to start reading

| Want to understand... | Start at |
|---|---|
| Protocol overall | `README.md` |
| Math behind every CKKS step | `docs/SSNS-presentation.pdf` |
| HTTP API contracts | `docs/api.md` |
| Training algorithm | `src/protocol/training.cpp` |
| Client (Student) NN | `src/nn/client.cpp` |
| Server (Teacher + helpers) | `src/nn/server.cpp` |
| CKKS encryption pipeline | `src/ckks/encrypt.cpp` then `linear_ops.cpp` |
| HTTP routing | `src/http/server.cpp` (handlers near top, route table near bottom) |
| Subprocess spawn (fork + execvp) | `src/http/server.cpp::spawn_detached` |
| Matmul optimizations | `src/linalg/matmul_native.cpp` |
