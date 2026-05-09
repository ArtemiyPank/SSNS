# SSNS — C++

Teacher–Student key-exchange protocol over Feedback Alignment with real
CKKS homomorphic encryption on the wire. Two hard rules:

1. **All neural-net math is hand-written.** No PyTorch, no Eigen, no
   external BLAS. Matmul uses an in-tree register-blocked AVX2 kernel
   with `std::thread` parallelism (`src/linalg/matmul_native.cpp`).
2. **CKKS is implemented from scratch.** Real polynomial-ring
   cryptography: NTT, RNS, encoder, keygen, encryption, eight
   homomorphic operations, RNS-gadget relinearisation, rescale. No
   SEAL, OpenFHE, HElib, TenSEAL.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires: C++20 compiler, CMake ≥ 3.20. No external numerical
libraries.

## Run the Visual IDE

```sh
./build/ssns-server --host 127.0.0.1 --port 8765 --ui-dir ui
```

Then open `http://127.0.0.1:8765/`.

## Layout

- `include/ssns/` — public headers
- `src/` — implementations
- `tests/` — Catch2 unit + integration tests (221 cases)
- `third_party/` — vendored single-header deps (cpp-httplib,
  nlohmann/json, Catch2 v2)
- `ui/` — frontend SPA (HTML/CSS/JS)
- `docs/project_book/` — K13 project book (23 markdown sections)

## Methodology

Strict TDD: failing test first, then minimum code to pass, then
refactor. See `docs/project_book/` for the full project book in K13
academic format.
