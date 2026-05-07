# SSNS — C++ port

Teacher-Student key-exchange protocol over Feedback Alignment with real CKKS
homomorphic encryption on the wire.  C++ rewrite of the Python prototype at
`../../Python/SSNS_mvp` with two hard rules:

1. **All neural-net math is hand-written.**  No PyTorch, no Eigen.  Matmul
   uses an in-tree register-blocked AVX2 kernel with `std::thread`
   parallelism (`src/linalg/matmul_native.cpp`).  OpenBLAS is opt-in via
   `-DSSNS_USE_BLAS=ON` and kept in tree only as a regression reference.
2. **CKKS is implemented from scratch.**  Real polynomial-ring cryptography:
   NTT, RNS, encoder, keygen, encryption, homomorphic operations, rescale.
   No SEAL, OpenFHE, HElib, TenSEAL.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires: C++20 compiler, CMake >= 3.20.  No external numerical libraries by
default; the optional `-DSSNS_USE_BLAS=ON` switch enables the legacy
OpenBLAS path (requires `libopenblaso.so` at link/runtime).

## Layout

- `include/ssns/` — public headers
- `src/` — implementations
- `tests/` — Catch2 unit + integration tests
- `third_party/` — vendored single-header deps (cpp-httplib, nlohmann/json, Catch2 v2)
- `ui/` — frontend (carried over verbatim from the Python project)

## Methodology

Strict TDD: failing test first, then minimum code to pass, then refactor.
See `/home/artemiypank/.claude/plans/cozy-waddling-kite.md` for the full
phased plan.
