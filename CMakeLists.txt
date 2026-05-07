cmake_minimum_required(VERSION 3.20)
project(SSNS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# Phase 8: SSNS_USE_BLAS toggles between OpenBLAS cblas_dgemm and the in-tree
# hand-written matmul kernel.  Default OFF — the academic deliverable forbids
# external numerical libraries on the algorithmic core, and the in-tree AVX2
# kernel in src/linalg/matmul_native.cpp matches or exceeds the system
# OpenBLAS at v9 dimensions on a 12th-gen Intel client core.  Set ON to
# reinstate the cblas_dgemm path (kept in tree as a regression reference).
option(SSNS_USE_BLAS "Use OpenBLAS cblas_dgemm for linalg::matmul" OFF)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Using OpenMP: ${OpenMP_CXX_VERSION}")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -funroll-loops -fno-omit-frame-pointer)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g -fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

if(SSNS_USE_BLAS)
    # OpenBLAS via runtime linkage; prototypes vendored in third_party/cblas_minimal.h.
    # find_library only matches lib<name>.so on Linux, so fall back to scanning for
    # the SONAME-versioned file when a -devel package (which provides the bare .so
    # symlink) is not installed.
    find_library(OPENBLAS_LIB
        NAMES openblaso openblas blas
        PATHS /lib64 /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu)

    if(NOT OPENBLAS_LIB)
        foreach(_candidate /lib64/libopenblaso.so.0
                           /lib64/libopenblas.so.0
                           /usr/lib64/libopenblaso.so.0
                           /usr/lib64/libopenblas.so.0
                           /usr/lib/x86_64-linux-gnu/libopenblas.so.0)
            if(EXISTS "${_candidate}")
                set(OPENBLAS_LIB "${_candidate}" CACHE FILEPATH "" FORCE)
                break()
            endif()
        endforeach()
    endif()

    if(NOT OPENBLAS_LIB)
        message(FATAL_ERROR "libopenblas/libblas not found on system. "
                            "Install the runtime package (e.g. dnf install openblas-openmp).")
    endif()
    message(STATUS "Using BLAS: ${OPENBLAS_LIB}")
else()
    message(STATUS "SSNS_USE_BLAS=OFF — using in-tree native matmul kernel")
    add_compile_definitions(SSNS_NO_BLAS)
endif()

include_directories(${CMAKE_SOURCE_DIR}/third_party
                    ${CMAKE_SOURCE_DIR}/include)

enable_testing()

# Core library — implementations live in src/<group>/*.cpp; tests and the
# CLI / server binaries link against this static archive.  Top-level entry
# points (src/benchmark.cpp, src/server_main.cpp) are excluded — they are
# wired as standalone executables below.
file(GLOB_RECURSE SSNS_CORE_SOURCES src/*/*.cpp)
if(SSNS_CORE_SOURCES)
    add_library(ssns_core STATIC ${SSNS_CORE_SOURCES})
    target_include_directories(ssns_core PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party)
    if(SSNS_USE_BLAS AND OPENBLAS_LIB)
        target_link_libraries(ssns_core PUBLIC ${OPENBLAS_LIB})
    endif()
    # cpp-httplib (used by src/http/server.cpp) needs pthread on POSIX.
    find_package(Threads REQUIRED)
    target_link_libraries(ssns_core PUBLIC Threads::Threads)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ssns_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# CLI subprocess target — same role as scripts/benchmark_fhe_vs_plain.py.
if(EXISTS ${CMAKE_SOURCE_DIR}/src/benchmark.cpp)
    add_executable(ssns-benchmark src/benchmark.cpp)
    if(SSNS_CORE_SOURCES)
        target_link_libraries(ssns-benchmark PRIVATE ssns_core)
    endif()
endif()

# HTTP server binary — mirrors `uvicorn app:app` from the Python reference.
if(EXISTS ${CMAKE_SOURCE_DIR}/src/server_main.cpp)
    add_executable(ssns-server src/server_main.cpp)
    if(SSNS_CORE_SOURCES)
        target_link_libraries(ssns-server PRIVATE ssns_core)
    endif()
    # cpp-httplib needs pthread on POSIX.
    find_package(Threads REQUIRED)
    target_link_libraries(ssns-server PRIVATE Threads::Threads)
endif()

# Single test runner aggregating every tests/test_*.cpp.  Catch2 v2 single-
# header style: one TU defines CATCH_CONFIG_MAIN; the others #include catch.hpp
# without the macro.  We keep that TU at tests/test_main.cpp.
file(GLOB SSNS_TEST_SOURCES tests/test_*.cpp)
add_executable(ssns_tests ${SSNS_TEST_SOURCES})
if(SSNS_CORE_SOURCES)
    target_link_libraries(ssns_tests PRIVATE ssns_core)
endif()
if(SSNS_USE_BLAS AND OPENBLAS_LIB AND NOT SSNS_CORE_SOURCES)
    target_link_libraries(ssns_tests PRIVATE ${OPENBLAS_LIB})
endif()

add_test(NAME ssns_tests COMMAND ssns_tests)

# Tell tests/test_benchmark.cpp where the CLI binary lives.
if(TARGET ssns-benchmark)
    set_property(TEST ssns_tests APPEND PROPERTY ENVIRONMENT
        "SSNS_BENCHMARK_BINARY=$<TARGET_FILE:ssns-benchmark>")
endif()
