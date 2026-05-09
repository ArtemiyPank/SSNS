cmake_minimum_required(VERSION 3.20)
project(SSNS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

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

include_directories(${CMAKE_SOURCE_DIR}/third_party
                    ${CMAKE_SOURCE_DIR}/include)

enable_testing()

# core static lib
# top level entries excluded here built as exes below
file(GLOB_RECURSE SSNS_CORE_SOURCES src/*/*.cpp)
if(SSNS_CORE_SOURCES)
    add_library(ssns_core STATIC ${SSNS_CORE_SOURCES})
    target_include_directories(ssns_core PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party)
    # httplib needs pthread on posix
    find_package(Threads REQUIRED)
    target_link_libraries(ssns_core PUBLIC Threads::Threads)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ssns_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# cli runs one full session
if(EXISTS ${CMAKE_SOURCE_DIR}/src/benchmark.cpp)
    add_executable(ssns-benchmark src/benchmark.cpp)
    if(SSNS_CORE_SOURCES)
        target_link_libraries(ssns-benchmark PRIVATE ssns_core)
    endif()
endif()

# http server for the ide
if(EXISTS ${CMAKE_SOURCE_DIR}/src/server_main.cpp)
    add_executable(ssns-server src/server_main.cpp)
    if(SSNS_CORE_SOURCES)
        target_link_libraries(ssns-server PRIVATE ssns_core)
    endif()
    # httplib pthread
    find_package(Threads REQUIRED)
    target_link_libraries(ssns-server PRIVATE Threads::Threads)
endif()

# one test runner for all test files
# catch2 v2 style one TU has CATCH_CONFIG_MAIN others just include
file(GLOB SSNS_TEST_SOURCES tests/test_*.cpp)
add_executable(ssns_tests ${SSNS_TEST_SOURCES})
if(SSNS_CORE_SOURCES)
    target_link_libraries(ssns_tests PRIVATE ssns_core)
endif()

add_test(NAME ssns_tests COMMAND ssns_tests)

# pass cli binary path to tests
if(TARGET ssns-benchmark)
    set_property(TEST ssns_tests APPEND PROPERTY ENVIRONMENT
        "SSNS_BENCHMARK_BINARY=$<TARGET_FILE:ssns-benchmark>")
endif()
