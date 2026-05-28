// ssns::protocol::bit_extract: repetition-code + dead-zone extract, big-endian pack, SHA-256 key
//
// boundary semantics must match Python ref src/ssns_clean/bit_extract.py:22-70
//   mean >= 0.5 + dead_zone  -> bit 1 (inclusive)
//   mean <= 0.5 - dead_zone  -> bit 0 (inclusive)
//   otherwise                -> cluster discarded
// trailing fragment (len % cluster_size != 0) is ignored
// bits_to_bytes packs big-endian, zero-pads to whole bytes
// hex_key = SHA-256(bits_to_bytes(bits)) as 64-char lowercase hex
#include <catch.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include <ssns/protocol/bit_extract.hpp>

using ssns::protocol::extract_with_indices;
using ssns::protocol::bits_to_bytes;
using ssns::protocol::hex_key;
using ssns::protocol::ExtractResult;

TEST_CASE("extract_with_indices: empty input produces no bits", "[protocol][bit_extract]") {
    auto r = extract_with_indices({}, 5, 0.1);
    REQUIRE(r.bits.empty());
    REQUIRE(r.indices.empty());
}

TEST_CASE("extract_with_indices: input shorter than cluster_size", "[protocol][bit_extract]") {
    // 3 entries, cluster_size=5 -> n=0 clusters
    auto r = extract_with_indices({0.9, 0.9, 0.9}, 5, 0.1);
    REQUIRE(r.bits.empty());
    REQUIRE(r.indices.empty());
}

TEST_CASE("extract_with_indices: trailing fragment is ignored", "[protocol][bit_extract]") {
    // 7 entries, cluster_size=3 -> n=2 clusters (idx 0,1); last value dropped
    auto r = extract_with_indices({0.9, 0.9, 0.9,  0.1, 0.1, 0.1,  0.5}, 3, 0.1);
    REQUIRE(r.bits == std::vector<int>{1, 0});
    REQUIRE(r.indices == std::vector<int>{0, 1});
}

TEST_CASE("extract_with_indices: cluster mean exactly 0.5+dz emits bit 1 (inclusive)", "[protocol][bit_extract]") {
    // mean = 0.6, dz = 0.1; hi boundary, must yield bit 1
    auto r = extract_with_indices({0.6, 0.6, 0.6, 0.6, 0.6}, 5, 0.1);
    REQUIRE(r.bits == std::vector<int>{1});
    REQUIRE(r.indices == std::vector<int>{0});
}

TEST_CASE("extract_with_indices: cluster mean exactly 0.5-dz emits bit 0 (inclusive)", "[protocol][bit_extract]") {
    // mean = 0.4, dz = 0.1; lo boundary, must yield bit 0
    auto r = extract_with_indices({0.4, 0.4, 0.4, 0.4, 0.4}, 5, 0.1);
    REQUIRE(r.bits == std::vector<int>{0});
    REQUIRE(r.indices == std::vector<int>{0});
}

TEST_CASE("extract_with_indices: cluster mean inside dead zone is discarded", "[protocol][bit_extract]") {
    // mean = 0.5, dz = 0.1; ambiguous, no bit emitted
    auto r = extract_with_indices({0.5, 0.5, 0.5, 0.5, 0.5}, 5, 0.1);
    REQUIRE(r.bits.empty());
    REQUIRE(r.indices.empty());
}

TEST_CASE("extract_with_indices: multi-cluster mixed (confident hi, ambiguous, confident lo)", "[protocol][bit_extract]") {
    // cluster_size=3, dz=0.1
    // c0 mean=0.7 -> bit 1 (idx 0); c1 mean=0.5 -> skipped; c2 mean=0.3 -> bit 0 (idx 2)
    auto r = extract_with_indices({0.7, 0.7, 0.7,  0.5, 0.5, 0.5,  0.3, 0.3, 0.3}, 3, 0.1);
    REQUIRE(r.bits == std::vector<int>{1, 0});
    REQUIRE(r.indices == std::vector<int>{0, 2});
}

TEST_CASE("bits_to_bytes: empty list", "[protocol][bit_extract]") {
    auto out = bits_to_bytes({});
    REQUIRE(out.empty());
}

TEST_CASE("bits_to_bytes: single 0 bit -> one zero byte", "[protocol][bit_extract]") {
    // big-endian, zero-padded; bit lands at MSB of byte 0
    auto out = bits_to_bytes({0});
    REQUIRE(out == std::vector<std::uint8_t>{0x00});
}

TEST_CASE("bits_to_bytes: single 1 bit -> 0x80 (MSB set)", "[protocol][bit_extract]") {
    auto out = bits_to_bytes({1});
    REQUIRE(out == std::vector<std::uint8_t>{0x80});
}

TEST_CASE("bits_to_bytes: 8 alternating bits [1,0,1,0,1,0,1,0] -> 0xAA", "[protocol][bit_extract]") {
    auto out = bits_to_bytes({1, 0, 1, 0, 1, 0, 1, 0});
    REQUIRE(out == std::vector<std::uint8_t>{0xAA});
}

TEST_CASE("bits_to_bytes: 9 bits zero-pads to two bytes (big-endian)", "[protocol][bit_extract]") {
    // [1,1,1,1,1,1,1,1, 1, 0,0,0,0,0,0,0] -> [0xFF, 0x80]
    auto out = bits_to_bytes({1, 1, 1, 1, 1, 1, 1, 1, 1});
    REQUIRE(out == std::vector<std::uint8_t>{0xFF, 0x80});
}

TEST_CASE("hex_key: empty bits hashes to SHA-256 of empty input", "[protocol][bit_extract][hex]") {
    // SHA-256("") well-known constant
    const std::string sha256_empty =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    auto k = hex_key({});
    REQUIRE(k == sha256_empty);
}
