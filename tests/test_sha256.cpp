// ssns::crypto::Sha256, FIPS 180-4 SHA-256
// self-implemented (no OpenSSL), validated vs NIST KAT vectors
// covers one-shot, streaming, to_hex helper
#include <catch.hpp>

#include <ssns/crypto/sha256.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

using ssns::crypto::Sha256;
using ssns::crypto::sha256;
using ssns::crypto::to_hex;

namespace {

// KATs from FIPS 180-4 Appendix B and NIST CAVS examples
constexpr const char* kEmpty   = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
constexpr const char* kAbc     = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
constexpr const char* kFox     = "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592";
constexpr const char* kMillion = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
// 448-bit (56-byte) FIPS vector, exercises alt padding path (length field crosses to 2nd block)
constexpr const char* kAbcdbcde =
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";

}  // namespace

TEST_CASE("SHA-256: empty string KAT", "[crypto][sha256]") {
    REQUIRE(to_hex(sha256("")) == kEmpty);
}

TEST_CASE("SHA-256: 'abc' KAT (FIPS 180-4 Appendix B.1)", "[crypto][sha256]") {
    REQUIRE(to_hex(sha256("abc")) == kAbc);
}

TEST_CASE("SHA-256: 'The quick brown fox...' KAT", "[crypto][sha256]") {
    REQUIRE(to_hex(sha256("The quick brown fox jumps over the lazy dog")) == kFox);
}

TEST_CASE("SHA-256: 56-byte 'abcdbcdec...' KAT (FIPS 180-4 Appendix B.2)",
          "[crypto][sha256]") {
    REQUIRE(to_hex(sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))
            == kAbcdbcde);
}

TEST_CASE("SHA-256: one million 'a' characters KAT (FIPS 180-4 Appendix B.3)",
          "[crypto][sha256][slow]") {
    std::string million(1'000'000, 'a');
    REQUIRE(to_hex(sha256(million)) == kMillion);
}

TEST_CASE("SHA-256: streaming update equals one-shot", "[crypto][sha256]") {
    const std::string msg =
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "Sphinx of black quartz, judge my vow.";

    // uneven chunks that cross 64-byte block boundaries
    Sha256 hasher;
    hasher.update(std::string_view(msg).substr(0, 7));    // 7 bytes
    hasher.update(std::string_view(msg).substr(7, 60));   // crosses block 1
    hasher.update(std::string_view(msg).substr(67, 1));   // single byte
    hasher.update(std::string_view(msg).substr(68));      // remainder
    auto streamed = hasher.finalize();

    REQUIRE(to_hex(streamed) == to_hex(sha256(msg)));
}

TEST_CASE("SHA-256: finalize resets internal state", "[crypto][sha256]") {
    Sha256 h;
    h.update("abc");
    auto first = h.finalize();
    REQUIRE(to_hex(first) == kAbc);

    // after finalize the same instance must hash a fresh message correctly
    h.update("");
    auto second = h.finalize();
    REQUIRE(to_hex(second) == kEmpty);
}

TEST_CASE("SHA-256: byte-pointer overload matches string_view overload",
          "[crypto][sha256]") {
    const std::string msg = "abc";
    auto a = sha256(msg);
    auto b = sha256(reinterpret_cast<const std::uint8_t*>(msg.data()), msg.size());
    REQUIRE(to_hex(a) == to_hex(b));
}

TEST_CASE("SHA-256: digest size is 32 bytes", "[crypto][sha256]") {
    std::array<std::uint8_t, 32> d = sha256("");
    REQUIRE(d.size() == 32);
}

TEST_CASE("SHA-256: to_hex produces 64 lowercase hex chars", "[crypto][sha256]") {
    auto hex = to_hex(sha256(""));
    REQUIRE(hex.size() == 64);
    for (char c : hex) {
        bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        REQUIRE(ok);
    }
}
