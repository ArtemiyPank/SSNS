// SHA-256 (FIPS 180-4) — self-implemented; no OpenSSL or third-party crypto.
//
// Used by the SSNS protocol to derive the final symmetric key from the bits
// extracted out of the Teacher/Student agreement layer.  The implementation
// processes 512-bit blocks per FIPS 180-4 §6.2 with the standard 64-round
// compression function and the eight initial hash values in §5.3.3.
//
// Usage (one-shot):
//     auto digest = ssns::crypto::sha256("hello");
//     std::string hex = ssns::crypto::to_hex(digest);
//
// Usage (streaming):
//     ssns::crypto::Sha256 h;
//     h.update(part1);
//     h.update(part2);
//     auto digest = h.finalize();   // resets internal state
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ssns::crypto {

class Sha256 {
public:
    // Output is 256 bits = 32 bytes.
    static constexpr std::size_t digest_size = 32;

    Sha256();

    // Feed bytes into the hash.  Safe to call repeatedly across arbitrary
    // chunk boundaries — the engine buffers partial blocks internally.
    void update(const std::uint8_t* data, std::size_t len);
    void update(std::string_view sv);

    // Append FIPS 180-4 padding, emit the final digest, and reset the engine
    // so the same instance can be reused for a fresh message.
    std::array<std::uint8_t, digest_size> finalize();

private:
    void reset();
    void process_block(const std::uint8_t block[64]);

    std::array<std::uint32_t, 8> H_;       // running hash state
    std::array<std::uint8_t, 64> buffer_;  // partial block
    std::size_t buffer_len_;               // bytes currently in buffer_
    std::uint64_t total_bits_;             // total message length in bits
};

// One-shot helpers.
std::array<std::uint8_t, 32> sha256(std::string_view input);
std::array<std::uint8_t, 32> sha256(const std::uint8_t* data, std::size_t len);

// Lowercase hex encoding (64 chars, no separators or prefix).
std::string to_hex(const std::array<std::uint8_t, 32>& digest);

}  // namespace ssns::crypto
