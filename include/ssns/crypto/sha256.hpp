// sha256 FIPS 180-4 self contained no openssl
// used by SSNS to derive final symmetric key from agreed bits
//
// one shot:
//     auto digest = ssns::crypto::sha256("hello");
//     std::string hex = ssns::crypto::to_hex(digest);
//
// streaming:
//     ssns::crypto::Sha256 h;
//     h.update(part1);
//     h.update(part2);
//     auto digest = h.finalize();   // resets state
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ssns::crypto {

// streaming sha256 engine one instance per message
class Sha256 {
public:
    // 32 bytes
    static constexpr std::size_t digest_size = 32;

    // ctor sets FIPS 180-4 §5.3.3 initial state
    Sha256();

    // feed bytes any chunk size
    void update(const std::uint8_t* data, std::size_t len);
    void update(std::string_view sv);

    // pad run final block return digest reset
    std::array<std::uint8_t, digest_size> finalize();

private:
    // restore initial hash values
    void reset();
    // 64 round compression on one block
    void process_block(const std::uint8_t block[64]);

    // инварианты: buffer_len_ всегда в [0, 63] полный блок съедается process_block сразу
    // total_bits_ считает только пользовательские байты служебная padding-длина не учитывается
    std::array<std::uint32_t, 8> H_;       // running state H[0..7] аккумулирует hash state
    std::array<std::uint8_t, 64> buffer_;  // partial block держит хвост недобранного блока
    std::size_t buffer_len_;               // bytes in buffer
    std::uint64_t total_bits_;             // total length in bits
};

// one shot string view
std::array<std::uint8_t, 32> sha256(std::string_view input);
// one shot raw bytes
std::array<std::uint8_t, 32> sha256(const std::uint8_t* data, std::size_t len);

// 32 byte digest to 64 lowercase hex chars
std::string to_hex(const std::array<std::uint8_t, 32>& digest);

}  // namespace ssns::crypto
