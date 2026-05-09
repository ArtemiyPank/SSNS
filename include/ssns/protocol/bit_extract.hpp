// bit extraction and key derivation helpers
//
// pure functions no model state no protection no network
// turn noisy sigmoid outputs into confident bits then hash into final key
//
// boundary rules:
//   * mean >= 0.5 + dead_zone -> bit 1 (inclusive)
//   * mean <= 0.5 - dead_zone -> bit 0 (inclusive)
//   * otherwise cluster discarded (no entry in indices)
//   * trailing partial cluster ignored
//   * bits_to_bytes packs big endian zero pads to whole byte
//   * hex_key is sha256(bits_to_bytes(bits)) as 64 char lowercase hex
//
// header only short functions covered by tests
#ifndef SSNS_PROTOCOL_BIT_EXTRACT_HPP
#define SSNS_PROTOCOL_BIT_EXTRACT_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <ssns/crypto/sha256.hpp>

namespace ssns::protocol {

struct ExtractResult {
    std::vector<int> bits;
    std::vector<int> indices;
};

// extract bits from sigmoid via cluster vote
// dead zone discards uncertain ones
// returned vectors parallel bits[k] is the bit at indices[k]
// dz выбирается чтобы перекрыть остаточный шум
inline ExtractResult extract_with_indices(
    const std::vector<double>& values,
    int cluster_size,
    double dead_zone)
{
    ExtractResult out;
    if (cluster_size <= 0) {
        return out;
    }
    const std::size_t cs = static_cast<std::size_t>(cluster_size);
    const std::size_t n = values.size() / cs;       // floor trailing fragment ignored
    // dead zone симметрична относительно 0.5 ширина 2*dz
    // dz должен быть больше чем std(noise)/sqrt(cs) типичный шум на cluster mean
    const double lo = 0.5 - dead_zone;
    const double hi = 0.5 + dead_zone;
    for (std::size_t c = 0; c < n; ++c) {
        double sum = 0.0;
        const std::size_t base = c * cs;
        // repetition code усреднение по cs нейронам подавляет шум в sqrt(cs) раз
        for (std::size_t k = 0; k < cs; ++k) {
            sum += values[base + k];
        }
        const double mean = sum / static_cast<double>(cs);
        if (mean >= hi) {
            // mean уверенно выше 0.5 эмитим 1 шанс что другая сторона попадёт ниже lo пренебрежимо мал
            out.bits.push_back(1);
            out.indices.push_back(static_cast<int>(c));
        } else if (mean <= lo) {
            out.bits.push_back(0);
            out.indices.push_back(static_cast<int>(c));
        }
        // ambiguous both sides drop it so no mismatch on agreed indices
        // именно поэтому mismatch=0 одинаковое отбрасывание гарантирует что shared indices совпадут
    }
    return out;
}

// pack bits big endian zero pad to whole bytes
// bits[0] is MSB of byte 0
inline std::vector<std::uint8_t> bits_to_bytes(const std::vector<int>& bits) {
    if (bits.empty()) {
        return {};
    }
    // round up to multiple of 8 zero padded
    const std::size_t n = bits.size();
    const std::size_t padded_len = (n + 7) / 8 * 8;
    std::vector<std::uint8_t> out(padded_len / 8, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (bits[i]) {
            const std::size_t byte_idx = i / 8;
            const std::size_t bit_pos = 7 - (i % 8);  // big endian bits[0] is MSB
            out[byte_idx] |= static_cast<std::uint8_t>(1u << bit_pos);
        }
    }
    return out;
}

// sha256 of bits_to_bytes as 64 char lowercase hex
inline std::string hex_key(const std::vector<int>& bits) {
    const std::vector<std::uint8_t> bytes = bits_to_bytes(bits);
    const auto digest = ssns::crypto::sha256(bytes.data(), bytes.size());
    return ssns::crypto::to_hex(digest);
}

}  // namespace ssns::protocol

#endif  // SSNS_PROTOCOL_BIT_EXTRACT_HPP
