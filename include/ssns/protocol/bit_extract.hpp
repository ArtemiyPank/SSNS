// Bit-extraction and key-derivation utilities for SSNS-Clean (C++ port).
//
// Pure functions ported one-to-one from `src/ssns_clean/bit_extract.py` in the
// reference Python implementation.  No model state, no protection logic, no
// network code — these helpers only convert noisy sigmoid outputs into
// confident bits and derive the final symmetric key from those bits.
//
// Boundary semantics MUST match the Python reference exactly:
//   * mean >= 0.5 + dead_zone   -> bit 1 (inclusive boundary)
//   * mean <= 0.5 - dead_zone   -> bit 0 (inclusive boundary)
//   * otherwise                  -> cluster discarded (no entry in `indices`)
//   * trailing entries that don't fill a full cluster are ignored
//   * `bits_to_bytes` packs big-endian, zero-pads to the next whole byte
//   * `hex_key` is SHA-256(bits_to_bytes(bits)) as 64-char lowercase hex
//
// Header-only: every operation is short and the test suite covers all
// boundary cases, so we keep the API close to its declarations and avoid a
// .cpp split.
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

// Cluster-mean repetition-code + dead-zone bit extraction.
//   - For each cluster of `cluster_size` consecutive entries, compute mean.
//   - mean >= 0.5 + dead_zone -> bit 1, cluster index appended to `indices`.
//   - mean <= 0.5 - dead_zone -> bit 0, cluster index appended to `indices`.
//   - otherwise -> cluster discarded (its index is NOT in `indices`).
// Returns parallel vectors: `bits[k]` is the bit value for `indices[k]`.
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
    const std::size_t n = values.size() / cs;       // floor — trailing fragment ignored
    const double lo = 0.5 - dead_zone;
    const double hi = 0.5 + dead_zone;
    for (std::size_t c = 0; c < n; ++c) {
        double sum = 0.0;
        const std::size_t base = c * cs;
        for (std::size_t k = 0; k < cs; ++k) {
            sum += values[base + k];
        }
        const double mean = sum / static_cast<double>(cs);
        if (mean >= hi) {
            out.bits.push_back(1);
            out.indices.push_back(static_cast<int>(c));
        } else if (mean <= lo) {
            out.bits.push_back(0);
            out.indices.push_back(static_cast<int>(c));
        }
        // Otherwise: ambiguous, discard.
    }
    return out;
}

// Pack bits big-endian into bytes; zero-pad to whole bytes.
//   bits[0] is the MSB of byte 0, bits[7] is the LSB of byte 0, etc.
inline std::vector<std::uint8_t> bits_to_bytes(const std::vector<int>& bits) {
    if (bits.empty()) {
        return {};
    }
    // Round up to a multiple of 8 with zero padding.
    const std::size_t n = bits.size();
    const std::size_t padded_len = (n + 7) / 8 * 8;
    std::vector<std::uint8_t> out(padded_len / 8, 0);
    for (std::size_t i = 0; i < n; ++i) {
        if (bits[i]) {
            const std::size_t byte_idx = i / 8;
            const std::size_t bit_pos  = 7 - (i % 8);  // big-endian
            out[byte_idx] |= static_cast<std::uint8_t>(1u << bit_pos);
        }
    }
    return out;
}

// SHA-256 of bits_to_bytes, returned as 64-char lowercase hex digest.
inline std::string hex_key(const std::vector<int>& bits) {
    const std::vector<std::uint8_t> bytes = bits_to_bytes(bits);
    const auto digest = ssns::crypto::sha256(bytes.data(), bytes.size());
    return ssns::crypto::to_hex(digest);
}

}  // namespace ssns::protocol

#endif  // SSNS_PROTOCOL_BIT_EXTRACT_HPP
