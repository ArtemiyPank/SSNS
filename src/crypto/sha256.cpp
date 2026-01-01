// SHA-256 implementation per FIPS 180-4.
//
// Layout:
//   - Round constants K (§4.2.2) and initial hash values H0 (§5.3.3) live in
//     an anonymous namespace below.
//   - process_block runs the 64-round compression function on a single
//     512-bit block already laid out in big-endian byte order.
//   - update() buffers partial blocks; finalize() emits the FIPS padding
//     (0x80, zero pad, 64-bit big-endian message length in bits) and pulls
//     the final digest out of the running state.
#include <ssns/crypto/sha256.hpp>

#include <cstring>

namespace ssns::crypto {

namespace {

// FIPS 180-4 §4.2.2 — first 32 bits of the fractional parts of the cube
// roots of the first 64 primes.
constexpr std::uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

// FIPS 180-4 §5.3.3 — first 32 bits of the fractional parts of the square
// roots of the first 8 primes.
constexpr std::uint32_t H0[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

inline std::uint32_t rotr(std::uint32_t x, unsigned n) {
    // n is always in [1, 31] for SHA-256 — no need to guard against UB.
    return (x >> n) | (x << (32u - n));
}

inline std::uint32_t big_sigma0(std::uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}
inline std::uint32_t big_sigma1(std::uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}
inline std::uint32_t small_sigma0(std::uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}
inline std::uint32_t small_sigma1(std::uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}
inline std::uint32_t ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (~x & z);
}
inline std::uint32_t maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline std::uint32_t load_be32(const std::uint8_t* p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16)
         | (std::uint32_t(p[2]) << 8)  |  std::uint32_t(p[3]);
}

inline void store_be32(std::uint8_t* p, std::uint32_t v) {
    p[0] = std::uint8_t((v >> 24) & 0xff);
    p[1] = std::uint8_t((v >> 16) & 0xff);
    p[2] = std::uint8_t((v >>  8) & 0xff);
    p[3] = std::uint8_t( v        & 0xff);
}

inline void store_be64(std::uint8_t* p, std::uint64_t v) {
    for (int i = 7; i >= 0; --i) {
        p[7 - i] = std::uint8_t((v >> (i * 8)) & 0xff);
    }
}

}  // namespace

Sha256::Sha256() { reset(); }

void Sha256::reset() {
    for (std::size_t i = 0; i < 8; ++i) H_[i] = H0[i];
    buffer_len_ = 0;
    total_bits_ = 0;
}

void Sha256::process_block(const std::uint8_t block[64]) {
    // §6.2.2 step 1 — message schedule.
    std::uint32_t W[64];
    for (std::size_t t = 0; t < 16; ++t) {
        W[t] = load_be32(block + t * 4);
    }
    for (std::size_t t = 16; t < 64; ++t) {
        W[t] = small_sigma1(W[t - 2]) + W[t - 7]
             + small_sigma0(W[t - 15]) + W[t - 16];
    }

    // §6.2.2 step 2 — initialise working variables.
    std::uint32_t a = H_[0], b = H_[1], c = H_[2], d = H_[3];
    std::uint32_t e = H_[4], f = H_[5], g = H_[6], h = H_[7];

    // §6.2.2 step 3 — 64-round compression.
    for (std::size_t t = 0; t < 64; ++t) {
        std::uint32_t T1 = h + big_sigma1(e) + ch(e, f, g) + K[t] + W[t];
        std::uint32_t T2 = big_sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // §6.2.2 step 4 — accumulate.
    H_[0] += a; H_[1] += b; H_[2] += c; H_[3] += d;
    H_[4] += e; H_[5] += f; H_[6] += g; H_[7] += h;
}

void Sha256::update(const std::uint8_t* data, std::size_t len) {
    if (len == 0) return;
    total_bits_ += static_cast<std::uint64_t>(len) * 8u;

    // Top off any partial block already in the buffer.
    if (buffer_len_ > 0) {
        std::size_t need = 64 - buffer_len_;
        std::size_t take = (len < need) ? len : need;
        std::memcpy(buffer_.data() + buffer_len_, data, take);
        buffer_len_ += take;
        data += take;
        len  -= take;
        if (buffer_len_ == 64) {
            process_block(buffer_.data());
            buffer_len_ = 0;
        }
    }

    // Burn full blocks straight from the input.
    while (len >= 64) {
        process_block(data);
        data += 64;
        len  -= 64;
    }

    // Stash the tail.
    if (len > 0) {
        std::memcpy(buffer_.data() + buffer_len_, data, len);
        buffer_len_ += len;
    }
}

void Sha256::update(std::string_view sv) {
    update(reinterpret_cast<const std::uint8_t*>(sv.data()), sv.size());
}

std::array<std::uint8_t, 32> Sha256::finalize() {
    // FIPS 180-4 §5.1.1 padding: append 0x80, then k zero bytes such that
    // (message length + 1 + k) ≡ 56 (mod 64), then 8 bytes of message length
    // in bits, big-endian.
    const std::uint64_t bits_at_finalize = total_bits_;

    std::uint8_t pad[64];
    pad[0] = 0x80;
    for (std::size_t i = 1; i < 64; ++i) pad[i] = 0x00;

    // Pad length: enough to leave exactly 8 bytes for the length field at the
    // end of the final block.
    std::size_t pad_len;
    if (buffer_len_ < 56) {
        pad_len = 56 - buffer_len_;
    } else {
        pad_len = (64 - buffer_len_) + 56;
    }
    update(pad, pad_len);

    // Append the original message length in bits as a 64-bit big-endian int.
    std::uint8_t len_be[8];
    store_be64(len_be, bits_at_finalize);

    // Bypass update()'s bit accounting — these 8 bytes are length, not data.
    std::memcpy(buffer_.data() + buffer_len_, len_be, 8);
    buffer_len_ += 8;
    // buffer_len_ must now equal 64 by construction.
    process_block(buffer_.data());
    buffer_len_ = 0;

    // Emit the digest.
    std::array<std::uint8_t, 32> out;
    for (std::size_t i = 0; i < 8; ++i) {
        store_be32(out.data() + i * 4, H_[i]);
    }

    // Reset so the same instance can hash a fresh message.
    reset();
    return out;
}

std::array<std::uint8_t, 32> sha256(std::string_view input) {
    Sha256 h;
    h.update(input);
    return h.finalize();
}

std::array<std::uint8_t, 32> sha256(const std::uint8_t* data, std::size_t len) {
    Sha256 h;
    h.update(data, len);
    return h.finalize();
}

std::string to_hex(const std::array<std::uint8_t, 32>& digest) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.resize(64);
    for (std::size_t i = 0; i < 32; ++i) {
        out[i * 2]     = hex[(digest[i] >> 4) & 0x0f];
        out[i * 2 + 1] = hex[ digest[i]       & 0x0f];
    }
    return out;
}

}  // namespace ssns::crypto
