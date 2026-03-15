// CKKS depth-0 linear operations — implementation.  See header for the
// op set and preconditions.
#include <ssns/ckks/linear_ops.hpp>

#include <ssns/ckks/ntt_ops.hpp>

#include <stdexcept>
#include <string>

namespace ssns::ckks {

namespace {

void check_compatible(const Ciphertext& a, const Ciphertext& b, const char* op) {
    if (a.scale != b.scale) {
        throw std::invalid_argument(
            std::string(op) + ": scale mismatch (" +
            std::to_string(a.scale) + " vs " + std::to_string(b.scale) + ")");
    }
    if (a.level != b.level) {
        throw std::invalid_argument(
            std::string(op) + ": level mismatch (" +
            std::to_string(a.level) + " vs " + std::to_string(b.level) + ")");
    }
}

void check_compatible(const Ciphertext& ct, const Plaintext& pt, const char* op) {
    if (ct.scale != pt.scale) {
        throw std::invalid_argument(
            std::string(op) + ": scale mismatch (" +
            std::to_string(ct.scale) + " vs " + std::to_string(pt.scale) + ")");
    }
    if (ct.level != pt.level) {
        throw std::invalid_argument(
            std::string(op) + ": level mismatch (" +
            std::to_string(ct.level) + " vs " + std::to_string(pt.level) + ")");
    }
}

}  // namespace

Ciphertext add(const Ciphertext& a, const Ciphertext& b) {
    check_compatible(a, b, "ckks::add");
    Ciphertext out;
    out.c0 = pointwise_add(a.c0, b.c0);
    out.c1 = pointwise_add(a.c1, b.c1);
    out.scale = a.scale;
    out.level = a.level;
    return out;
}

Ciphertext sub(const Ciphertext& a, const Ciphertext& b) {
    check_compatible(a, b, "ckks::sub");
    Ciphertext out;
    out.c0 = pointwise_sub(a.c0, b.c0);
    out.c1 = pointwise_sub(a.c1, b.c1);
    out.scale = a.scale;
    out.level = a.level;
    return out;
}

Ciphertext add_plain(const Ciphertext& ct, const Plaintext& pt) {
    check_compatible(ct, pt, "ckks::add_plain");
    Ciphertext out;
    out.c0 = pointwise_add(ct.c0, pt.poly);
    out.c1 = ct.c1;
    out.scale = ct.scale;
    out.level = ct.level;
    return out;
}

Ciphertext sub_plain(const Ciphertext& ct, const Plaintext& pt) {
    check_compatible(ct, pt, "ckks::sub_plain");
    Ciphertext out;
    out.c0 = pointwise_sub(ct.c0, pt.poly);
    out.c1 = ct.c1;
    out.scale = ct.scale;
    out.level = ct.level;
    return out;
}

}  // namespace ssns::ckks
