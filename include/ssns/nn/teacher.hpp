// Teacher network — fixed random MLP used as the distillation reference
// in the Teacher-Student key-exchange protocol.  Weights NEVER change
// after construction; the Student is trained to approximate Teacher(X).
//
// Mirrors src/ssns_teacher.py from the Python reference.  Output is
// LINEAR (no sigmoid) — sigmoid is applied identically on both sides
// at key-extraction time, not during training.
#ifndef SSNS_NN_TEACHER_HPP
#define SSNS_NN_TEACHER_HPP

#include <cstdint>

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

class Teacher {
public:
    Teacher(std::size_t input_dim, std::size_t hidden_dim,
            std::size_t output_dim, std::uint64_t seed);

    // Plaintext forward: Y = ReLU(X @ W1) @ W2.  No grad bookkeeping; the
    // Teacher is frozen so backward never flows through it.
    [[nodiscard]] linalg::Matrix forward(const linalg::Matrix& X) const;

    [[nodiscard]] const linalg::Matrix& W1() const noexcept { return W1_; }
    [[nodiscard]] const linalg::Matrix& W2() const noexcept { return W2_; }

private:
    linalg::Matrix W1_;
    linalg::Matrix W2_;
};

}  // namespace ssns::nn

#endif  // SSNS_NN_TEACHER_HPP
