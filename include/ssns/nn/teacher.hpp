// teacher: fixed random 2-layer mlp defines fn the student must learn
//
// shared "ground truth": teacher with given seed = deterministic fn
// student trained via fa to approximate it
//
// after construction teacher weights never change backward never flows through it
// at keygen sigmoid applied to teacher/student output
// network output itself is linear sigmoid is part of keygen not the model
#ifndef SSNS_NN_TEACHER_HPP
#define SSNS_NN_TEACHER_HPP

#include <cstdint>

#include <ssns/linalg/matrix.hpp>

namespace ssns::nn {

class Teacher {
public:
    // he-init both weights from seed if w2_scale != 1 applied to W2 once at ctor
    // w2_scale knob to control output magnitude before sigmoid
    // bigger -> more confident clusters
    Teacher(std::size_t input_dim, std::size_t hidden_dim,
            std::size_t output_dim, std::uint64_t seed,
            double w2_scale = 1.0);

    // plaintext forward Y = relu(X @ W1) @ W2
    // no grad bookkeeping teacher is frozen
    [[nodiscard]] linalg::Matrix forward(const linalg::Matrix& X) const;

    [[nodiscard]] const linalg::Matrix& W1() const noexcept { return W1_; }
    [[nodiscard]] const linalg::Matrix& W2() const noexcept { return W2_; }

private:
    linalg::Matrix W1_;
    linalg::Matrix W2_;
};

}  // namespace ssns::nn

#endif  // SSNS_NN_TEACHER_HPP
