// nn::CleanServer owns Teacher + B_FA + plaintext FA gradient compute
// mirrors src/ssns_clean/server.py
#include <catch.hpp>

#include <ssns/linalg/matrix.hpp>
#include <ssns/nn/server.hpp>

using ssns::linalg::Matrix;
using ssns::nn::CleanServer;

TEST_CASE("CleanServer: B_FA shape and seed-determinism", "[nn][server]") {
    CleanServer a(/*input_dim=*/8, /*student_hidden=*/16,
                  /*teacher_hidden=*/4, /*output_dim=*/12,
                  /*teacher_seed=*/42, /*bfa_seed=*/43);
    CleanServer b(8, 16, 4, 12, 42, 43);

    REQUIRE(a.b_fa().rows() == 12);   // [output_dim, student_hidden]
    REQUIRE(a.b_fa().cols() == 16);
    for (std::size_t i = 0; i < a.b_fa().size(); ++i)
        REQUIRE(a.b_fa().data()[i] == b.b_fa().data()[i]);
}

TEST_CASE("CleanServer: teacher_forward delegates to Teacher", "[nn][server]") {
    CleanServer s(8, 16, 4, 12, 42, 43);
    auto X = Matrix::zeros(3, 8);
    auto Y = s.teacher_forward(X);
    REQUIRE(Y.rows() == 3);
    REQUIRE(Y.cols() == 12);
    // ReLU(0) @ W2 = 0
    for (std::size_t i = 0; i < Y.size(); ++i)
        REQUIRE(Y.data()[i] == 0.0);
}

TEST_CASE("CleanServer: compute_gradients math", "[nn][server][grad]") {
    // hand-crafted fixture, exact integer arithmetic
    // batch=2, hidden=2, output=3, B_FA scaled by sqrt(1/3)
    CleanServer s(/*input_dim=*/4, /*student_hidden=*/2,
                  /*teacher_hidden=*/2, /*output_dim=*/3,
                  /*teacher_seed=*/0, /*bfa_seed=*/0);

    auto H      = Matrix::from_rows({{1.0, 0.0}, {0.5, 2.0}});      // [batch, hidden]
    auto Y_pred = Matrix::from_rows({{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}});
    auto Y_true = Matrix::from_rows({{0.5, 2.0, 4.0}, {0.0, 1.0, 2.5}});

    auto [grad_W2, error_hidden] = s.compute_gradients(H, Y_pred, Y_true);

    // error = (Y_pred - Y_true) / batch
    //   row 0 = [0.5, 0, -1]   / 2 = [0.25, 0,    -0.5]
    //   row 1 = [0.5, 0.5, 0]  / 2 = [0.25, 0.25,  0.0]
    //
    // grad_W2 = H^T @ error, shape [2, 3]
    //   row 0 (H[:,0] = [1, 0.5]):
    //     1*[0.25, 0, -0.5] + 0.5*[0.25, 0.25, 0] = [0.375, 0.125, -0.5]
    //   row 1 (H[:,1] = [0, 2]):
    //     0*err[0] + 2*[0.25, 0.25, 0] = [0.5, 0.5, 0]
    REQUIRE(grad_W2.rows() == 2);
    REQUIRE(grad_W2.cols() == 3);
    REQUIRE(grad_W2(0, 0) == Approx(0.375));
    REQUIRE(grad_W2(0, 1) == Approx(0.125));
    REQUIRE(grad_W2(0, 2) == Approx(-0.5));
    REQUIRE(grad_W2(1, 0) == Approx(0.5));
    REQUIRE(grad_W2(1, 1) == Approx(0.5));
    REQUIRE(grad_W2(1, 2) == Approx(0.0));

    // error_hidden = error @ B_FA, [batch, output] @ [output, hidden]
    // shape [batch=2, hidden=2]
    REQUIRE(error_hidden.rows() == 2);
    REQUIRE(error_hidden.cols() == 2);
}

TEST_CASE("CleanServer: compute_gradients rejects mismatched batch", "[nn][server][grad]") {
    CleanServer s(4, 2, 2, 3, 0, 0);
    auto H      = Matrix::zeros(2, 2);
    auto Y_pred = Matrix::zeros(3, 3);   // batch=3, mismatched with H.batch=2
    auto Y_true = Matrix::zeros(2, 3);
    REQUIRE_THROWS(s.compute_gradients(H, Y_pred, Y_true));
}
