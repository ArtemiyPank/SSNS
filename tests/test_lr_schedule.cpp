// warmup-cosine LR schedule, mirrors Python formula at src/ssns_clean/client.py:170-192
// fixed-seed run must be replayable bit-for-bit
#include <catch.hpp>

#include <ssns/nn/lr_schedule.hpp>

using ssns::nn::warmup_cosine_lr;

TEST_CASE("lr_schedule: warmup phase ramps linearly to lr_max", "[nn][lr]") {
    // total=100, warmup_frac=0.05 -> warmup_steps = round(5) = 5
    // step 0 -> lr_max * 1/5 = 0.002
    // step 1 -> lr_max * 2/5 = 0.004
    // step 4 -> lr_max * 5/5 = 0.010 (peak, reached at step warmup_steps-1)
    REQUIRE(warmup_cosine_lr(0, 100, 0.05, 0.01, 0.0) == Approx(0.002).margin(1e-12));
    REQUIRE(warmup_cosine_lr(1, 100, 0.05, 0.01, 0.0) == Approx(0.004).margin(1e-12));
    REQUIRE(warmup_cosine_lr(4, 100, 0.05, 0.01, 0.0) == Approx(0.010).margin(1e-12));
}

TEST_CASE("lr_schedule: cosine decay starts at lr_max", "[nn][lr]") {
    // step == warmup_steps: frac = 0 -> cos(0) = 1 -> lr = lr_max
    REQUIRE(warmup_cosine_lr(5, 100, 0.05, 0.01, 0.0) == Approx(0.010).margin(1e-12));
}

TEST_CASE("lr_schedule: cosine decay ends at lr_min", "[nn][lr]") {
    // step == total_steps: frac = 1 -> cos(pi) = -1 -> lr = lr_min
    REQUIRE(warmup_cosine_lr(100, 100, 0.05, 0.01, 0.0) == Approx(0.0).margin(1e-12));
    REQUIRE(warmup_cosine_lr(100, 100, 0.05, 0.01, 0.001) == Approx(0.001).margin(1e-12));
}

TEST_CASE("lr_schedule: cosine midpoint is half of (lr_max - lr_min) plus lr_min", "[nn][lr]") {
    // halfway through cosine phase: frac = 0.5 -> cos(pi/2) = 0
    // -> lr = lr_min + 0.5 * (lr_max - lr_min) = average(lr_min, lr_max)
    // careful: warmup_steps = max(1, round(0)) = 1, so cosine spans 1..total
    // for midpoint test use total=201, tiny warmup_frac, midpoint = step 101
    // denom = 200, frac at step 101 = 100/200 = 0.5, cos(pi*0.5)=0
    const double lr = warmup_cosine_lr(101, 201, 1.0/201.0, 0.01, 0.0);
    REQUIRE(lr == Approx(0.005).margin(1e-12));
}

TEST_CASE("lr_schedule: warmup_steps clamps to 1 when warmup_frac=0", "[nn][lr]") {
    // total=100, frac=0 -> warmup_steps = max(1, 0) = 1
    // step 0 -> lr_max * 1/1 = lr_max
    REQUIRE(warmup_cosine_lr(0, 100, 0.0, 0.01, 0.0) == Approx(0.01).margin(1e-12));
}

TEST_CASE("lr_schedule: never goes below lr_min in cosine phase", "[nn][lr]") {
    // past nominal end (step > total_steps) frac is clamped to 1
    REQUIRE(warmup_cosine_lr(150, 100, 0.05, 0.01, 0.0001) == Approx(0.0001).margin(1e-12));
}
