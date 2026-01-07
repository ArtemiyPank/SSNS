// Warmup-cosine learning-rate schedule.  Pure function, header-only.
// Mirrors src/ssns_clean/client.py:170-192 in the Python reference exactly:
//
//   warmup_steps = max(1, round(total_steps * warmup_frac))
//   if step < warmup_steps:
//       lr = lr_max * (step + 1) / warmup_steps
//   else:
//       denom = max(1, total_steps - warmup_steps)
//       frac  = min(1.0, (step - warmup_steps) / denom)
//       lr    = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(pi * frac))
#ifndef SSNS_NN_LR_SCHEDULE_HPP
#define SSNS_NN_LR_SCHEDULE_HPP

#include <algorithm>
#include <cmath>
#include <numbers>

namespace ssns::nn {

[[nodiscard]] inline double warmup_cosine_lr(
    long step, long total_steps,
    double warmup_frac, double lr_max, double lr_min = 0.0)
{
    long warmup_steps = static_cast<long>(std::lround(
        static_cast<double>(total_steps) * warmup_frac));
    if (warmup_steps < 1) warmup_steps = 1;

    if (step < warmup_steps) {
        return lr_max * static_cast<double>(step + 1)
                      / static_cast<double>(warmup_steps);
    }
    long denom = total_steps - warmup_steps;
    if (denom < 1) denom = 1;
    double frac = static_cast<double>(step - warmup_steps)
                / static_cast<double>(denom);
    if (frac > 1.0) frac = 1.0;
    return lr_min + (lr_max - lr_min) * 0.5
                  * (1.0 + std::cos(std::numbers::pi * frac));
}

}  // namespace ssns::nn

#endif  // SSNS_NN_LR_SCHEDULE_HPP
