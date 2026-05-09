// warmup-cosine lr schedule pure fn header-only
//
// shape:
//   step in [0 warmup_steps)     lr ramps linear 0 -> lr_max
//   step in [warmup_steps total) lr decays half-cos lr_max -> lr_min
//
// why warmup
// adam v starts at 0 and bias correction blows up effective step early on
// ramping lr from 0 prevents weight blowup
#ifndef SSNS_NN_LR_SCHEDULE_HPP
#define SSNS_NN_LR_SCHEDULE_HPP

#include <algorithm>
#include <cmath>
#include <numbers>

namespace ssns::nn {

// lr for given step linear warmup then cos decay lr_max -> lr_min
// warmup_steps = round(total*warmup_frac) but at least 1
// чтобы не делить на ноль если warmup_frac = 0
[[nodiscard]] inline double warmup_cosine_lr(
    long step, long total_steps,
    double warmup_frac, double lr_max, double lr_min = 0.0)
{
    long warmup_steps = static_cast<long>(std::lround(
        static_cast<double>(total_steps) * warmup_frac));
    if (warmup_steps < 1) warmup_steps = 1;

    if (step < warmup_steps) {
        // linear ramp +1 so first step isnt lr=0 (avoid wasted iter)
        // линейный warmup плавно повышает lr пока adam v не накопит надёжную статистику
        return lr_max * static_cast<double>(step + 1)
                      / static_cast<double>(warmup_steps);
    }
    // cos half-period 0 to pi cos goes 1 -> -1 so 0.5*(1+cos) goes 1 -> 0
    // cos плавнее чем step decay в конце lr медленно тает помогает settle в плоском минимуме
    long denom = total_steps - warmup_steps;
    if (denom < 1) denom = 1;
    double frac = static_cast<double>(step - warmup_steps)
                / static_cast<double>(denom);
    // clamp frac на 1 чтобы за пределами total_steps не получить отрицательный cos
    if (frac > 1.0) frac = 1.0;
    return lr_min + (lr_max - lr_min) * 0.5
                  * (1.0 + std::cos(std::numbers::pi * frac));
}

}  // namespace ssns::nn

#endif  // SSNS_NN_LR_SCHEDULE_HPP
