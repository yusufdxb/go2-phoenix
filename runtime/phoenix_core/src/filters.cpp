// Copyright 2026 Yusuf Guenena. MIT License.
// See filters.hpp. Ported from phoenix/sim2real/safety.py:78-109.
#include "phoenix_core/filters.hpp"

#include <cmath>
#include <limits>

namespace phoenix_core
{

SlewResult slew_clip(
  const JointArray & target, const JointArray & current, float max_delta) noexcept
{
  SlewResult r;
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    const float t = target[i];
    const float c = current[i];

    if (!std::isfinite(t) || !std::isfinite(c)) {
      r.input_non_finite = true;
    }

    // np.clip is minimum(maximum(a, lo), hi), so its non-finite behaviour is
    // NOT uniform, and the parity fixtures caught this:
    //
    //   NaN target   -> NaN out (NaN propagates through both comparisons)
    //   NaN current  -> NaN out (both bounds are NaN, so the result is NaN)
    //   +/-Inf target -> CLAMPED to the bound like any out-of-range value
    //
    // Treating all three the same, which is the obvious C++ reading, diverges
    // from the Python on infinities. std::clamp is unusable here regardless:
    // its behaviour on NaN is undefined.
    if (!std::isfinite(c) || std::isnan(t)) {
      r.value[i] = std::numeric_limits<float>::quiet_NaN();
      continue;
    }

    const float lo = c - max_delta;
    const float hi = c + max_delta;
    if (t > hi) {
      r.value[i] = hi;
      r.clipped = true;
    } else if (t < lo) {
      r.value[i] = lo;
      r.clipped = true;
    } else {
      r.value[i] = t;
    }
  }
  return r;
}

ClampResult position_clamp(const JointArray & target, const PositionLimits & limits) noexcept
{
  ClampResult r;
  r.value = target;

  for (std::size_t i = 0; i < kNumJoints; ++i) {
    if (!std::isfinite(target[i])) {
      r.non_finite = true;
    }
  }

  // Disabled is the default, so enabling position limits is an explicit act.
  // Silently clamping on a config that never had limits would change the
  // behaviour of an existing deploy without anyone opting in.
  if (!limits.enabled) {
    return r;
  }

  for (std::size_t i = 0; i < kNumJoints; ++i) {
    const float t = r.value[i];
    if (!std::isfinite(t)) {
      continue;  // already flagged; do not manufacture a finite value
    }
    if (t < limits.min[i]) {
      r.value[i] = limits.min[i];
      r.clamped = true;
    } else if (t > limits.max[i]) {
      r.value[i] = limits.max[i];
      r.clamped = true;
    }
  }
  return r;
}

}  // namespace phoenix_core
