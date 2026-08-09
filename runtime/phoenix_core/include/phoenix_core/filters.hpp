// Copyright 2026 Yusuf Guenena. MIT License.
//
// Terminal actuation filters.
//
// These are not precedence ranks. They apply to whatever the gate ladder
// decided, unconditionally, on every path including every abort pose. In the
// Python the abort path bypasses the slew clip entirely and is bounded only by
// the separate bridge process (audit finding L3); keeping the filters terminal
// here is what closes that hole without depending on another process being
// alive.
//
// Split into two named filters rather than one "action filter" so each is
// separately testable and so a future position clamp cannot be conflated with
// the rate clamp that already ships.
#ifndef PHOENIX_CORE__FILTERS_HPP_
#define PHOENIX_CORE__FILTERS_HPP_

#include "phoenix_core/types.hpp"

namespace phoenix_core
{

// Slew-rate cap applied to every joint command on every control tick.
//
// Ported from phoenix/sim2real/safety.py:95. The Python defines this once and
// imports it into the policy node, the lowcmd bridge, and the training-time
// rate limiter so the three cannot drift. This constant is the native mirror
// of that single source; a parity test pins them together.
constexpr float kMaxDeltaPerStepRad = 0.175f;

// Clip target to current +/- max_delta, element-wise.
//
// Non-finite handling is explicit, and is the reason this is not std::clamp
// (whose behaviour on NaN is undefined). np.clip is
// minimum(maximum(a, lo), hi), so it does NOT treat all non-finite values
// alike, and the parity fixtures caught a port that assumed it did:
//
//   NaN target    -> NaN out
//   NaN current   -> NaN out, because both bounds are then NaN
//   +/-Inf target -> clamped to the bound, exactly like any out-of-range value
//
// See audit risk R13.
struct SlewResult
{
  JointArray value{};
  // True if any element was clipped. Used for telemetry: the one hardware
  // measurement this repo has is a slew-saturation percentage, so the native
  // runtime must be able to reproduce that number.
  bool clipped = false;
  // True if any element of target OR current was non-finite on input.
  //
  // This reports INPUT corruption, not output shape: an infinite target is
  // clamped to a finite bound above, so the value alone no longer shows that
  // a corrupt command arrived. The Python has no equivalent signal, and its
  // downstream finiteness check would accept the clamped value; surfacing it
  // here is a deliberate strengthening so the caller can fail closed on a
  // corrupt command rather than actuating a sanitized version of it.
  bool input_non_finite = false;
};

SlewResult slew_clip(
  const JointArray & target, const JointArray & current,
  float max_delta = kMaxDeltaPerStepRad) noexcept;

// Clamp to absolute joint position limits.
//
// This does not exist in the Python deploy path at all: the node's own
// docstring records that only rate limiting is applied (audit finding L6).
// It is introduced here as a terminal filter, disabled unless limits are
// configured, so enabling it is an explicit act rather than a silent
// behaviour change on an existing config.
struct PositionLimits
{
  bool enabled = false;
  JointArray min{};
  JointArray max{};
};

struct ClampResult
{
  JointArray value{};
  bool clamped = false;
  bool non_finite = false;
};

ClampResult position_clamp(const JointArray & target, const PositionLimits & limits) noexcept;

}  // namespace phoenix_core

#endif  // PHOENIX_CORE__FILTERS_HPP_
