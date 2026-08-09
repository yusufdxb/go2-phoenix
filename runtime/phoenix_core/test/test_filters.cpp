// Copyright 2026 Yusuf Guenena. MIT License.
//
// Terminal actuation filters. The non-finite cases are the reason this file
// exists. np.clip is minimum(maximum(a, lo), hi), so it propagates NaN but
// CLAMPS infinities, std::clamp on NaN is undefined behaviour, and a naive
// port turns a corrupt command into a plausible finite one. The asymmetry
// between NaN and Inf was found by the parity fixtures, not by inspection.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "phoenix_core/filters.hpp"

using phoenix_core::ClampResult;
using phoenix_core::JointArray;
using phoenix_core::PositionLimits;
using phoenix_core::SlewResult;
using phoenix_core::kMaxDeltaPerStepRad;
using phoenix_core::kNumJoints;
using phoenix_core::position_clamp;
using phoenix_core::slew_clip;

namespace
{
JointArray filled(float v)
{
  JointArray a;
  a.fill(v);
  return a;
}
const float kNan = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();
}  // namespace

TEST(SlewClip, WithinLimitPassesThroughUnchanged)
{
  const auto current = filled(0.0f);
  auto target = filled(0.1f);
  const auto r = slew_clip(target, current);
  EXPECT_FALSE(r.clipped);
  EXPECT_FALSE(r.input_non_finite);
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(r.value[i], 0.1f);
  }
}

TEST(SlewClip, ClampsAboveAndBelow)
{
  const auto current = filled(0.0f);
  auto target = filled(0.0f);
  target[0] = 5.0f;
  target[1] = -5.0f;
  const auto r = slew_clip(target, current);
  EXPECT_TRUE(r.clipped);
  EXPECT_FLOAT_EQ(r.value[0], kMaxDeltaPerStepRad);
  EXPECT_FLOAT_EQ(r.value[1], -kMaxDeltaPerStepRad);
}

TEST(SlewClip, BoundIsRelativeToMeasuredCurrent)
{
  // The cap is current +/- delta, not an absolute band. Clipping against the
  // wrong reference is the divergence risk between the two clip sites.
  auto current = filled(1.0f);
  auto target = filled(2.0f);
  const auto r = slew_clip(target, current);
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(r.value[i], 1.0f + kMaxDeltaPerStepRad);
  }
}

TEST(SlewClip, BoundaryIsInclusive)
{
  const auto current = filled(0.0f);
  auto target = filled(kMaxDeltaPerStepRad);
  const auto r = slew_clip(target, current);
  EXPECT_FALSE(r.clipped) << "exactly at the limit must not count as clipped";
}

TEST(SlewClip, ConstantMatchesThePythonSingleSourceOfTruth)
{
  // phoenix/sim2real/safety.py:95. Both the policy node and the lowcmd bridge
  // import that one constant; this is its native mirror.
  EXPECT_FLOAT_EQ(kMaxDeltaPerStepRad, 0.175f);
}

TEST(SlewClip, NonFiniteTargetNaNPropagatesButInfClamps)
{
  const auto current = filled(0.0f);
  for (float bad : {kNan, kInf, -kInf}) {
    auto target = filled(0.0f);
    target[3] = bad;
    const auto r = slew_clip(target, current);
    EXPECT_TRUE(r.input_non_finite) << "caller must fail closed on corrupt input";
    if (std::isnan(bad)) {
      // NaN propagates, matching np.clip.
      EXPECT_TRUE(std::isnan(r.value[3]));
    } else {
      // Infinities are CLAMPED by np.clip, so the value is finite; the
      // corruption is visible only via input_non_finite. This asymmetry is
      // numpy's, not ours, and the parity fixtures pin it.
      EXPECT_TRUE(std::isfinite(r.value[3]));
      EXPECT_TRUE(r.clipped);
    }
  }
}

TEST(SlewClip, NonFiniteCurrentIsFlagged)
{
  auto current = filled(0.0f);
  current[5] = kNan;
  const auto target = filled(0.1f);
  const auto r = slew_clip(target, current);
  EXPECT_TRUE(r.input_non_finite) << "bounds derived from NaN are meaningless";
  EXPECT_TRUE(std::isnan(r.value[5])) << "np.clip with NaN bounds yields NaN";
}

TEST(SlewClip, FlagsAreIndependent)
{
  const auto current = filled(0.0f);
  auto target = filled(0.0f);
  target[0] = 9.0f;   // clipped
  target[1] = kNan;   // non-finite
  const auto r = slew_clip(target, current);
  EXPECT_TRUE(r.clipped);
  EXPECT_TRUE(r.input_non_finite);
}

// --------------------------------------------------------------------------
// Position clamp
// --------------------------------------------------------------------------

TEST(PositionClamp, DisabledByDefaultAndPassesThrough)
{
  // Enabling position limits must be an explicit act: silently clamping on a
  // config that never had limits would change an existing deploy's behaviour.
  PositionLimits limits;
  auto target = filled(99.0f);
  const auto r = position_clamp(target, limits);
  EXPECT_FALSE(r.clamped);
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(r.value[i], 99.0f);
  }
}

TEST(PositionClamp, ClampsToConfiguredRange)
{
  PositionLimits limits;
  limits.enabled = true;
  limits.min = filled(-1.0f);
  limits.max = filled(1.0f);

  auto target = filled(0.0f);
  target[0] = 5.0f;
  target[1] = -5.0f;
  const auto r = position_clamp(target, limits);
  EXPECT_TRUE(r.clamped);
  EXPECT_FLOAT_EQ(r.value[0], 1.0f);
  EXPECT_FLOAT_EQ(r.value[1], -1.0f);
  EXPECT_FLOAT_EQ(r.value[2], 0.0f);
}

TEST(PositionClamp, NonFiniteIsFlaggedAndNotManufacturedIntoAValue)
{
  PositionLimits limits;
  limits.enabled = true;
  limits.min = filled(-1.0f);
  limits.max = filled(1.0f);

  auto target = filled(0.0f);
  target[4] = kNan;
  const auto r = position_clamp(target, limits);
  EXPECT_TRUE(r.non_finite);
  EXPECT_TRUE(std::isnan(r.value[4])) << "must not become a plausible finite command";
}

TEST(PositionClamp, NonFiniteFlaggedEvenWhenDisabled)
{
  PositionLimits limits;  // disabled
  auto target = filled(0.0f);
  target[2] = kInf;
  const auto r = position_clamp(target, limits);
  EXPECT_TRUE(r.non_finite);
}
