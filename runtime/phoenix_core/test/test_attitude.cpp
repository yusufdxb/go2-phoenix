// Copyright 2026 Yusuf Guenena. MIT License.
//
// Mirror of tests/test_projected_gravity.py. The sign convention is the point:
// a shipped version of this expression once had gx/gy flipped, which is a
// mirror-image gravity vector in the policy's observation and is not obviously
// wrong from the robot's behaviour.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "phoenix_core/attitude.hpp"

using phoenix_core::QuatWXYZ;
using phoenix_core::QuatXYZW;
using phoenix_core::projected_gravity_from_wxyz;
using phoenix_core::projected_gravity_from_xyzw;
using phoenix_core::roll_pitch_from_xyzw;

TEST(Attitude, IdentityQuaternionGivesGravityStraightDown)
{
  const auto g = projected_gravity_from_xyzw(QuatXYZW{0.0f, 0.0f, 0.0f, 1.0f});
  EXPECT_NEAR(g[0], 0.0, 1e-7);
  EXPECT_NEAR(g[1], 0.0, 1e-7);
  EXPECT_NEAR(g[2], -1.0, 1e-7);
}

TEST(Attitude, GravityIsUnitLengthForUnitQuaternions)
{
  // Sweep a set of unit quaternions; |g| must stay 1.
  const float qs[][4] = {
    {0.0f, 0.0f, 0.0f, 1.0f},
    {0.2588190f, 0.0f, 0.0f, 0.9659258f},    // 30 deg roll
    {0.0f, 0.3826834f, 0.0f, 0.9238795f},    // 45 deg pitch
    {0.0f, 0.0f, 0.7071068f, 0.7071068f},    // 90 deg yaw
    {0.5f, 0.5f, 0.5f, 0.5f},
  };
  for (const auto & q : qs) {
    const auto g = projected_gravity_from_xyzw(QuatXYZW{q[0], q[1], q[2], q[3]});
    const double n = std::sqrt(
      static_cast<double>(g[0]) * g[0] + static_cast<double>(g[1]) * g[1] +
      static_cast<double>(g[2]) * g[2]);
    EXPECT_NEAR(n, 1.0, 1e-6);
  }
}

TEST(Attitude, YawDoesNotChangeProjectedGravity)
{
  // Gravity is invariant to rotation about the gravity axis. A sign error in
  // gx/gy would break this while leaving |g| == 1 intact, so this is the test
  // that actually catches the historical bug.
  const auto g0 = projected_gravity_from_xyzw(QuatXYZW{0.0f, 0.0f, 0.0f, 1.0f});
  const auto g90 = projected_gravity_from_xyzw(
    QuatXYZW{0.0f, 0.0f, 0.7071068f, 0.7071068f});
  EXPECT_NEAR(g0[0], g90[0], 1e-6);
  EXPECT_NEAR(g0[1], g90[1], 1e-6);
  EXPECT_NEAR(g0[2], g90[2], 1e-6);
}

TEST(Attitude, PitchForwardTipsGravityForward)
{
  // +45 deg about body Y. Pins the SIGN of gx, which is what the flipped
  // version got wrong.
  const auto g = projected_gravity_from_xyzw(QuatXYZW{0.0f, 0.3826834f, 0.0f, 0.9238795f});
  EXPECT_NEAR(g[0], std::sin(M_PI / 4.0), 1e-5);
  EXPECT_NEAR(g[1], 0.0, 1e-6);
  EXPECT_NEAR(g[2], -std::cos(M_PI / 4.0), 1e-5);
}

TEST(Attitude, RollRightTipsGravitySideways)
{
  // +30 deg about body X. Pins the sign of gy.
  const auto g = projected_gravity_from_xyzw(QuatXYZW{0.2588190f, 0.0f, 0.0f, 0.9659258f});
  EXPECT_NEAR(g[0], 0.0, 1e-6);
  EXPECT_NEAR(g[1], -std::sin(M_PI / 6.0), 1e-5);
  EXPECT_NEAR(g[2], -std::cos(M_PI / 6.0), 1e-5);
}

TEST(Attitude, WxyzAndXyzwAgreeOnTheSameRotation)
{
  // The two entry points exist so a convention swap is a compile-time visible
  // choice rather than an invisible reordering of four floats.
  const QuatXYZW a{0.2588190f, 0.1f, -0.3f, 0.9659258f};
  const QuatWXYZ b{a.w, a.x, a.y, a.z};
  const auto ga = projected_gravity_from_xyzw(a);
  const auto gb = projected_gravity_from_wxyz(b);
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(ga[static_cast<std::size_t>(i)], gb[static_cast<std::size_t>(i)]);
  }
}

TEST(Attitude, NonFiniteQuaternionPropagatesRatherThanBeingClamped)
{
  // The gate ladder relies on NaN reaching it so the nan_in_imu predicate can
  // fire. If asin's argument were clamped with something NaN-unsafe, a corrupt
  // IMU would yield a plausible finite angle and defeat that gate.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto rp = roll_pitch_from_xyzw(QuatXYZW{nan, 0.0f, 0.0f, 1.0f});
  EXPECT_TRUE(std::isnan(rp.roll));
  EXPECT_TRUE(std::isnan(rp.pitch));

  const auto g = projected_gravity_from_xyzw(QuatXYZW{nan, 0.0f, 0.0f, 1.0f});
  EXPECT_TRUE(std::isnan(g[0]) || std::isnan(g[1]) || std::isnan(g[2]));
}

TEST(Attitude, RollPitchIdentityIsZero)
{
  const auto rp = roll_pitch_from_xyzw(QuatXYZW{0.0f, 0.0f, 0.0f, 1.0f});
  EXPECT_NEAR(rp.roll, 0.0, 1e-9);
  EXPECT_NEAR(rp.pitch, 0.0, 1e-9);
}

TEST(Attitude, PitchIsClampedIntoAsinDomain)
{
  // A slightly non-unit quaternion can push 2(wy - zx) just outside [-1,1].
  // numpy clips before arcsin; so must this, or the result is NaN and the
  // attitude gate misfires as an IMU fault.
  const auto rp = roll_pitch_from_xyzw(QuatXYZW{0.0f, 0.7071f, 0.0f, 0.7071f});
  EXPECT_FALSE(std::isnan(rp.pitch));
  EXPECT_NEAR(std::fabs(rp.pitch), M_PI / 2.0, 1e-2);
}
