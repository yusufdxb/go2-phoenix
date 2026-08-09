// Copyright 2026 Yusuf Guenena. MIT License.
// See attitude.hpp. Ported literally from ros2_policy_node.py:704-724.
#include "phoenix_core/attitude.hpp"

#include <algorithm>
#include <cmath>

namespace phoenix_core
{

namespace
{

// The operation order below is transcribed from the Python, not simplified.
// gz negates the whole term: `-(1 - 2(x^2 + y^2))`, not `2(x^2+y^2) - 1`.
// They are algebraically equal and not bit-equal, and a shipped version of this
// function once had gx/gy flipped, so the transcription is the point (R9).
Vec3 gravity(double x, double y, double z, double w) noexcept
{
  const double gx = -2.0 * (x * z - w * y);
  const double gy = -2.0 * (y * z + w * x);
  const double gz = -(1.0 - 2.0 * (x * x + y * y));
  return Vec3{static_cast<float>(gx), static_cast<float>(gy), static_cast<float>(gz)};
}

RollPitch rp(double x, double y, double z, double w) noexcept
{
  const double roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
  // np.clip before arcsin: guards the domain when the quaternion is slightly
  // non-unit. std::clamp is avoided because its behaviour on NaN is undefined,
  // and a NaN quaternion must reach the gate ladder as NaN so the nan_in_imu
  // predicate fires, rather than being silently clamped to a valid angle.
  const double s = 2.0 * (w * y - z * x);
  const double clamped = s < -1.0 ? -1.0 : (s > 1.0 ? 1.0 : s);
  const double pitch = std::asin(clamped);
  return RollPitch{roll, pitch};
}

}  // namespace

Vec3 projected_gravity_from_xyzw(const QuatXYZW & q) noexcept
{
  return gravity(q.x, q.y, q.z, q.w);
}

Vec3 projected_gravity_from_wxyz(const QuatWXYZ & q) noexcept
{
  return gravity(q.x, q.y, q.z, q.w);
}

RollPitch roll_pitch_from_xyzw(const QuatXYZW & q) noexcept
{
  return rp(q.x, q.y, q.z, q.w);
}

RollPitch roll_pitch_from_wxyz(const QuatWXYZ & q) noexcept
{
  return rp(q.x, q.y, q.z, q.w);
}

}  // namespace phoenix_core
