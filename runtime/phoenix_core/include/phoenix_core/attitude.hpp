// Copyright 2026 Yusuf Guenena. MIT License.
//
// Projected gravity and roll/pitch from a body quaternion.
//
// Ported literally from phoenix/sim2real/ros2_policy_node.py:709-721 and the
// tests in tests/test_projected_gravity.py. "Literally" is load-bearing: a
// previously shipped version of that expression had gx and gy flipped, so the
// port must reproduce the operation order rather than an algebraically
// equivalent rearrangement (audit risk R9).
//
// Two entry points, one per quaternion convention, because the runtime reads
// Unitree LowState ([w,x,y,z]) while the ROS path carries sensor_msgs/Imu
// ((x,y,z,w)). Taking a typed quaternion instead of four bare floats is what
// stops the two being swapped silently (audit risk R8).
//
// Intermediates are double and the result is narrowed once at the end. The
// Python side evaluates in double as well, because ROS Imu orientation fields
// are float64, so matching the width here is what makes bit-exact parity
// achievable rather than merely close.
#ifndef PHOENIX_CORE__ATTITUDE_HPP_
#define PHOENIX_CORE__ATTITUDE_HPP_

#include "phoenix_core/types.hpp"

namespace phoenix_core
{

// Gravity direction expressed in the body frame, unit length for a unit
// quaternion. Matches _projected_gravity_from_quat.
Vec3 projected_gravity_from_xyzw(const QuatXYZW & q) noexcept;
Vec3 projected_gravity_from_wxyz(const QuatWXYZ & q) noexcept;

// Roll and pitch in radians. Matches _rpy_from_quat_xyzw; yaw is not returned
// because the deploy path never uses it.
struct RollPitch
{
  double roll;
  double pitch;
};

RollPitch roll_pitch_from_xyzw(const QuatXYZW & q) noexcept;
RollPitch roll_pitch_from_wxyz(const QuatWXYZ & q) noexcept;

}  // namespace phoenix_core

#endif  // PHOENIX_CORE__ATTITUDE_HPP_
