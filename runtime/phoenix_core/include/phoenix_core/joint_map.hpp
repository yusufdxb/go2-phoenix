// Copyright 2026 Yusuf Guenena. MIT License.
//
// Joint orderings, and the two DIFFERENT permutations between them.
//
// This file exists because of audit risks R6 and R7, which are the quietest
// failures in the whole deploy path: a wrong permutation produces a robot that
// walks with its legs swapped left-for-right. The gait looks plausible. No gate
// fires. Nothing is NaN. It is only obvious if you are watching the animal.
//
// Three orderings are in play and no two are the same:
//
//   Policy order   FL, FR, RL, RR grouped by JOINT
//                  0:FL_hip  1:FR_hip  2:RL_hip  3:RR_hip
//                  4:FL_thigh ... 8:FL_calf ... 11:RR_calf
//
//   ROS order      whatever /joint_states happens to publish. Resolved by
//                  NAME at runtime, never assumed, because a positional copy
//                  is exactly the silent leg swap this file guards against.
//
//   Unitree order  FR, FL, RR, RL grouped by LEG
//                  motor_cmd[0..2] = FR hip/thigh/calf, [3..5] = FL, etc.
//
// R7 is the specific trap: PHOENIX_FOR_MOTOR is the COMMAND-side permutation
// and it is NOT the inverse of the observation-side name remap. Using one
// where the other belongs is a silent left/right swap. Both directions are
// provided here under unambiguous names and the round trip is unit-tested.
#ifndef PHOENIX_CORE__JOINT_MAP_HPP_
#define PHOENIX_CORE__JOINT_MAP_HPP_

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "phoenix_core/types.hpp"

namespace phoenix_core
{

// Canonical policy joint order. Must match configs/sim2real/*.yaml joint_order
// and the order the policy was trained with.
extern const std::array<const char *, kNumJoints> kPolicyJointNames;

// Index into a policy-ordered array for each Unitree motor slot:
//   motor_cmd[k] takes its value from policy[kPhoenixForMotor[k]].
extern const std::array<std::size_t, kNumJoints> kPhoenixForMotor;

// Resolves ROS /joint_states name order onto policy order, once at startup.
//
// The resolution is by name and the mapping is built exactly once, because
// doing it per tick would be both wasteful and a place for a partial failure
// to appear mid-run.
class JointIndexMap
{
public:
  // Returns kMissingJoint if any canonical joint is absent from `names`.
  // Partial success is not a thing here: a missing joint means the whole
  // mapping is unusable, and continuing with eleven of twelve joints is worse
  // than refusing to start.
  Status resolve(const std::vector<std::string> & ros_joint_names) noexcept;

  bool ready() const noexcept {return ready_;}

  // Name of the first joint that was missing, for the startup error message.
  const std::string & missing_joint() const noexcept {return missing_;}

  // Gather a ROS-ordered array into policy order.
  Status gather(
    const float * ros_values, std::size_t n, JointArray & out) const noexcept;

private:
  std::array<std::size_t, kNumJoints> index_{};
  std::string missing_;
  bool ready_ = false;
};

// Policy order -> Unitree motor order. This is the COMMAND direction.
void policy_to_motor(const JointArray & policy, JointArray & motor) noexcept;

// Unitree motor order -> policy order. The exact inverse of the above, and
// deliberately NOT the same as the observation-side name remap (R7).
void motor_to_policy(const JointArray & motor, JointArray & policy) noexcept;

}  // namespace phoenix_core

#endif  // PHOENIX_CORE__JOINT_MAP_HPP_
