// Copyright 2026 Yusuf Guenena. MIT License.
// See joint_map.hpp. Ported from observation.py and motor_crc.py:42-55.
#include "phoenix_core/joint_map.hpp"

namespace phoenix_core
{

// Matches configs/sim2real/*.yaml joint_order.
const std::array<const char *, kNumJoints> kPolicyJointNames = {
  "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
  "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
  "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
};

// motor_cmd[k] = policy[kPhoenixForMotor[k]]. Transcribed from
// motor_crc.py:42-55, not re-derived: re-deriving a permutation from a comment
// is how the two directions get confused.
const std::array<std::size_t, kNumJoints> kPhoenixForMotor = {
  1, 5, 9,    // FR: hip, thigh, calf
  0, 4, 8,    // FL
  3, 7, 11,   // RR
  2, 6, 10,   // RL
};

Status JointIndexMap::resolve(const std::vector<std::string> & ros_joint_names) noexcept
{
  ready_ = false;
  missing_.clear();

  for (std::size_t i = 0; i < kNumJoints; ++i) {
    const std::string want(kPolicyJointNames[i]);
    bool found = false;
    for (std::size_t j = 0; j < ros_joint_names.size(); ++j) {
      if (ros_joint_names[j] == want) {
        index_[i] = j;
        found = true;
        break;
      }
    }
    if (!found) {
      missing_ = want;
      return Status::kMissingJoint;
    }
  }
  ready_ = true;
  return Status::kOk;
}

Status JointIndexMap::gather(
  const float * ros_values, std::size_t n, JointArray & out) const noexcept
{
  if (!ready_ || ros_values == nullptr) {
    return Status::kNotInitialized;
  }
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    if (index_[i] >= n) {
      return Status::kDimMismatch;
    }
    out[i] = ros_values[index_[i]];
  }
  return Status::kOk;
}

void policy_to_motor(const JointArray & policy, JointArray & motor) noexcept
{
  for (std::size_t k = 0; k < kNumJoints; ++k) {
    motor[k] = policy[kPhoenixForMotor[k]];
  }
}

void motor_to_policy(const JointArray & motor, JointArray & policy) noexcept
{
  for (std::size_t k = 0; k < kNumJoints; ++k) {
    policy[kPhoenixForMotor[k]] = motor[k];
  }
}

}  // namespace phoenix_core
