// Copyright 2026 Yusuf Guenena. MIT License.
//
// The 48-dimensional policy observation.
//
// Port of phoenix/sim2real/observation.py. Term order is a contract with the
// trained network, not a layout choice, and it is transcribed from Isaac Lab's
// velocity_env_cfg PolicyCfg via the Python builder:
//
//   [0:3]   base_lin_vel
//   [3:6]   base_ang_vel
//   [6:9]   projected_gravity
//   [9:12]  velocity_command
//   [12:24] joint_pos - default_q     (RELATIVE, not absolute)
//   [24:36] joint_vel
//   [36:48] last_action
//
// There are no per-term scale factors. Adding one would be a silent
// distribution shift.
//
// R4, and it is not a bug to be fixed here: base_lin_vel is fed as ZEROS at
// deploy while training used true body linear velocity. The robot has no such
// estimator. Wiring one in would change the input distribution the policy was
// trained and the shield was calibrated on, so it is a research decision, not
// a port decision. This module reproduces the zeros and says so.
//
// R5: default_q is both the action offset and the joint_pos_rel reference
// (Isaac Lab's use_default_offset=True), so a wrong default pose corrupts the
// observation AND the command. Five of six shipped configs disagree with the
// training asset, so the caller must supply it explicitly and the builder
// refuses to invent one.
#ifndef PHOENIX_CORE__OBSERVATION_HPP_
#define PHOENIX_CORE__OBSERVATION_HPP_

#include <array>

#include "phoenix_core/types.hpp"

namespace phoenix_core
{

// Term offsets, exposed so tests and telemetry refer to them by name rather
// than by a magic number that can drift out of sync with the layout.
constexpr std::size_t kObsBaseLinVel = 0;
constexpr std::size_t kObsBaseAngVel = 3;
constexpr std::size_t kObsProjectedGravity = 6;
constexpr std::size_t kObsVelocityCommand = 9;
constexpr std::size_t kObsJointPosRel = 12;
constexpr std::size_t kObsJointVel = 24;
constexpr std::size_t kObsLastAction = 36;

using ObsArray = std::array<float, kObsDim>;

struct ObservationInputs
{
  // Fed as zeros at deploy. Kept as an explicit field rather than hard-coded
  // so the choice is visible at the call site and testable (R4).
  Vec3 base_lin_vel{{0.0f, 0.0f, 0.0f}};
  Vec3 base_ang_vel{};
  Vec3 projected_gravity{};
  Vec3 velocity_command{};
  JointArray joint_pos{};   // absolute, policy order
  JointArray joint_vel{};   // policy order
  JointArray last_action{};  // previous raw policy output
};

class ObservationBuilder
{
public:
  // default_q must come from the checkpoint's provenance, not from a guess.
  Status initialize(const JointArray & default_q) noexcept;

  const JointArray & default_q() const noexcept {return default_q_;}

  // Assembles the observation. Returns kNonFinite if any input is non-finite;
  // the gate ladder should already have caught that, so reaching it here means
  // the caller skipped a gate.
  Status build(const ObservationInputs & in, ObsArray & out) const noexcept;

private:
  JointArray default_q_{};
  bool ready_ = false;
};

}  // namespace phoenix_core

#endif  // PHOENIX_CORE__OBSERVATION_HPP_
