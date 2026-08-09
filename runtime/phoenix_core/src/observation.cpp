// Copyright 2026 Yusuf Guenena. MIT License.
// See observation.hpp. Port of phoenix/sim2real/observation.py:85-97.
#include "phoenix_core/observation.hpp"

#include <cmath>

namespace phoenix_core
{

namespace
{
bool finite3(const Vec3 & v) noexcept
{
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}
bool finite12(const JointArray & v) noexcept
{
  for (float x : v) {
    if (!std::isfinite(x)) {return false;}
  }
  return true;
}
}  // namespace

Status ObservationBuilder::initialize(const JointArray & default_q) noexcept
{
  if (!finite12(default_q)) {
    return Status::kNonFinite;
  }
  default_q_ = default_q;
  ready_ = true;
  return Status::kOk;
}

Status ObservationBuilder::build(const ObservationInputs & in, ObsArray & out) const noexcept
{
  if (!ready_) {
    return Status::kNotInitialized;
  }
  if (!finite3(in.base_lin_vel) || !finite3(in.base_ang_vel) ||
    !finite3(in.projected_gravity) || !finite3(in.velocity_command) ||
    !finite12(in.joint_pos) || !finite12(in.joint_vel) || !finite12(in.last_action))
  {
    return Status::kNonFinite;
  }

  for (std::size_t i = 0; i < 3; ++i) {
    out[kObsBaseLinVel + i] = in.base_lin_vel[i];
    out[kObsBaseAngVel + i] = in.base_ang_vel[i];
    out[kObsProjectedGravity + i] = in.projected_gravity[i];
    out[kObsVelocityCommand + i] = in.velocity_command[i];
  }
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    // The only arithmetic in the whole builder. Relative, not absolute.
    out[kObsJointPosRel + i] = in.joint_pos[i] - default_q_[i];
    out[kObsJointVel + i] = in.joint_vel[i];
    out[kObsLastAction + i] = in.last_action[i];
  }
  return Status::kOk;
}

}  // namespace phoenix_core
