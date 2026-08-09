// Copyright 2026 Yusuf Guenena. MIT License.
// See gate.hpp. Line-for-line port of phoenix/sim2real/gate.py.
#include "phoenix_core/gate.hpp"

#include <cmath>

namespace phoenix_core
{

namespace
{

constexpr double kNsPerSec = 1e9;

double age_s(std::int64_t last_ns, std::int64_t now_ns) noexcept
{
  return static_cast<double>(now_ns - last_ns) / kNsPerSec;
}

bool all_finite(const JointArray & a) noexcept
{
  for (float v : a) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

bool all_finite(const Vec3d & a) noexcept
{
  for (double v : a) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}

bool quat_finite(const QuatXYZW & q) noexcept
{
  return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w);
}

}  // namespace

bool estop_is_active(
  std::int64_t last_msg_ns, bool value, bool value_known, std::int64_t now_ns,
  double timeout_s) noexcept
{
  // Fail closed: never seen counts as asserted (safety.py:36-37).
  if (last_msg_ns == SensorSnapshot::kNeverSeen || !value_known) {
    return true;
  }
  if (age_s(last_msg_ns, now_ns) > timeout_s) {
    return true;
  }
  return value;
}

bool sensor_is_stale(std::int64_t last_msg_ns, std::int64_t now_ns, double timeout_s) noexcept
{
  if (last_msg_ns == SensorSnapshot::kNeverSeen) {
    return true;
  }
  return age_s(last_msg_ns, now_ns) > timeout_s;
}

GateDecision evaluate_gates(
  const SensorSnapshot & s, const GateConfig & c, bool already_latched) noexcept
{
  GateDecision d;

  // Rank 1: runtime watchdog. Latches but emits nothing, because the Python
  // latched here and then hit the already-latched early return on the same
  // tick. The asymmetry with every other abort is deliberate.
  if (s.elapsed_s > c.max_runtime_s && !already_latched) {
    d.outcome = Outcome::kLatchSilent;
    d.reason = AbortReason::kMaxRuntime;
    return d;
  }

  // Rank 0/1 continued: an abort latched on any previous tick, including the
  // out-of-band external estop, which latches between ticks.
  if (already_latched) {
    d.outcome = Outcome::kSilent;
    return d;
  }

  // Rank 3: startup gate (safety.py:167-211).
  const bool all_seen = s.seen_estop && s.seen_imu && s.seen_joint_state;
  if (!all_seen) {
    if (age_s(s.node_started_ns, s.now_ns) <= c.first_message_timeout_s) {
      d.outcome = Outcome::kPublishDefault;
      return d;
    }
    d.outcome = Outcome::kLatchAndPublishDefault;
    d.reason = AbortReason::kFirstMessageTimeout;
    d.missing_estop = !s.seen_estop;
    d.missing_imu = !s.seen_imu;
    d.missing_joint_state = !s.seen_joint_state;
    return d;
  }

  // Rank 2/4: estop-chain integrity and sensor freshness
  // (is_ready_to_command_motion, safety.py:112-164). Reason ordering is part
  // of the contract: a missing publisher is distinguished from an asserted
  // estop, which is distinguished from a stale heartbeat, because the reason
  // string is what drives the lab post-mortem.
  if (s.estop_last_ns == SensorSnapshot::kNeverSeen || !s.estop_value_known) {
    d.outcome = Outcome::kLatchAndPublishDefault;
    d.reason = AbortReason::kEstopPublisherMissing;
    return d;
  }
  if (estop_is_active(
      s.estop_last_ns, s.estop_value, s.estop_value_known, s.now_ns, c.estop_timeout_s))
  {
    d.outcome = Outcome::kLatchAndPublishDefault;
    d.reason = s.estop_value ? AbortReason::kExternalEstop : AbortReason::kEstopHeartbeatStale;
    return d;
  }
  if (s.imu_last_ns == SensorSnapshot::kNeverSeen ||
    s.joint_state_last_ns == SensorSnapshot::kNeverSeen)
  {
    d.outcome = Outcome::kLatchAndPublishDefault;
    d.reason = AbortReason::kSensorMissing;
    return d;
  }
  if (sensor_is_stale(s.imu_last_ns, s.now_ns, c.sensor_timeout_s) ||
    sensor_is_stale(s.joint_state_last_ns, s.now_ns, c.sensor_timeout_s))
  {
    d.outcome = Outcome::kLatchAndPublishDefault;
    d.reason = AbortReason::kSensorStale;
    return d;
  }

  // Rank 4: joint-state validity.
  if (!all_finite(s.joint_pos) || !all_finite(s.joint_vel)) {
    d.outcome = Outcome::kLatchAndPublishDefault;
    d.reason = AbortReason::kNanInJointState;
    return d;
  }

  // Rank 4: IMU validity. Must precede the attitude gate, which cannot fire on
  // NaN because abs(NaN) > threshold is false in both languages.
  if (!quat_finite(s.quat) || !all_finite(s.ang_vel)) {
    d.outcome = Outcome::kLatchAndPublishDefault;
    d.reason = AbortReason::kNanInImu;
    return d;
  }

  // Rank 6: attitude abort. Thresholds are asymmetric (0.8 pitch, 0.6 roll)
  // and the comparison is strictly greater than.
  if (std::fabs(s.pitch) > c.pitch_rad || std::fabs(s.roll) > c.roll_rad) {
    d.outcome = Outcome::kLatchAndPublishDefault;
    d.reason = AbortReason::kAttitude;
    return d;
  }

  // Rank 8: nothing seized the output; the policy may command motion.
  d.outcome = Outcome::kRunPolicy;
  return d;
}

}  // namespace phoenix_core
