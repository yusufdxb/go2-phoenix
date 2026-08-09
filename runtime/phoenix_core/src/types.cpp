// Copyright 2026 Yusuf Guenena. MIT License.
// See types.hpp.
#include "phoenix_core/types.hpp"

namespace phoenix_core
{

const char * reason_string(AbortReason reason) noexcept
{
  // Strings match the Python reason prefixes in phoenix/sim2real/gate.py so a
  // lab log from either runtime reads identically. first_message_timeout is a
  // prefix here: the Python appends a CSV of the missing topics, which the ROS
  // adapter reconstructs from the flags on GateDecision.
  switch (reason) {
    case AbortReason::kNone:
      return "none";
    case AbortReason::kMaxRuntime:
      return "max_runtime";
    case AbortReason::kExternalEstop:
      return "external_estop";
    case AbortReason::kEstopHeartbeatStale:
      return "estop_heartbeat_stale";
    case AbortReason::kEstopPublisherMissing:
      return "estop_publisher_missing";
    case AbortReason::kSensorMissing:
      return "sensor_missing";
    case AbortReason::kSensorStale:
      return "sensor_stale";
    case AbortReason::kFirstMessageTimeout:
      return "first_message_timeout";
    case AbortReason::kNanInJointState:
      return "nan_in_joint_state";
    case AbortReason::kNanInImu:
      return "nan_in_imu";
    case AbortReason::kAttitude:
      return "attitude";
    case AbortReason::kUnknownSafetyGate:
      return "unknown_safety_gate";
  }
  return "unknown_safety_gate";
}

}  // namespace phoenix_core
