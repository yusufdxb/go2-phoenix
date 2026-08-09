// Copyright 2026 Yusuf Guenena. MIT License.
//
// The deploy-path safety gate ladder, as a pure function.
//
// Line-for-line port of phoenix/sim2real/gate.py, which is itself the
// extraction of the ladder that used to live inline in
// ros2_policy_node._control_step. The Python module is the parity oracle: any
// divergence between the two is a bug in this file, and tests/test_gate_ladder.py
// and test_gate_parity.cpp exist to keep them pinned together.
//
// The function is pure. It touches no ROS, no clock, no inference session and
// no member state, so the composition (which gate wins, what each publishes,
// whether it latches) is exhaustively testable without a robot or a graph.
// That composition had zero test coverage in Python before the extraction.
//
// Freshness is computed from a caller-supplied monotonic timestamp in
// nanoseconds. The core never reads a clock itself: doing so would make it
// untestable and would tie safety timing to whatever clock the ROS adapter
// happened to be configured with (the Python bridge and node currently
// disagree on exactly this, audit finding L8).
#ifndef PHOENIX_CORE__GATE_HPP_
#define PHOENIX_CORE__GATE_HPP_

#include <cstdint>

#include "phoenix_core/types.hpp"

namespace phoenix_core
{

// What the caller must do with this tick. Encodes the full output contract:
// whether to publish, what to publish, and whether to latch.
enum class Outcome : std::uint8_t
{
  // Abort already latched on a previous tick. Publish nothing, ever again.
  kSilent = 0,
  // Latch now, publish nothing. Used only by max_runtime, matching the Python
  // behaviour where that latch fell through to the already-latched early
  // return without emitting a pose.
  kLatchSilent = 1,
  // Publish the default stand pose, do not latch. The only repeatedly
  // publishing state; used while waiting for every topic to be seen once.
  kPublishDefault = 2,
  // Latch now and publish the default stand pose exactly once.
  kLatchAndPublishDefault = 3,
  // All gates passed. Run inference, apply the shield, filter and publish.
  kRunPolicy = 4,
};

struct GateConfig
{
  double max_runtime_s;
  double estop_timeout_s;
  double sensor_timeout_s;
  double first_message_timeout_s;
  double pitch_rad;
  double roll_rad;
};

// Everything the ladder is allowed to look at, sampled once per tick.
//
// joint_pos / joint_vel are already remapped into policy joint order by the
// caller, because that remap is resolved by joint name and a positional copy
// would be a silent leg swap (audit risk R6).
//
// The *_last_ns fields use a sentinel rather than an optional: kNeverSeen means
// no message has ever arrived, which the predicates treat as stale. That is
// the fail-closed convention the Python uses via None.
struct SensorSnapshot
{
  static constexpr std::int64_t kNeverSeen = -1;

  std::int64_t now_ns;
  double elapsed_s;
  std::int64_t node_started_ns;

  bool seen_estop;
  bool seen_imu;
  bool seen_joint_state;

  std::int64_t estop_last_ns = kNeverSeen;
  bool estop_value = false;
  bool estop_value_known = false;
  std::int64_t imu_last_ns = kNeverSeen;
  std::int64_t joint_state_last_ns = kNeverSeen;

  JointArray joint_pos{};
  JointArray joint_vel{};

  QuatXYZW quat{};
  Vec3d ang_vel{};

  double roll = 0.0;
  double pitch = 0.0;
};

struct GateDecision
{
  Outcome outcome = Outcome::kRunPolicy;
  AbortReason reason = AbortReason::kNone;

  // Which first messages were missing, when reason is kFirstMessageTimeout.
  // Kept as flags rather than a formatted string so the hot path never builds
  // one; the ROS adapter renders the CSV that the Python emits.
  bool missing_estop = false;
  bool missing_imu = false;
  bool missing_joint_state = false;

  bool latches() const noexcept
  {
    return outcome == Outcome::kLatchSilent || outcome == Outcome::kLatchAndPublishDefault;
  }

  bool publishes_default() const noexcept
  {
    return outcome == Outcome::kPublishDefault || outcome == Outcome::kLatchAndPublishDefault;
  }
};

// Evaluate the ladder. The first gate to fire wins and short-circuits.
GateDecision evaluate_gates(
  const SensorSnapshot & snapshot, const GateConfig & config, bool already_latched) noexcept;

// Freshness predicates, exposed for direct testing. Ported from
// phoenix/sim2real/safety.py:13-55.
bool estop_is_active(
  std::int64_t last_msg_ns, bool value, bool value_known, std::int64_t now_ns,
  double timeout_s) noexcept;

bool sensor_is_stale(std::int64_t last_msg_ns, std::int64_t now_ns, double timeout_s) noexcept;

}  // namespace phoenix_core

#endif  // PHOENIX_CORE__GATE_HPP_
