// Copyright 2026 Yusuf Guenena. MIT License.
//
// Fixed-size value types for the Phoenix native deploy runtime.
//
// Nothing in phoenix_core allocates on the control path, depends on ROS, or
// throws across an API boundary. Every type here is a POD with a compile-time
// size so a control tick can be executed against stack storage only.
//
// Sizes are pinned by static_assert rather than left implicit: a silent change
// to the joint count or observation width is exactly the class of error the
// native port exists to make impossible.
#ifndef PHOENIX_CORE__TYPES_HPP_
#define PHOENIX_CORE__TYPES_HPP_

#include <array>
#include <cstdint>

namespace phoenix_core
{

// Actuated joints on the GO2. The policy commands all twelve.
constexpr std::size_t kNumJoints = 12;

// Proprioceptive observation width, matching ObservationBuilder in
// phoenix/sim2real/observation.py:85-97:
//   base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
//   + velocity_command(3) + joint_pos(12) + joint_vel(12) + last_action(12)
constexpr std::size_t kObsDim = 48;

static_assert(kObsDim == 3 + 3 + 3 + 3 + 3 * kNumJoints, "observation layout drifted");

using JointArray = std::array<float, kNumJoints>;

// Observation-width vector. float32, because that is the dtype the ONNX graph
// consumes and what ObservationBuilder assembles.
using Vec3 = std::array<float, 3>;

// Sensor-width vector. double, because sensor_msgs/Imu carries float64 and the
// Python evaluates in that width before narrowing once at observation
// assembly. Narrowing earlier than Python does costs about one ULP in the
// projected-gravity result, which the parity fixtures detect.
using Vec3d = std::array<double, 3>;

// Quaternion in Unitree LowState order, [w, x, y, z].
// Deliberately a distinct type from QuatXYZW: reading /lowstate directly means
// inheriting Unitree's ordering, and a bare four-float signature is how a
// wrong-order bug becomes invisible. See the audit's risk R8.
//
// Components are double to match the width the Python computes in. See Vec3d.
struct QuatWXYZ
{
  double w, x, y, z;
};

// Quaternion in ROS sensor_msgs/Imu order, (x, y, z, w).
struct QuatXYZW
{
  double x, y, z, w;
};

// Error codes for the deterministic core. Returned, never thrown: the control
// path is noexcept and the ROS adapter owns the exception boundary.
enum class Status : std::uint8_t
{
  kOk = 0,
  kNonFinite = 1,        // NaN or Inf where a finite value was required
  kDimMismatch = 2,      // a span was not the size the contract requires
  kMissingJoint = 3,     // a canonical joint name was absent from the input
  kNotInitialized = 4,   // used before initialize() succeeded
  kInferenceFailed = 5,  // the inference backend reported a failure
};

// Abort causes, mirroring the reason strings in phoenix/sim2real/gate.py.
//
// These are numeric on the hot path on purpose: the Python node builds a
// std::string reason on every abort, and formatting inside a control tick is a
// latency source with no operational benefit at the moment it happens. The
// human-readable form is produced outside the loop by reason_string().
enum class AbortReason : std::uint8_t
{
  kNone = 0,
  kMaxRuntime = 1,
  kExternalEstop = 2,
  kEstopHeartbeatStale = 3,
  kEstopPublisherMissing = 4,
  kSensorMissing = 5,
  kSensorStale = 6,
  kFirstMessageTimeout = 7,
  kNanInJointState = 8,
  kNanInImu = 9,
  kAttitude = 10,
  kUnknownSafetyGate = 11,
};

// Stable, allocation-free text for an abort cause. Safe to call off the hot
// path (logging, telemetry, post-mortem). The strings match the Python reason
// prefixes so lab logs from either runtime read the same.
const char * reason_string(AbortReason reason) noexcept;

}  // namespace phoenix_core

#endif  // PHOENIX_CORE__TYPES_HPP_
