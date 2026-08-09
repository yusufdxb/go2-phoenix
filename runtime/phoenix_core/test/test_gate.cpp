// Copyright 2026 Yusuf Guenena. MIT License.
//
// Mirror of tests/test_gate_ladder.py. Every test here has a counterpart there
// with the same name and the same expectation; the two suites are the pinning
// mechanism between the Python oracle and the native port.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "phoenix_core/gate.hpp"

using phoenix_core::AbortReason;
using phoenix_core::GateConfig;
using phoenix_core::GateDecision;
using phoenix_core::Outcome;
using phoenix_core::SensorSnapshot;
using phoenix_core::evaluate_gates;

namespace
{

constexpr std::int64_t kSec = 1000000000;
constexpr std::int64_t kNow = 10 * kSec;

const GateConfig kConfig{
  /*max_runtime_s=*/120.0,
  /*estop_timeout_s=*/0.5,
  /*sensor_timeout_s=*/0.2,
  /*first_message_timeout_s=*/15.0,
  /*pitch_rad=*/0.8,
  /*roll_rad=*/0.6,
};

// Nominal, everything-healthy snapshot. Tests mutate one field.
SensorSnapshot nominal()
{
  SensorSnapshot s;
  s.now_ns = kNow;
  s.elapsed_s = 10.0;
  s.node_started_ns = 0;
  s.seen_estop = true;
  s.seen_imu = true;
  s.seen_joint_state = true;
  s.estop_last_ns = kNow;
  s.estop_value = false;
  s.estop_value_known = true;
  s.imu_last_ns = kNow;
  s.joint_state_last_ns = kNow;
  s.joint_pos.fill(0.0f);
  s.joint_vel.fill(0.0f);
  s.quat = {0.0f, 0.0f, 0.0f, 1.0f};
  s.ang_vel = {0.0f, 0.0f, 0.0f};
  s.roll = 0.0;
  s.pitch = 0.0;
  return s;
}

GateDecision decide(const SensorSnapshot & s, bool latched = false)
{
  return evaluate_gates(s, kConfig, latched);
}

const float kNan = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

}  // namespace

// --------------------------------------------------------------------------
// Nominal and latched
// --------------------------------------------------------------------------

TEST(Gate, NominalRunsPolicy)
{
  const auto d = decide(nominal());
  EXPECT_EQ(d.outcome, Outcome::kRunPolicy);
  EXPECT_EQ(d.reason, AbortReason::kNone);
  EXPECT_FALSE(d.latches());
  EXPECT_FALSE(d.publishes_default());
}

TEST(Gate, AlreadyLatchedIsPermanentlySilent)
{
  const auto d = decide(nominal(), true);
  EXPECT_EQ(d.outcome, Outcome::kSilent);
  // Post-abort rebroadcast fought real posture and caused a Jetson brownout.
  EXPECT_FALSE(d.publishes_default());
}

TEST(Gate, AlreadyLatchedWinsOverEveryOtherFault)
{
  auto s = nominal();
  s.estop_value = true;
  s.joint_pos.fill(kNan);
  s.pitch = 3.0;
  EXPECT_EQ(decide(s, true).outcome, Outcome::kSilent);
}

// --------------------------------------------------------------------------
// Runtime watchdog
// --------------------------------------------------------------------------

TEST(Gate, MaxRuntimeLatchesWithoutPublishing)
{
  auto s = nominal();
  s.elapsed_s = 120.1;
  const auto d = decide(s);
  EXPECT_EQ(d.outcome, Outcome::kLatchSilent);
  EXPECT_EQ(d.reason, AbortReason::kMaxRuntime);
  EXPECT_TRUE(d.latches());
  EXPECT_FALSE(d.publishes_default());
}

TEST(Gate, MaxRuntimeBoundaryIsStrictGreaterThan)
{
  auto s = nominal();
  s.elapsed_s = 120.0;
  EXPECT_EQ(decide(s).outcome, Outcome::kRunPolicy);
  s.elapsed_s = 120.001;
  EXPECT_EQ(decide(s).outcome, Outcome::kLatchSilent);
}

TEST(Gate, MaxRuntimeDoesNotRelatchWhenAlreadyLatched)
{
  auto s = nominal();
  s.elapsed_s = 999.0;
  const auto d = decide(s, true);
  EXPECT_EQ(d.outcome, Outcome::kSilent);
  // First cause must be preserved, not overwritten.
  EXPECT_EQ(d.reason, AbortReason::kNone);
}

TEST(Gate, MaxRuntimeOutranksAllGatesBelowIt)
{
  auto s = nominal();
  s.elapsed_s = 999.0;
  s.estop_value = true;
  s.pitch = 3.0;
  EXPECT_EQ(decide(s).reason, AbortReason::kMaxRuntime);
}

// --------------------------------------------------------------------------
// Startup gate
// --------------------------------------------------------------------------

TEST(Gate, StartupWaitingPublishesDefaultWithoutLatching)
{
  for (int which = 0; which < 3; ++which) {
    auto s = nominal();
    if (which == 0) {s.seen_estop = false;}
    if (which == 1) {s.seen_imu = false;}
    if (which == 2) {s.seen_joint_state = false;}
    const auto d = decide(s);
    EXPECT_EQ(d.outcome, Outcome::kPublishDefault) << "which=" << which;
    EXPECT_FALSE(d.latches());
    EXPECT_TRUE(d.publishes_default());
  }
}

TEST(Gate, StartupTimeoutLatchesWithSpecificMissingTopic)
{
  auto s = nominal();
  s.seen_imu = false;
  s.node_started_ns = kNow - 16 * kSec;
  const auto d = decide(s);
  EXPECT_EQ(d.outcome, Outcome::kLatchAndPublishDefault);
  EXPECT_EQ(d.reason, AbortReason::kFirstMessageTimeout);
  EXPECT_TRUE(d.missing_imu);
  EXPECT_FALSE(d.missing_estop);
  EXPECT_FALSE(d.missing_joint_state);
}

TEST(Gate, StartupOutranksSensorAndAttitudeGates)
{
  auto s = nominal();
  s.seen_imu = false;
  s.estop_value = true;
  s.pitch = 3.0;
  EXPECT_EQ(decide(s).outcome, Outcome::kPublishDefault);
}

// --------------------------------------------------------------------------
// Estop chain and sensor freshness
// --------------------------------------------------------------------------

TEST(Gate, ExternalEstopLatches)
{
  auto s = nominal();
  s.estop_value = true;
  const auto d = decide(s);
  EXPECT_EQ(d.outcome, Outcome::kLatchAndPublishDefault);
  EXPECT_EQ(d.reason, AbortReason::kExternalEstop);
}

TEST(Gate, StaleEstopHeartbeatLatches)
{
  auto s = nominal();
  s.estop_last_ns = kNow - static_cast<std::int64_t>(0.6 * kSec);
  EXPECT_EQ(decide(s).reason, AbortReason::kEstopHeartbeatStale);
}

TEST(Gate, MissingEstopPublisherLatches)
{
  auto s = nominal();
  s.estop_last_ns = SensorSnapshot::kNeverSeen;
  s.estop_value_known = false;
  EXPECT_EQ(decide(s).reason, AbortReason::kEstopPublisherMissing);
}

TEST(Gate, StaleSensorLatches)
{
  for (int which = 0; which < 2; ++which) {
    auto s = nominal();
    const auto stale = kNow - static_cast<std::int64_t>(0.3 * kSec);
    if (which == 0) {s.imu_last_ns = stale;} else {s.joint_state_last_ns = stale;}
    const auto d = decide(s);
    EXPECT_EQ(d.outcome, Outcome::kLatchAndPublishDefault);
    EXPECT_EQ(d.reason, AbortReason::kSensorStale) << "which=" << which;
  }
}

TEST(Gate, EstopOutranksSensorStaleness)
{
  auto s = nominal();
  s.estop_value = true;
  s.imu_last_ns = kNow - 5 * kSec;
  EXPECT_EQ(decide(s).reason, AbortReason::kExternalEstop);
}

// --------------------------------------------------------------------------
// Non-finite sensor data
// --------------------------------------------------------------------------

TEST(Gate, NonFiniteJointStateLatches)
{
  for (float bad : {kNan, kInf, -kInf}) {
    for (int field = 0; field < 2; ++field) {
      auto s = nominal();
      if (field == 0) {s.joint_pos[7] = bad;} else {s.joint_vel[7] = bad;}
      const auto d = decide(s);
      EXPECT_EQ(d.outcome, Outcome::kLatchAndPublishDefault);
      EXPECT_EQ(d.reason, AbortReason::kNanInJointState);
    }
  }
}

TEST(Gate, NonFiniteQuaternionLatches)
{
  // Regression for L2. Before the IMU finiteness gate existed a corrupt
  // quaternion produced NaN roll/pitch, sailed through the attitude gate
  // (fabs(NaN) > 0.8 is false), and poisoned the observation.
  for (float bad : {kNan, kInf, -kInf}) {
    for (int idx = 0; idx < 4; ++idx) {
      auto s = nominal();
      double * q[4] = {&s.quat.x, &s.quat.y, &s.quat.z, &s.quat.w};
      *q[idx] = bad;
      s.roll = std::nan("");
      s.pitch = std::nan("");
      const auto d = decide(s);
      EXPECT_EQ(d.outcome, Outcome::kLatchAndPublishDefault);
      EXPECT_EQ(d.reason, AbortReason::kNanInImu) << "idx=" << idx;
    }
  }
}

TEST(Gate, NonFiniteAngularVelocityLatches)
{
  for (int idx = 0; idx < 3; ++idx) {
    auto s = nominal();
    s.ang_vel[static_cast<std::size_t>(idx)] = kNan;
    EXPECT_EQ(decide(s).reason, AbortReason::kNanInImu);
  }
}

TEST(Gate, NanAttitudeCannotSilentlyPassTheAttitudeGate)
{
  auto s = nominal();
  s.quat.x = kNan;
  s.roll = std::nan("");
  s.pitch = std::nan("");
  EXPECT_NE(decide(s).outcome, Outcome::kRunPolicy);
}

TEST(Gate, JointStateValidityOutranksImuValidity)
{
  auto s = nominal();
  s.joint_pos.fill(kNan);
  s.quat.x = kNan;
  EXPECT_EQ(decide(s).reason, AbortReason::kNanInJointState);
}

// --------------------------------------------------------------------------
// Attitude
// --------------------------------------------------------------------------

TEST(Gate, PitchAndRollAbort)
{
  auto s = nominal();
  s.pitch = 0.81;
  EXPECT_EQ(decide(s).reason, AbortReason::kAttitude);
  s = nominal();
  s.roll = -0.61;
  EXPECT_EQ(decide(s).reason, AbortReason::kAttitude);
}

TEST(Gate, AttitudeThresholdsAreAsymmetricAndExclusive)
{
  // roll 0.6 and pitch 0.8 are deliberately different; a symmetric port would
  // silently loosen roll or tighten pitch.
  auto s = nominal();
  s.pitch = 0.8;
  EXPECT_EQ(decide(s).outcome, Outcome::kRunPolicy);
  s = nominal();
  s.roll = 0.6;
  EXPECT_EQ(decide(s).outcome, Outcome::kRunPolicy);
  s = nominal();
  s.roll = 0.61;
  EXPECT_EQ(decide(s).outcome, Outcome::kLatchAndPublishDefault);
  s = nominal();
  s.pitch = 0.79;
  s.roll = 0.59;
  EXPECT_EQ(decide(s).outcome, Outcome::kRunPolicy);
}

TEST(Gate, ImuValidityOutranksAttitude)
{
  auto s = nominal();
  s.ang_vel[0] = kNan;
  s.pitch = 3.0;
  EXPECT_EQ(decide(s).reason, AbortReason::kNanInImu);
}

// --------------------------------------------------------------------------
// Full-ladder precedence
// --------------------------------------------------------------------------

TEST(Gate, FullPrecedenceOrder)
{
  // Every gate firing at once; they must resolve in ladder order as each
  // higher cause is removed.
  auto s = nominal();
  s.elapsed_s = 999.0;
  s.seen_imu = false;
  s.estop_value = true;
  s.imu_last_ns = kNow - 5 * kSec;
  s.joint_pos.fill(kNan);
  s.quat = {kNan, kNan, kNan, kNan};
  s.pitch = 3.0;
  s.roll = 3.0;

  EXPECT_EQ(decide(s).reason, AbortReason::kMaxRuntime);

  s.elapsed_s = 10.0;
  EXPECT_EQ(decide(s).outcome, Outcome::kPublishDefault);

  s.seen_imu = true;
  EXPECT_EQ(decide(s).reason, AbortReason::kExternalEstop);

  s.estop_value = false;
  EXPECT_EQ(decide(s).reason, AbortReason::kSensorStale);

  s.imu_last_ns = kNow;
  EXPECT_EQ(decide(s).reason, AbortReason::kNanInJointState);

  s.joint_pos.fill(0.0f);
  EXPECT_EQ(decide(s).reason, AbortReason::kNanInImu);

  s.quat = {0.0f, 0.0f, 0.0f, 1.0f};
  EXPECT_EQ(decide(s).reason, AbortReason::kAttitude);

  s.pitch = 0.0;
  s.roll = 0.0;
  EXPECT_EQ(decide(s).outcome, Outcome::kRunPolicy);
}

TEST(Gate, OnlyLatchingOutcomesCarryAReason)
{
  auto healthy = nominal();
  auto waiting = nominal();
  waiting.seen_imu = false;

  for (const auto & pair : {std::make_pair(healthy, false), std::make_pair(healthy, true),
      std::make_pair(waiting, false)})
  {
    const auto d = decide(pair.first, pair.second);
    EXPECT_EQ(d.reason != AbortReason::kNone, d.latches());
  }
}
