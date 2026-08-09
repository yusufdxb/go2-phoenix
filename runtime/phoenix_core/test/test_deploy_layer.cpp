// Copyright 2026 Yusuf Guenena. MIT License.
//
// Joint mapping, observation assembly, and the Unitree CRC.
//
// The joint-map tests are the important ones. A wrong permutation produces a
// robot that walks with its legs swapped: the gait looks plausible, no gate
// fires, nothing is NaN, and the only symptom is the animal. R6 and R7.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "phoenix_core/joint_map.hpp"
#include "phoenix_core/motor_crc.hpp"
#include "phoenix_core/observation.hpp"

using namespace phoenix_core;  // NOLINT(build/namespaces) - test-local

namespace
{
JointArray iota_joints()
{
  JointArray a{};
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    a[i] = static_cast<float>(i);
  }
  return a;
}
}  // namespace

// --------------------------------------------------------------------------
// Joint permutations (R6, R7)
// --------------------------------------------------------------------------

TEST(JointMap, PolicyToMotorMatchesTheDocumentedLegGrouping)
{
  // Policy order groups by JOINT (all hips, all thighs, all calves); Unitree
  // groups by LEG with FR first. Values are joint indices so the permutation
  // is readable in the expectation.
  const auto policy = iota_joints();
  JointArray motor{};
  policy_to_motor(policy, motor);

  const float expect[kNumJoints] = {
    1, 5, 9,    // FR hip, thigh, calf
    0, 4, 8,    // FL
    3, 7, 11,   // RR
    2, 6, 10,   // RL
  };
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(motor[i], expect[i]) << "motor slot " << i;
  }
}

TEST(JointMap, MotorAndPolicyConversionsRoundTrip)
{
  const auto original = iota_joints();
  JointArray motor{}, back{};
  policy_to_motor(original, motor);
  motor_to_policy(motor, back);
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(back[i], original[i]) << "joint " << i;
  }
}

TEST(JointMap, TheCommandPermutationIsNotItsOwnInverse)
{
  // R7 stated as an executable fact. PHOENIX_FOR_MOTOR applied twice is NOT
  // the identity, so using the command-side map where the observation-side
  // remap belongs is a silent left/right swap rather than a no-op.
  const auto original = iota_joints();
  JointArray once{}, twice{};
  policy_to_motor(original, once);
  policy_to_motor(once, twice);

  bool differs = false;
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    if (twice[i] != original[i]) {differs = true;}
  }
  EXPECT_TRUE(differs) << "if this ever becomes an involution, R7 has changed";
}

TEST(JointMap, EveryMotorSlotIsWrittenExactlyOnce)
{
  // A permutation that dropped or duplicated an index would leave one joint
  // uncommanded and another double-commanded.
  std::vector<int> seen(kNumJoints, 0);
  for (std::size_t k = 0; k < kNumJoints; ++k) {
    ASSERT_LT(kPhoenixForMotor[k], kNumJoints);
    seen[kPhoenixForMotor[k]]++;
  }
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    EXPECT_EQ(seen[i], 1) << "policy index " << i << " used " << seen[i] << " times";
  }
}

TEST(JointMap, ResolvesRosOrderByNameNotPosition)
{
  // Deliberately reversed relative to policy order: a positional copy would
  // pass a naive test and swap every leg here.
  std::vector<std::string> ros;
  for (std::size_t i = kNumJoints; i-- > 0; ) {
    ros.emplace_back(kPolicyJointNames[i]);
  }

  JointIndexMap map;
  ASSERT_EQ(map.resolve(ros), Status::kOk);

  std::vector<float> values(kNumJoints);
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    values[i] = static_cast<float>(i);  // value == position in ROS order
  }
  JointArray out{};
  ASSERT_EQ(map.gather(values.data(), values.size(), out), Status::kOk);

  // policy[i] came from ROS position (11 - i).
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(out[i], static_cast<float>(kNumJoints - 1 - i));
  }
}

TEST(JointMap, ToleratesExtraJointsInTheRosMessage)
{
  std::vector<std::string> ros{"some_other_joint"};
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    ros.emplace_back(kPolicyJointNames[i]);
  }
  ros.emplace_back("yet_another");

  JointIndexMap map;
  EXPECT_EQ(map.resolve(ros), Status::kOk);
}

TEST(JointMap, RefusesWhenACanonicalJointIsMissing)
{
  // Eleven of twelve is not a partial success; continuing would command a
  // joint from whatever happened to be at that index.
  std::vector<std::string> ros;
  for (std::size_t i = 1; i < kNumJoints; ++i) {
    ros.emplace_back(kPolicyJointNames[i]);
  }
  JointIndexMap map;
  EXPECT_EQ(map.resolve(ros), Status::kMissingJoint);
  EXPECT_FALSE(map.ready());
  EXPECT_EQ(map.missing_joint(), std::string(kPolicyJointNames[0]));

  JointArray out{};
  std::vector<float> v(kNumJoints, 0.0f);
  EXPECT_EQ(map.gather(v.data(), v.size(), out), Status::kNotInitialized);
}

// --------------------------------------------------------------------------
// Observation
// --------------------------------------------------------------------------

TEST(Observation, LayoutOffsetsMatchTheTrainedContract)
{
  EXPECT_EQ(kObsDim, 48u);
  EXPECT_EQ(kObsBaseLinVel, 0u);
  EXPECT_EQ(kObsBaseAngVel, 3u);
  EXPECT_EQ(kObsProjectedGravity, 6u);
  EXPECT_EQ(kObsVelocityCommand, 9u);
  EXPECT_EQ(kObsJointPosRel, 12u);
  EXPECT_EQ(kObsJointVel, 24u);
  EXPECT_EQ(kObsLastAction, 36u);
}

TEST(Observation, JointPositionIsRelativeToDefaultNotAbsolute)
{
  // The single most consequential arithmetic in the builder. Absolute joint
  // positions would be a large, silent distribution shift.
  JointArray def{};
  def.fill(0.5f);
  ObservationBuilder b;
  ASSERT_EQ(b.initialize(def), Status::kOk);

  ObservationInputs in;
  in.joint_pos.fill(0.75f);
  ObsArray obs{};
  ASSERT_EQ(b.build(in, obs), Status::kOk);

  for (std::size_t i = 0; i < kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(obs[kObsJointPosRel + i], 0.25f);
  }
}

TEST(Observation, BaseLinearVelocityIsZeroAtDeploy)
{
  // R4. The robot has no linear-velocity estimator, so the policy is fed
  // zeros here even though training used the true value. Reproducing that is
  // the correct behaviour; "improving" it is a research decision that would
  // invalidate the shield calibration.
  JointArray def{};
  ObservationBuilder b;
  ASSERT_EQ(b.initialize(def), Status::kOk);

  ObservationInputs in;  // base_lin_vel defaults to zeros
  ObsArray obs{};
  ASSERT_EQ(b.build(in, obs), Status::kOk);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(obs[kObsBaseLinVel + i], 0.0f);
  }
}

TEST(Observation, TermsLandInTheirDeclaredSlots)
{
  JointArray def{};
  ObservationBuilder b;
  ASSERT_EQ(b.initialize(def), Status::kOk);

  ObservationInputs in;
  in.base_ang_vel = {1.0f, 2.0f, 3.0f};
  in.projected_gravity = {4.0f, 5.0f, 6.0f};
  in.velocity_command = {7.0f, 8.0f, 9.0f};
  for (std::size_t i = 0; i < kNumJoints; ++i) {
    in.joint_pos[i] = 10.0f + static_cast<float>(i);
    in.joint_vel[i] = 100.0f + static_cast<float>(i);
    in.last_action[i] = 200.0f + static_cast<float>(i);
  }
  ObsArray obs{};
  ASSERT_EQ(b.build(in, obs), Status::kOk);

  EXPECT_FLOAT_EQ(obs[kObsBaseAngVel], 1.0f);
  EXPECT_FLOAT_EQ(obs[kObsProjectedGravity], 4.0f);
  EXPECT_FLOAT_EQ(obs[kObsVelocityCommand], 7.0f);
  EXPECT_FLOAT_EQ(obs[kObsJointPosRel], 10.0f);
  EXPECT_FLOAT_EQ(obs[kObsJointVel], 100.0f);
  EXPECT_FLOAT_EQ(obs[kObsLastAction], 200.0f);
  EXPECT_FLOAT_EQ(obs[kObsLastAction + 11], 211.0f);
}

TEST(Observation, RejectsNonFiniteInputsAndUninitializedUse)
{
  ObservationInputs in;
  ObsArray obs{};

  ObservationBuilder fresh;
  EXPECT_EQ(fresh.build(in, obs), Status::kNotInitialized);

  JointArray def{};
  ObservationBuilder b;
  ASSERT_EQ(b.initialize(def), Status::kOk);
  in.projected_gravity[1] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(b.build(in, obs), Status::kNonFinite);

  JointArray bad_def{};
  bad_def[3] = std::numeric_limits<float>::infinity();
  ObservationBuilder b2;
  EXPECT_EQ(b2.initialize(bad_def), Status::kNonFinite);
}

// --------------------------------------------------------------------------
// CRC (R18, R19)
// --------------------------------------------------------------------------

TEST(MotorCrc, StructSizeIsPinned)
{
  EXPECT_EQ(kLowCmdSize, 812u);
  EXPECT_EQ((kLowCmdSize >> 2) - 1u, 202u) << "word count the CRC covers";
}

TEST(MotorCrc, IsNotStandardCrc32)
{
  // R19 as an executable fact. zlib's CRC32 of a single zero word is
  // 0x2144DF1C; this algorithm has no reflection and no final XOR, so it must
  // differ. If these ever coincide, someone has swapped in a table-driven
  // standard implementation and the firmware will reject the frame.
  const std::uint32_t word = 0;
  const std::uint32_t got = crc32_core(&word, 1);
  EXPECT_NE(got, 0x2144DF1Cu);
}

TEST(MotorCrc, RejectsAWrongSizedBuffer)
{
  std::vector<std::uint8_t> buf(kLowCmdSize - 1, 0);
  EXPECT_EQ(compute_lowcmd_crc(buf.data(), buf.size()), 0u);
  EXPECT_EQ(compute_lowcmd_crc(nullptr, kLowCmdSize), 0u);
}

TEST(MotorCrc, IsSensitiveToEveryCoveredByte)
{
  // A checksum that ignored part of the payload would still look fine on a
  // single fixture. Flip one byte at a time across the covered region.
  std::vector<std::uint8_t> buf(kLowCmdSize, 0);
  for (std::size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<std::uint8_t>(i * 7 + 3);
  }
  const std::uint32_t base = compute_lowcmd_crc(buf.data(), buf.size());

  for (std::size_t i = 0; i < kLowCmdSize - 4; i += 37) {
    auto mutated = buf;
    mutated[i] = static_cast<std::uint8_t>(mutated[i] ^ 0xFF);
    EXPECT_NE(compute_lowcmd_crc(mutated.data(), mutated.size()), base)
      << "byte " << i << " does not affect the CRC";
  }
}

TEST(MotorCrc, IgnoresTheTrailingCrcField)
{
  std::vector<std::uint8_t> buf(kLowCmdSize, 0x5A);
  const std::uint32_t base = compute_lowcmd_crc(buf.data(), buf.size());
  for (std::size_t i = kLowCmdSize - 4; i < kLowCmdSize; ++i) {
    auto mutated = buf;
    mutated[i] ^= 0xFFu;
    EXPECT_EQ(compute_lowcmd_crc(mutated.data(), mutated.size()), base)
      << "the crc field itself must not be covered";
  }
}
