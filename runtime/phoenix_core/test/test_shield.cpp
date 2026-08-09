// Copyright 2026 Yusuf Guenena. MIT License.
//
// Shield unit tests. R17 comes first because it is the audit's nominated
// "single most dangerous naive-port bug": a non-finite score must count as
// ABOVE TRIP and never toward CLEAR, and the obvious C++ transcription
// (`score > trip`) is false for NaN, which inverts that exactly when the
// monitor is most suspicious.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "phoenix_core/shield.hpp"

using phoenix_core::ArbiterConfig;
using phoenix_core::DeployMonitor;
using phoenix_core::DeployShield;
using phoenix_core::ShieldState;
using phoenix_core::SimplexArbiter;
using phoenix_core::Status;

namespace
{

const double kNan = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

ArbiterConfig cfg()
{
  ArbiterConfig c;
  c.trip_threshold = 10.0;
  c.clear_threshold = 2.0;
  c.trip_persistence = 3;
  c.clear_persistence = 4;
  c.handoff_ticks = 5;
  c.recover_ticks = 6;
  c.min_fallback_ticks = 4;
  c.latch = false;
  return c;
}

// Drive the arbiter to FALLBACK and return it there.
void drive_to_fallback(SimplexArbiter & a)
{
  for (int i = 0; i < 3; ++i) {a.update(50.0);}      // trip persistence
  for (int i = 0; i < 5; ++i) {a.update(50.0);}      // handoff ramp
}

}  // namespace

// --------------------------------------------------------------------------
// R17: the non-finite convention
// --------------------------------------------------------------------------

TEST(ArbiterR17, NonFiniteScoreCountsAsAboveTripAndTrips)
{
  for (double bad : {kNan, kInf, -kInf}) {
    SimplexArbiter a(cfg());
    ASSERT_EQ(a.state(), ShieldState::kNominal);
    a.update(bad);
    a.update(bad);
    const auto out = a.update(bad);
    EXPECT_NE(out.state, ShieldState::kNominal)
      << "a non-finite score must trip the shield, not read as healthy";
  }
}

TEST(ArbiterR17, NaNMustNotBeTreatedAsBelowThreshold)
{
  // The precise failure mode. `NaN > trip` is false, so a port that writes
  // only that comparison leaves the arbiter in NOMINAL forever while the
  // monitor emits garbage.
  SimplexArbiter a(cfg());
  for (int i = 0; i < 20; ++i) {a.update(kNan);}
  EXPECT_NE(a.state(), ShieldState::kNominal);
  EXPECT_GT(a.blend(), 0.0);
}

TEST(ArbiterR17, NonFiniteScoreNeverContributesToClearing)
{
  // The symmetric half, which a port can get wrong independently: NaN must
  // not count toward release either. `NaN < clear` is also false, so this
  // happens to be right by accident in C++, and the test pins it so a later
  // "simplification" cannot break it.
  SimplexArbiter a(cfg());
  drive_to_fallback(a);
  ASSERT_EQ(a.state(), ShieldState::kFallback);

  for (int i = 0; i < 100; ++i) {a.update(kNan);}
  EXPECT_EQ(a.state(), ShieldState::kFallback) << "NaN must never release the shield";
  EXPECT_DOUBLE_EQ(a.blend(), 1.0);
}

TEST(ArbiterR17, NonFiniteAbortsAnInProgressRecovery)
{
  SimplexArbiter a(cfg());
  drive_to_fallback(a);
  for (int i = 0; i < 4; ++i) {a.update(50.0);}   // satisfy min_fallback_ticks
  for (int i = 0; i < 4; ++i) {a.update(0.5);}    // clear persistence
  ASSERT_EQ(a.state(), ShieldState::kRecovering);

  for (int i = 0; i < 3; ++i) {a.update(kNan);}
  EXPECT_EQ(a.state(), ShieldState::kFallback) << "safety dominates recovery";
  EXPECT_DOUBLE_EQ(a.blend(), 1.0);
}

TEST(MonitorR17, NonFiniteLatentScoresInfiniteNotZero)
{
  DeployMonitor m;
  std::vector<float> mean(4, 0.0f);
  std::vector<float> w(16, 0.0f);
  for (int i = 0; i < 4; ++i) {w[static_cast<std::size_t>(i * 4 + i)] = 1.0f;}
  ASSERT_EQ(m.initialize(mean.data(), w.data(), 4), Status::kOk);

  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity()})
  {
    std::vector<float> latent(4, 0.0f);
    latent[2] = bad;
    const double s = m.score_one(latent.data(), latent.size());
    EXPECT_TRUE(std::isinf(s) && s > 0) << "a garbage frame must push toward fallback";
  }
}

TEST(MonitorR17, WrongDimensionScoresInfiniteRatherThanZero)
{
  DeployMonitor m;
  std::vector<float> mean(4, 0.0f), w(16, 0.0f);
  ASSERT_EQ(m.initialize(mean.data(), w.data(), 4), Status::kOk);
  std::vector<float> latent(3, 0.0f);
  EXPECT_TRUE(std::isinf(m.score_one(latent.data(), latent.size())));
}

// --------------------------------------------------------------------------
// State machine
// --------------------------------------------------------------------------

TEST(Arbiter, StaysNominalBelowTrip)
{
  SimplexArbiter a(cfg());
  for (int i = 0; i < 50; ++i) {a.update(0.0);}
  EXPECT_EQ(a.state(), ShieldState::kNominal);
  EXPECT_DOUBLE_EQ(a.blend(), 0.0);
}

TEST(Arbiter, RequiresPersistenceToTrip)
{
  SimplexArbiter a(cfg());
  a.update(50.0);
  a.update(50.0);
  EXPECT_EQ(a.state(), ShieldState::kNominal) << "two ticks is below trip_persistence=3";
  a.update(50.0);
  EXPECT_EQ(a.state(), ShieldState::kHandoff);
}

TEST(Arbiter, AnInterruptedStreakResetsTheCounter)
{
  SimplexArbiter a(cfg());
  a.update(50.0);
  a.update(50.0);
  a.update(0.0);   // breaks the streak
  a.update(50.0);
  a.update(50.0);
  EXPECT_EQ(a.state(), ShieldState::kNominal);
}

TEST(Arbiter, HandoffRampIsMonotoneAndBounded)
{
  SimplexArbiter a(cfg());
  for (int i = 0; i < 3; ++i) {a.update(50.0);}
  ASSERT_EQ(a.state(), ShieldState::kHandoff);

  double prev = a.blend();
  for (int i = 0; i < 5; ++i) {
    const auto out = a.update(50.0);
    EXPECT_GE(out.blend, prev);
    EXPECT_GE(out.blend, 0.0);
    EXPECT_LE(out.blend, 1.0);
    prev = out.blend;
  }
  EXPECT_EQ(a.state(), ShieldState::kFallback);
  EXPECT_DOUBLE_EQ(a.blend(), 1.0);
}

TEST(Arbiter, HandoffIgnoresTheScoreSoItCannotChatter)
{
  SimplexArbiter a(cfg());
  for (int i = 0; i < 3; ++i) {a.update(50.0);}
  ASSERT_EQ(a.state(), ShieldState::kHandoff);
  // Score drops to healthy mid-ramp; the arbiter must commit anyway.
  for (int i = 0; i < 5; ++i) {a.update(0.0);}
  EXPECT_EQ(a.state(), ShieldState::kFallback);
}

TEST(Arbiter, DwellIsRequiredBeforeRelease)
{
  SimplexArbiter a(cfg());
  drive_to_fallback(a);
  // Healthy immediately, but min_fallback_ticks=4 has not elapsed.
  a.update(0.0);
  a.update(0.0);
  EXPECT_EQ(a.state(), ShieldState::kFallback);
}

TEST(Arbiter, RecoversThroughRampBackToNominal)
{
  SimplexArbiter a(cfg());
  drive_to_fallback(a);
  for (int i = 0; i < 4; ++i) {a.update(50.0);}
  for (int i = 0; i < 4; ++i) {a.update(0.0);}
  ASSERT_EQ(a.state(), ShieldState::kRecovering);

  double prev = a.blend();
  for (int i = 0; i < 6; ++i) {
    const auto out = a.update(0.0);
    EXPECT_LE(out.blend, prev);
    prev = out.blend;
  }
  EXPECT_EQ(a.state(), ShieldState::kNominal);
  EXPECT_DOUBLE_EQ(a.blend(), 0.0);
}

TEST(Arbiter, LatchNeverReleases)
{
  auto c = cfg();
  c.latch = true;
  SimplexArbiter a(c);
  drive_to_fallback(a);
  for (int i = 0; i < 500; ++i) {a.update(0.0);}
  EXPECT_EQ(a.state(), ShieldState::kFallback);
  EXPECT_DOUBLE_EQ(a.blend(), 1.0);
}

TEST(Arbiter, BlendStaysInUnitInterval)
{
  SimplexArbiter a(cfg());
  const double scores[] = {0.0, 50.0, kNan, 1.0, 11.0, kInf, 2.5, 0.1};
  for (int rep = 0; rep < 200; ++rep) {
    const auto out = a.update(scores[static_cast<std::size_t>(rep) % 8]);
    ASSERT_GE(out.blend, 0.0);
    ASSERT_LE(out.blend, 1.0);
  }
}

TEST(Arbiter, RejectsInvertedHysteresis)
{
  auto c = cfg();
  c.clear_threshold = c.trip_threshold;
  EXPECT_FALSE(c.valid());
  c.clear_threshold = c.trip_threshold + 1.0;
  EXPECT_FALSE(c.valid());
}

TEST(Arbiter, StateWireEncodingIsPinned)
{
  // The Python telemetry encodes state as an index into a literal tuple,
  // independent of its enum order (risk R22). If these drift, lab logs from
  // the two runtimes silently disagree about what happened.
  EXPECT_EQ(static_cast<int>(ShieldState::kNominal), 0);
  EXPECT_EQ(static_cast<int>(ShieldState::kHandoff), 1);
  EXPECT_EQ(static_cast<int>(ShieldState::kFallback), 2);
  EXPECT_EQ(static_cast<int>(ShieldState::kRecovering), 3);
}

// --------------------------------------------------------------------------
// Monitor
// --------------------------------------------------------------------------

TEST(Monitor, IdentityWhitenerGivesSquaredEuclideanDistance)
{
  DeployMonitor m;
  std::vector<float> mean{1.0f, 2.0f, 3.0f};
  std::vector<float> w{1, 0, 0, 0, 1, 0, 0, 0, 1};
  ASSERT_EQ(m.initialize(mean.data(), w.data(), 3), Status::kOk);

  std::vector<float> latent{2.0f, 4.0f, 6.0f};  // diff = (1,2,3)
  EXPECT_NEAR(m.score_one(latent.data(), latent.size()), 14.0, 1e-5);
}

TEST(Monitor, ScoreIsZeroAtTheMean)
{
  DeployMonitor m;
  std::vector<float> mean{1.0f, -2.0f};
  std::vector<float> w{1, 0, 0, 1};
  ASSERT_EQ(m.initialize(mean.data(), w.data(), 2), Status::kOk);
  EXPECT_DOUBLE_EQ(m.score_one(mean.data(), mean.size()), 0.0);
}

TEST(Monitor, RejectsNonFiniteConstants)
{
  // A monitor fitted on garbage cannot fail safe later, so it must not load.
  DeployMonitor m;
  std::vector<float> mean{1.0f, std::numeric_limits<float>::quiet_NaN()};
  std::vector<float> w{1, 0, 0, 1};
  EXPECT_EQ(m.initialize(mean.data(), w.data(), 2), Status::kNonFinite);

  std::vector<float> mean2{1.0f, 2.0f};
  std::vector<float> w2{1, 0, 0, std::numeric_limits<float>::infinity()};
  EXPECT_EQ(m.initialize(mean2.data(), w2.data(), 2), Status::kNonFinite);
}

// --------------------------------------------------------------------------
// DeployShield: arming (R16) and no temporal filtering (R15)
// --------------------------------------------------------------------------

namespace
{
DeployShield make_shield(int arming)
{
  DeployShield s;
  std::vector<float> mean(2, 0.0f);
  std::vector<float> w{1, 0, 0, 1};
  EXPECT_EQ(s.initialize(mean.data(), w.data(), 2, cfg(), arming), Status::kOk);
  return s;
}
}  // namespace

TEST(ShieldR16, ArmingWindowIsCountedInTicksNotTime)
{
  auto s = make_shield(15);
  std::vector<float> hot{100.0f, 100.0f};  // wildly out of distribution

  for (int i = 0; i < 15; ++i) {
    const auto d = s.step(hot.data(), hot.size());
    EXPECT_FALSE(d.armed) << "tick " << i;
    EXPECT_EQ(d.state, ShieldState::kNominal);
    EXPECT_DOUBLE_EQ(d.blend, 0.0) << "the arbiter must not advance while disarmed";
    EXPECT_GT(d.raw_score, 0.0) << "the true score is still reported while disarmed";
  }
  EXPECT_TRUE(s.armed()) << "exactly 15 disarmed ticks, then armed";
}

TEST(ShieldR16, ArbiterDoesNotAdvanceDuringArming)
{
  // If the disarmed path advanced the arbiter, the shield would trip the
  // instant it armed rather than after trip_persistence fresh ticks.
  auto s = make_shield(5);
  std::vector<float> hot{100.0f, 100.0f};
  for (int i = 0; i < 5; ++i) {s.step(hot.data(), hot.size());}
  ASSERT_TRUE(s.armed());

  const auto first = s.step(hot.data(), hot.size());
  EXPECT_EQ(first.state, ShieldState::kNominal)
    << "trip persistence must be counted from the first ARMED tick";
}

TEST(ShieldR15, FilteredScoreEqualsRawScore)
{
  // The deploy shield applies no temporal filtering; the offline ShieldRuntime
  // does. Porting the offline one would move the calibrated operating point.
  auto s = make_shield(0);
  for (int i = 0; i < 30; ++i) {
    std::vector<float> latent{static_cast<float>(i), static_cast<float>(-i)};
    const auto d = s.step(latent.data(), latent.size());
    EXPECT_DOUBLE_EQ(d.filtered_score, d.raw_score) << "no EWMA at deploy";
  }
}

TEST(Shield, ResetClearsArmingAndArbiterState)
{
  auto s = make_shield(2);
  std::vector<float> hot{100.0f, 100.0f};
  for (int i = 0; i < 20; ++i) {s.step(hot.data(), hot.size());}
  ASSERT_NE(s.step(hot.data(), hot.size()).state, ShieldState::kNominal);

  s.reset();
  EXPECT_FALSE(s.armed());
  const auto d = s.step(hot.data(), hot.size());
  EXPECT_EQ(d.state, ShieldState::kNominal);
  EXPECT_DOUBLE_EQ(d.blend, 0.0);
}

TEST(Shield, UninitializedShieldIsInertRatherThanFabricating)
{
  DeployShield s;
  std::vector<float> latent{1.0f, 2.0f};
  const auto d = s.step(latent.data(), latent.size());
  EXPECT_EQ(d.state, ShieldState::kNominal);
  EXPECT_DOUBLE_EQ(d.blend, 0.0);
  EXPECT_FALSE(d.armed);
}

TEST(Shield, RejectsInvalidConfigAtInit)
{
  DeployShield s;
  std::vector<float> mean(2, 0.0f);
  std::vector<float> w{1, 0, 0, 1};
  auto bad = cfg();
  bad.clear_threshold = bad.trip_threshold + 1.0;
  EXPECT_NE(s.initialize(mean.data(), w.data(), 2, bad, 0), Status::kOk);
  EXPECT_NE(s.initialize(mean.data(), w.data(), 2, cfg(), -1), Status::kOk);
}
