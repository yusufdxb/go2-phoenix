// Copyright 2026 Yusuf Guenena. MIT License.
//
// Python/C++ golden-vector parity.
//
// Replays fixtures generated from the Python deploy path by
// scripts/generate_parity_fixtures.py and asserts the native runtime agrees.
// The Python is the oracle; a mismatch is a bug in phoenix_core.
//
// TOLERANCES ARE DECLARED HERE, BEFORE THE COMPARISON RUNS, and are derived
// from the dtype and the operation chain rather than from an observed diff:
//
//   Projected gravity  bit-exact. Six multiplies and three add/subtracts,
//                      evaluated in double on both sides and narrowed once.
//                      There is no accumulation and no reassociation
//                      opportunity, so anything other than equality is a
//                      transcription error, not rounding.
//   Roll / pitch       <= 2 ULP of float64, PLUS a zero-width ambiguous band
//                      against the attitude threshold.
//
//                      Bit-exactness was declared first and FAILED, and the
//                      cause is not the port: numpy's arctan2 and glibc's
//                      atan2 disagree by 1 ULP on the same double input
//                      (verified directly, 0x1.08ab61898531ep+0 vs
//                      0x1.08ab61898531fp+0). Transcendental functions are not
//                      required to be correctly rounded and these two
//                      implementations differ. No amount of care in this port
//                      removes that, so the tolerance is widened to the
//                      smallest value the mechanism justifies, and the risk it
//                      creates is then measured rather than assumed away:
//                      roll/pitch feed exactly one consumer, the attitude
//                      gate, so the ambiguous-band check below asserts that no
//                      fixture lies close enough to the threshold for a 2 ULP
//                      difference to change the decision.
//   Slew clip          bit-exact. The operation is a comparison and a
//                      subtraction in float32; np.clip and the C++ branch
//                      produce the same bits or the port is wrong.
//   Gate decision      exact. Outcome and abort cause are discrete.
//
// Non-finite values are compared by classification (both NaN, or both the same
// infinity) because NaN != NaN. That is the correct comparison, not a weaker
// one: the fixtures deliberately contain non-finite inputs because the
// fail-closed semantics depend on them propagating.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "phoenix_core/attitude.hpp"
#include "phoenix_core/filters.hpp"
#include "phoenix_core/gate.hpp"
#include "phoenix_core/inference.hpp"

using namespace phoenix_core;  // NOLINT(build/namespaces) - test-local

namespace
{

// Path is injected by CMake so the test does not depend on the working dir.
#ifndef PHOENIX_DEPLOY_DIR
#define PHOENIX_DEPLOY_DIR "../../deploy"
#endif

#ifndef PHOENIX_PARITY_FIXTURE
#define PHOENIX_PARITY_FIXTURE "fixtures/parity_v1.txt"
#endif

double parse_double(const std::string & tok)
{
  // strtod parses C99 hexfloat, which is what Python's float.hex() emits.
  return std::strtod(tok.c_str(), nullptr);
}

float parse_float(const std::string & tok)
{
  return static_cast<float>(parse_double(tok));
}

// Bit-exact comparison that treats NaN as equal to NaN. Returns false for a
// NaN/non-NaN pair and for differing infinities.
template<typename T>
bool same_bits(T a, T b)
{
  if (std::isnan(a) || std::isnan(b)) {
    return std::isnan(a) && std::isnan(b);
  }
  return a == b;
}

struct Fixtures
{
  std::vector<std::vector<std::string>> gravity;
  std::vector<std::vector<std::string>> slew;
  std::vector<std::vector<std::string>> gate;
  std::vector<std::vector<std::string>> onnx;
};

Fixtures load()
{
  Fixtures f;
  std::ifstream in(PHOENIX_PARITY_FIXTURE);
  // A missing fixture must fail the test, never silently pass with zero
  // records compared. An empty parity suite that reports success is worse
  // than no parity suite at all.
  EXPECT_TRUE(in.good()) << "cannot open fixture " << PHOENIX_PARITY_FIXTURE
                         << " (regenerate with scripts/generate_parity_fixtures.py)";

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    std::vector<std::string> tok;
    std::string t;
    while (ss >> t) {
      tok.push_back(t);
    }
    if (tok.empty()) {
      continue;
    }
    if (tok[0] == "G") {
      f.gravity.push_back(tok);
    } else if (tok[0] == "S") {
      f.slew.push_back(tok);
    } else if (tok[0] == "L") {
      f.gate.push_back(tok);
    } else if (tok[0] == "O") {
      f.onnx.push_back(tok);
    }
  }
  return f;
}

const Fixtures & fixtures()
{
  static const Fixtures f = load();
  return f;
}

}  // namespace

TEST(Parity, FixturesLoaded)
{
  const auto & f = fixtures();
  EXPECT_GT(f.gravity.size(), 100u);
  EXPECT_GT(f.slew.size(), 100u);
  EXPECT_GT(f.gate.size(), 1000u);
}

TEST(Parity, ProjectedGravityIsBitExact)
{
  const auto & recs = fixtures().gravity;
  ASSERT_FALSE(recs.empty());
  std::size_t compared = 0;

  for (const auto & r : recs) {
    ASSERT_EQ(r.size(), 10u);
    const QuatXYZW q{parse_double(r[1]), parse_double(r[2]), parse_double(r[3]),
      parse_double(r[4])};
    const auto g = projected_gravity_from_xyzw(q);

    for (int i = 0; i < 3; ++i) {
      const float expect = parse_float(r[static_cast<std::size_t>(5 + i)]);
      EXPECT_TRUE(same_bits(g[static_cast<std::size_t>(i)], expect))
        << "gravity[" << i << "] q=(" << q.x << "," << q.y << "," << q.z << "," << q.w
        << ") got " << g[static_cast<std::size_t>(i)] << " want " << expect;
    }
    ++compared;
  }
  EXPECT_EQ(compared, recs.size());
}

// Distance in representable doubles between a and b. 0 means bit-identical.
std::int64_t ulp_diff(double a, double b)
{
  if (std::isnan(a) || std::isnan(b)) {
    return (std::isnan(a) && std::isnan(b)) ? 0 : -1;
  }
  if (a == b) {
    return 0;
  }
  std::int64_t ia, ib;
  std::memcpy(&ia, &a, sizeof(ia));
  std::memcpy(&ib, &b, sizeof(ib));
  // Map to a monotone ordering so the subtraction is meaningful across zero.
  if (ia < 0) {ia = std::numeric_limits<std::int64_t>::min() - ia;}
  if (ib < 0) {ib = std::numeric_limits<std::int64_t>::min() - ib;}
  return ia > ib ? ia - ib : ib - ia;
}

TEST(Parity, RollPitchWithinDeclaredUlpTolerance)
{
  constexpr std::int64_t kMaxUlp = 2;
  const auto & recs = fixtures().gravity;
  ASSERT_FALSE(recs.empty());

  std::int64_t worst_roll = 0, worst_pitch = 0;
  std::size_t exact = 0;

  for (const auto & r : recs) {
    const QuatXYZW q{parse_double(r[1]), parse_double(r[2]), parse_double(r[3]),
      parse_double(r[4])};
    const auto rp = roll_pitch_from_xyzw(q);

    const auto dr = ulp_diff(rp.roll, parse_double(r[8]));
    const auto dp = ulp_diff(rp.pitch, parse_double(r[9]));
    ASSERT_GE(dr, 0) << "roll NaN mismatch";
    ASSERT_GE(dp, 0) << "pitch NaN mismatch";
    EXPECT_LE(dr, kMaxUlp) << "roll drift beyond declared tolerance";
    EXPECT_LE(dp, kMaxUlp) << "pitch drift beyond declared tolerance";

    worst_roll = std::max(worst_roll, dr);
    worst_pitch = std::max(worst_pitch, dp);
    if (dr == 0 && dp == 0) {++exact;}
  }

  // Reported, not asserted: the ULP counts are the diagnostic that
  // distinguishes benign libm rounding (1-2 ULP) from a wrong formula
  // (millions of ULP).
  std::cout << "[ parity  ] roll/pitch worst ULP: roll=" << worst_roll
            << " pitch=" << worst_pitch << ", bit-exact " << exact << "/"
            << recs.size() << std::endl;
}

TEST(Parity, AttitudeGateCannotBeFlippedByRollPitchDrift)
{
  // The consequence check for the tolerance above. roll/pitch feed exactly one
  // decision: |pitch| > 0.8 or |roll| > 0.6. If no fixture lands within the
  // declared tolerance of a threshold, then a 2 ULP disagreement between numpy
  // and libm provably cannot change the gate's verdict, and the widened
  // tolerance costs nothing in safety terms.
  //
  // Required outcome: zero ambiguous frames. If this ever fails, the drift has
  // become decision-relevant and the port needs a shared implementation of the
  // transcendental, not a wider tolerance.
  constexpr double kPitchThresh = 0.8;
  constexpr double kRollThresh = 0.6;

  std::size_t ambiguous = 0;
  for (const auto & r : fixtures().gravity) {
    const QuatXYZW q{parse_double(r[1]), parse_double(r[2]), parse_double(r[3]),
      parse_double(r[4])};
    const auto rp = roll_pitch_from_xyzw(q);
    if (std::isnan(rp.roll) || std::isnan(rp.pitch)) {
      continue;  // handled by the nan_in_imu gate, which sits above attitude
    }
    // Band = a few ULP of the threshold magnitude, generously rounded up.
    const double band = 8.0 * std::nextafter(kPitchThresh, 1.0) - 8.0 * kPitchThresh;
    if (std::fabs(std::fabs(rp.pitch) - kPitchThresh) < band ||
      std::fabs(std::fabs(rp.roll) - kRollThresh) < band)
    {
      ++ambiguous;
    }
  }
  EXPECT_EQ(ambiguous, 0u) << "roll/pitch drift is now decision-relevant";
}

TEST(Parity, SlewClipIsBitExact)
{
  const auto & recs = fixtures().slew;
  ASSERT_FALSE(recs.empty());

  for (const auto & r : recs) {
    ASSERT_EQ(r.size(), 1u + 36u);
    JointArray target, current;
    for (std::size_t i = 0; i < kNumJoints; ++i) {
      target[i] = parse_float(r[1 + i]);
      current[i] = parse_float(r[13 + i]);
    }
    const auto got = slew_clip(target, current, kMaxDeltaPerStepRad);

    for (std::size_t i = 0; i < kNumJoints; ++i) {
      const float expect = parse_float(r[25 + i]);
      EXPECT_TRUE(same_bits(got.value[i], expect))
        << "joint " << i << " target=" << target[i] << " current=" << current[i]
        << " got=" << got.value[i] << " want=" << expect;
    }
  }
}

TEST(Parity, GateDecisionsMatchExactly)
{
  const auto & recs = fixtures().gate;
  ASSERT_FALSE(recs.empty());

  const GateConfig config{120.0, 0.5, 0.2, 15.0, 0.8, 0.6};

  // Track which outcomes the fixture actually exercised. A parity suite that
  // only ever hits the nominal path proves nothing about the ladder.
  std::vector<int> outcome_hits(5, 0);
  std::vector<int> reason_hits(12, 0);

  for (const auto & r : recs) {
    ASSERT_EQ(r.size(), 13u + 12u + 12u + 4u + 3u + 2u + 2u);

    SensorSnapshot s;
    std::size_t k = 1;
    const bool latched = std::stoi(r[k++]) != 0;
    s.elapsed_s = parse_double(r[k++]);
    s.now_ns = std::stoll(r[k++]);
    s.node_started_ns = std::stoll(r[k++]);
    s.seen_estop = std::stoi(r[k++]) != 0;
    s.seen_imu = std::stoi(r[k++]) != 0;
    s.seen_joint_state = std::stoi(r[k++]) != 0;
    s.estop_last_ns = std::stoll(r[k++]);
    s.estop_value = std::stoi(r[k++]) != 0;
    s.estop_value_known = std::stoi(r[k++]) != 0;
    s.imu_last_ns = std::stoll(r[k++]);
    s.joint_state_last_ns = std::stoll(r[k++]);

    for (std::size_t i = 0; i < kNumJoints; ++i) {s.joint_pos[i] = parse_float(r[k++]);}
    for (std::size_t i = 0; i < kNumJoints; ++i) {s.joint_vel[i] = parse_float(r[k++]);}
    s.quat.x = parse_double(r[k++]);
    s.quat.y = parse_double(r[k++]);
    s.quat.z = parse_double(r[k++]);
    s.quat.w = parse_double(r[k++]);
    for (std::size_t i = 0; i < 3; ++i) {s.ang_vel[i] = parse_double(r[k++]);}
    s.roll = parse_double(r[k++]);
    s.pitch = parse_double(r[k++]);

    const int want_outcome = std::stoi(r[k++]);
    const int want_reason = std::stoi(r[k++]);

    const auto d = evaluate_gates(s, config, latched);
    EXPECT_EQ(static_cast<int>(d.outcome), want_outcome);
    EXPECT_EQ(static_cast<int>(d.reason), want_reason);

    outcome_hits[static_cast<std::size_t>(want_outcome)]++;
    reason_hits[static_cast<std::size_t>(want_reason)]++;
  }

  // Coverage assertions: every outcome and every abort cause must appear, or
  // the fixture is not exercising the ladder and its passing is meaningless.
  for (std::size_t i = 0; i < outcome_hits.size(); ++i) {
    EXPECT_GT(outcome_hits[i], 0) << "fixture never produced outcome " << i;
  }
  for (std::size_t i = 0; i < reason_hits.size(); ++i) {
    // Reason 11 (unknown_safety_gate) is a defensive fallback for a
    // (bool ok, reason) pair that returns not-ok with no reason. The Python
    // predicate cannot produce that, so the code is unreachable by
    // construction and no fixture can cover it. Excluded deliberately rather
    // than by weakening the loop.
    if (i == 11) {continue;}
    EXPECT_GT(reason_hits[i], 0) << "fixture never produced abort reason " << i;
  }
}

// --------------------------------------------------------------------------
// ONNX inference parity
// --------------------------------------------------------------------------
//
// Declared tolerance: BIT-EXACT on both action and latent.
//
// Justified by construction rather than by hope: the fixture header records
// the onnxruntime version, and both sides are pinned to the same version, the
// same CPUExecutionProvider, one intra-op and one inter-op thread, and
// sequential execution. Under those conditions the same kernels execute the
// same operations in the same order over the same weights, so the results are
// bit-identical or one of those five conditions is not actually held. A
// tolerance here would hide exactly the configuration drift worth catching.
TEST(Parity, OnnxInferenceIsBitExact)
{
  if (!has_ort_backend()) {
    GTEST_SKIP() << "built without ONNX Runtime";
  }
  const auto & recs = fixtures().onnx;
  if (recs.empty()) {
    GTEST_SKIP() << "fixture contains no inference records (generated without onnxruntime)";
  }

  constexpr std::size_t kObs = 48, kAct = 12, kLat = 384;

  ModelConfig cfg;
  cfg.model_path = std::string(PHOENIX_DEPLOY_DIR) + "/stand_v3_latent.onnx";
  cfg.obs_dim = kObs;
  cfg.action_dim = kAct;
  cfg.latent_dim = kLat;
  cfg.require_latent = true;

  auto engine = make_ort_engine();
  ASSERT_NE(engine, nullptr);
  ASSERT_EQ(engine->initialize(cfg), Status::kOk) << engine->last_error();

  std::vector<float> obs(kObs), action(kAct), latent(kLat);
  std::size_t action_mismatch = 0, latent_mismatch = 0;
  std::int64_t worst_action_ulp = 0, worst_latent_ulp = 0;

  for (const auto & r : recs) {
    ASSERT_EQ(r.size(), 1u + kObs + kAct + kLat);
    std::size_t k = 1;
    for (std::size_t i = 0; i < kObs; ++i) {obs[i] = parse_float(r[k++]);}

    const auto res = engine->infer(
      obs.data(), obs.size(), action.data(), action.size(), latent.data(), latent.size());
    ASSERT_EQ(res.status, Status::kOk);

    for (std::size_t i = 0; i < kAct; ++i) {
      const float want = parse_float(r[k++]);
      if (!same_bits(action[i], want)) {
        ++action_mismatch;
        worst_action_ulp = std::max(
          worst_action_ulp, ulp_diff(static_cast<double>(action[i]), static_cast<double>(want)));
      }
    }
    for (std::size_t i = 0; i < kLat; ++i) {
      const float want = parse_float(r[k++]);
      if (!same_bits(latent[i], want)) {
        ++latent_mismatch;
        worst_latent_ulp = std::max(
          worst_latent_ulp, ulp_diff(static_cast<double>(latent[i]), static_cast<double>(want)));
      }
    }
  }

  std::cout << "[ parity  ] onnx: " << recs.size() << " frames, "
            << action_mismatch << " action / " << latent_mismatch
            << " latent non-bit-exact elements" << std::endl;

  EXPECT_EQ(action_mismatch, 0u)
    << "worst " << worst_action_ulp << " ULP; check that both sides run the same "
    << "onnxruntime version, provider, thread count and execution mode";
  EXPECT_EQ(latent_mismatch, 0u) << "worst " << worst_latent_ulp << " ULP";
}
