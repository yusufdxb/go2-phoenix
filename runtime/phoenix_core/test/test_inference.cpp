// Copyright 2026 Yusuf Guenena. MIT License.
//
// Inference engine tests, run against the REAL shipped artifacts in deploy/.
// A mock model would not exercise the two things most likely to break: the
// external-data sidecar (deploy/*.onnx keeps its weights in a separate
// .onnx.data file) and the exact input/output contract the exporter produced.
//
// Tests skip rather than fail when ONNX Runtime is absent, so the
// deterministic core stays testable on a machine with no model runtime. They
// do NOT skip when the runtime is present and a model is missing: that is a
// real failure.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "phoenix_core/inference.hpp"

using phoenix_core::InferenceEngine;
using phoenix_core::ModelConfig;
using phoenix_core::Status;
using phoenix_core::has_ort_backend;
using phoenix_core::kNumJoints;
using phoenix_core::kObsDim;
using phoenix_core::make_ort_engine;

namespace
{

#ifndef PHOENIX_DEPLOY_DIR
#define PHOENIX_DEPLOY_DIR "../../deploy"
#endif

std::string model(const char * name)
{
  return std::string(PHOENIX_DEPLOY_DIR) + "/" + name;
}

constexpr std::size_t kLatentDim = 384;

ModelConfig stand_config()
{
  ModelConfig c;
  c.model_path = model("stand_v3_latent.onnx");
  c.obs_dim = kObsDim;
  c.action_dim = kNumJoints;
  c.latent_dim = kLatentDim;
  c.require_latent = true;
  return c;
}

}  // namespace

#define SKIP_WITHOUT_ORT() \
  if (!has_ort_backend()) {GTEST_SKIP() << "built without ONNX Runtime";}

TEST(Inference, LoadsTheRealStandModel)
{
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  ASSERT_NE(e, nullptr);
  const auto s = e->initialize(stand_config());
  EXPECT_EQ(s, Status::kOk) << e->last_error();
}

TEST(Inference, LoadsTheRealFlatModelWithExternalData)
{
  // flat_v4_latent.onnx is a small stub whose weights live in
  // flat_v4_latent.onnx.data. This is the case a buffer-based session cannot
  // resolve (audit risk R3), so loading it successfully is the actual
  // assertion here.
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  auto c = stand_config();
  c.model_path = model("flat_v4_latent.onnx");
  EXPECT_EQ(e->initialize(c), Status::kOk) << e->last_error();
}

TEST(Inference, ProducesFiniteBoundedActionForZeroObservation)
{
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  ASSERT_EQ(e->initialize(stand_config()), Status::kOk) << e->last_error();

  std::vector<float> obs(kObsDim, 0.0f);
  std::vector<float> action(kNumJoints, 0.0f);
  std::vector<float> latent(kLatentDim, 0.0f);

  const auto r = e->infer(
    obs.data(), obs.size(), action.data(), action.size(), latent.data(), latent.size());
  EXPECT_EQ(r.status, Status::kOk);
  EXPECT_FALSE(r.non_finite_output);
  EXPECT_GT(r.duration_ns, 0);

  for (std::size_t i = 0; i < action.size(); ++i) {
    EXPECT_TRUE(std::isfinite(action[i])) << "action[" << i << "]";
  }
  bool latent_nonzero = false;
  for (float v : latent) {
    EXPECT_TRUE(std::isfinite(v));
    if (v != 0.0f) {latent_nonzero = true;}
  }
  EXPECT_TRUE(latent_nonzero) << "latent tap produced all zeros; the shield would be blind";
}

TEST(Inference, IsDeterministicAcrossRepeatedCalls)
{
  // Self-determinism is a precondition for the cross-language parity gate: a
  // runtime that disagrees with itself cannot be meaningfully compared to
  // Python. This is why threads are pinned to 1 and execution is sequential.
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  ASSERT_EQ(e->initialize(stand_config()), Status::kOk) << e->last_error();

  std::vector<float> obs(kObsDim);
  for (std::size_t i = 0; i < obs.size(); ++i) {
    obs[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.37));
  }

  std::vector<float> a1(kNumJoints), a2(kNumJoints);
  std::vector<float> l1(kLatentDim), l2(kLatentDim);
  ASSERT_EQ(
    e->infer(obs.data(), obs.size(), a1.data(), a1.size(), l1.data(), l1.size()).status,
    Status::kOk);
  for (int rep = 0; rep < 8; ++rep) {
    ASSERT_EQ(
      e->infer(obs.data(), obs.size(), a2.data(), a2.size(), l2.data(), l2.size()).status,
      Status::kOk);
    for (std::size_t i = 0; i < a1.size(); ++i) {
      EXPECT_EQ(a1[i], a2[i]) << "action[" << i << "] drifted on repeat " << rep;
    }
    for (std::size_t i = 0; i < l1.size(); ++i) {
      EXPECT_EQ(l1[i], l2[i]) << "latent[" << i << "] drifted on repeat " << rep;
    }
  }
}

TEST(Inference, TwoEnginesOnTheSameModelAgreeExactly)
{
  // Guards against session-local state leaking between ticks: a fresh engine
  // must produce identical output to a warm one on the same input.
  SKIP_WITHOUT_ORT();
  auto a = make_ort_engine();
  auto b = make_ort_engine();
  ASSERT_EQ(a->initialize(stand_config()), Status::kOk);
  ASSERT_EQ(b->initialize(stand_config()), Status::kOk);

  std::vector<float> obs(kObsDim, 0.25f);
  std::vector<float> ra(kNumJoints), rb(kNumJoints);
  std::vector<float> la(kLatentDim), lb(kLatentDim);

  // Warm one of them so they are in different internal states.
  for (int i = 0; i < 5; ++i) {
    a->infer(obs.data(), obs.size(), ra.data(), ra.size(), la.data(), la.size());
  }
  b->infer(obs.data(), obs.size(), rb.data(), rb.size(), lb.data(), lb.size());
  for (std::size_t i = 0; i < ra.size(); ++i) {
    EXPECT_EQ(ra[i], rb[i]) << "action[" << i << "]";
  }
}

// --------------------------------------------------------------------------
// Fail-closed behaviour
// --------------------------------------------------------------------------

TEST(Inference, MissingModelFailsClosed)
{
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  auto c = stand_config();
  c.model_path = model("definitely_not_a_model.onnx");
  EXPECT_NE(e->initialize(c), Status::kOk);
  EXPECT_FALSE(e->last_error().empty());
}

TEST(Inference, CorruptModelFailsClosed)
{
  SKIP_WITHOUT_ORT();
  // A real file that is not a valid graph. ORT throws; the engine must convert
  // that into a status rather than letting it escape into a control loop.
  auto e = make_ort_engine();
  auto c = stand_config();
  c.model_path = model("shield_stand_v3.npz");  // exists, not an ONNX graph
  EXPECT_NE(e->initialize(c), Status::kOk);
  EXPECT_FALSE(e->last_error().empty());
}

TEST(Inference, WrongExpectedObsWidthIsRejectedAtInit)
{
  // The single most valuable startup check: a model whose observation width
  // drifted from the runtime's contract must not reach the robot.
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  auto c = stand_config();
  c.obs_dim = 235;  // the obs_pad_zeros trap from the audit
  EXPECT_EQ(e->initialize(c), Status::kDimMismatch);
  EXPECT_NE(e->last_error().find("48"), std::string::npos) << e->last_error();
}

TEST(Inference, WrongExpectedLatentWidthIsRejectedAtInit)
{
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  auto c = stand_config();
  c.latent_dim = 128;
  EXPECT_EQ(e->initialize(c), Status::kDimMismatch);
}

TEST(Inference, InferBeforeInitializeFailsClosedAndPoisonsOutput)
{
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  std::vector<float> obs(kObsDim, 0.0f);
  std::vector<float> action(kNumJoints, 7.0f);  // sentinel: must not survive

  const auto r = e->infer(obs.data(), obs.size(), action.data(), action.size(), nullptr, 0);
  EXPECT_EQ(r.status, Status::kNotInitialized);
  for (float v : action) {
    EXPECT_TRUE(std::isnan(v)) << "stale sentinel survived a failed inference";
  }
}

TEST(Inference, WrongObsLengthAtInferTimeFailsClosed)
{
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  ASSERT_EQ(e->initialize(stand_config()), Status::kOk);

  std::vector<float> obs(kObsDim - 1, 0.0f);
  std::vector<float> action(kNumJoints, 7.0f);
  const auto r = e->infer(obs.data(), obs.size(), action.data(), action.size(), nullptr, 0);
  EXPECT_EQ(r.status, Status::kDimMismatch);
  for (float v : action) {EXPECT_TRUE(std::isnan(v));}
}

TEST(Inference, NonFiniteObservationIsRejectedBeforeTheGraph)
{
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  ASSERT_EQ(e->initialize(stand_config()), Status::kOk);

  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity()})
  {
    std::vector<float> obs(kObsDim, 0.0f);
    obs[17] = bad;
    std::vector<float> action(kNumJoints, 7.0f);
    std::vector<float> latent(kLatentDim, 7.0f);

    const auto r = e->infer(
      obs.data(), obs.size(), action.data(), action.size(), latent.data(), latent.size());
    EXPECT_EQ(r.status, Status::kNonFinite);
    EXPECT_TRUE(r.non_finite_output);
    for (float v : action) {
      EXPECT_TRUE(std::isnan(v)) << "a corrupt observation must not yield a usable action";
    }
  }
}

TEST(Inference, EngineRecoversAfterARejectedCall)
{
  // A rejected tick must not wedge the session. The runtime latches aborts at
  // a higher level; the engine itself has to stay usable.
  SKIP_WITHOUT_ORT();
  auto e = make_ort_engine();
  ASSERT_EQ(e->initialize(stand_config()), Status::kOk);

  std::vector<float> bad(kObsDim, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> good(kObsDim, 0.0f);
  std::vector<float> action(kNumJoints);
  std::vector<float> latent(kLatentDim);

  EXPECT_EQ(
    e->infer(bad.data(), bad.size(), action.data(), action.size(), latent.data(), latent.size())
    .status, Status::kNonFinite);
  const auto r = e->infer(
    good.data(), good.size(), action.data(), action.size(), latent.data(), latent.size());
  EXPECT_EQ(r.status, Status::kOk);
  for (float v : action) {EXPECT_TRUE(std::isfinite(v));}
}
