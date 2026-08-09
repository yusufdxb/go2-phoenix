// Copyright 2026 Yusuf Guenena. MIT License.
// See inference.hpp.
#include "phoenix_core/inference.hpp"

#ifdef PHOENIX_WITH_ORT

#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace phoenix_core
{

namespace
{

constexpr const char * kInputName = "obs";
constexpr const char * kActionName = "action";
constexpr const char * kLatentName = "latent";

void poison(float * p, std::size_t n) noexcept
{
  if (p == nullptr) {
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = std::numeric_limits<float>::quiet_NaN();
  }
}

bool any_non_finite(const float * p, std::size_t n) noexcept
{
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(p[i])) {
      return true;
    }
  }
  return false;
}

class OrtEngine final : public InferenceEngine
{
public:
  Status initialize(const ModelConfig & config) override
  {
    error_.clear();
    ready_ = false;
    try {
      cfg_ = config;

      // Fail closed on a missing sidecar BEFORE asking ORT to load the graph.
      // The .onnx here can be a 2 KB stub whose weights live in .onnx.data,
      // and diagnosing that from ORT's own error is materially harder than
      // saying so directly (audit risk R3).
      if (!file_exists(cfg_.model_path)) {
        error_ = "model not found: " + cfg_.model_path;
        return Status::kNotInitialized;
      }
      const std::string sidecar = cfg_.model_path + ".data";
      if (file_exists(sidecar)) {
        external_data_ = true;
      }

      env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "phoenix_rt");

      Ort::SessionOptions so;
      // Deterministic by construction. See ModelConfig: thread count and
      // execution mode change reduction order and therefore change results.
      so.SetIntraOpNumThreads(cfg_.intra_op_threads);
      so.SetInterOpNumThreads(cfg_.inter_op_threads);
      so.SetExecutionMode(ORT_SEQUENTIAL);
      so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      // Path-based, so ORT resolves external data relative to the model.
      session_ = std::make_unique<Ort::Session>(*env_, cfg_.model_path.c_str(), so);

      if (const Status s = verify_contract(); s != Status::kOk) {
        return s;
      }

      // Preallocate every buffer the hot path touches. After this point
      // infer() performs no heap allocation of its own.
      obs_buf_.assign(cfg_.obs_dim, 0.0f);
      action_buf_.assign(cfg_.action_dim, 0.0f);
      latent_buf_.assign(cfg_.require_latent ? cfg_.latent_dim : 0, 0.0f);

      mem_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

      in_shape_ = {1, static_cast<std::int64_t>(cfg_.obs_dim)};
      action_shape_ = {1, static_cast<std::int64_t>(cfg_.action_dim)};
      latent_shape_ = {1, static_cast<std::int64_t>(cfg_.latent_dim)};

      // Tensors are built once over the preallocated storage and reused every
      // tick, so Run() neither allocates nor copies the outputs.
      inputs_.clear();
      inputs_.emplace_back(
        Ort::Value::CreateTensor<float>(
          *mem_info_, obs_buf_.data(), obs_buf_.size(), in_shape_.data(), in_shape_.size()));

      outputs_.clear();
      outputs_.emplace_back(
        Ort::Value::CreateTensor<float>(
          *mem_info_, action_buf_.data(), action_buf_.size(), action_shape_.data(),
          action_shape_.size()));
      if (cfg_.require_latent) {
        outputs_.emplace_back(
          Ort::Value::CreateTensor<float>(
            *mem_info_, latent_buf_.data(), latent_buf_.size(), latent_shape_.data(),
            latent_shape_.size()));
      }

      ready_ = true;
      return Status::kOk;
    } catch (const Ort::Exception & e) {
      error_ = std::string("onnxruntime: ") + e.what();
      return Status::kInferenceFailed;
    } catch (const std::exception & e) {
      error_ = e.what();
      return Status::kInferenceFailed;
    }
  }

  InferenceResult infer(
    const float * obs, std::size_t obs_len,
    float * action_out, std::size_t action_len,
    float * latent_out, std::size_t latent_len) noexcept override
  {
    InferenceResult r;

    if (!ready_) {
      r.status = Status::kNotInitialized;
      poison(action_out, action_len);
      poison(latent_out, latent_len);
      return r;
    }
    if (obs == nullptr || obs_len != cfg_.obs_dim || action_out == nullptr ||
      action_len != cfg_.action_dim ||
      (cfg_.require_latent && latent_out != nullptr && latent_len != cfg_.latent_dim))
    {
      r.status = Status::kDimMismatch;
      poison(action_out, action_len);
      poison(latent_out, latent_len);
      return r;
    }

    // A non-finite observation must never reach the graph. It would produce a
    // non-finite action that the caller then has to catch downstream, and the
    // diagnosis is much clearer here.
    if (any_non_finite(obs, obs_len)) {
      r.status = Status::kNonFinite;
      r.non_finite_output = true;
      poison(action_out, action_len);
      poison(latent_out, latent_len);
      return r;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
      for (std::size_t i = 0; i < obs_len; ++i) {
        obs_buf_[i] = obs[i];
      }

      const char * in_names[] = {kInputName};
      const char * out_names[] = {kActionName, kLatentName};

      session_->Run(
        Ort::RunOptions{nullptr}, in_names, inputs_.data(), 1, out_names, outputs_.data(),
        outputs_.size());
    } catch (const Ort::Exception & e) {
      r.status = Status::kInferenceFailed;
      r.duration_ns = elapsed_ns(t0);
      poison(action_out, action_len);
      poison(latent_out, latent_len);
      return r;
    } catch (...) {
      r.status = Status::kInferenceFailed;
      r.duration_ns = elapsed_ns(t0);
      poison(action_out, action_len);
      poison(latent_out, latent_len);
      return r;
    }
    r.duration_ns = elapsed_ns(t0);

    for (std::size_t i = 0; i < action_len; ++i) {
      action_out[i] = action_buf_[i];
    }
    if (latent_out != nullptr && !latent_buf_.empty()) {
      const std::size_t n = latent_len < latent_buf_.size() ? latent_len : latent_buf_.size();
      for (std::size_t i = 0; i < n; ++i) {
        latent_out[i] = latent_buf_[i];
      }
    }

    if (any_non_finite(action_out, action_len) ||
      (latent_out != nullptr && any_non_finite(latent_out, latent_len)))
    {
      // Report, and poison, so a caller that ignores status cannot actuate a
      // NaN-tainted command. The shield treats a non-finite score as above
      // trip, but the action path has no such convention, so it fails closed
      // here instead.
      r.status = Status::kNonFinite;
      r.non_finite_output = true;
      poison(action_out, action_len);
      poison(latent_out, latent_len);
    }
    return r;
  }

  const std::string & last_error() const noexcept override {return error_;}

private:
  static bool file_exists(const std::string & p)
  {
    std::ifstream f(p, std::ios::binary);
    return f.good();
  }

  static std::int64_t elapsed_ns(std::chrono::steady_clock::time_point t0)
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - t0).count();
  }

  // Verify the graph is the shape this runtime expects. A model whose obs
  // width or output names drifted must fail at startup, not on the robot.
  Status verify_contract()
  {
    Ort::AllocatorWithDefaultOptions al;

    if (session_->GetInputCount() != 1) {
      error_ = "expected exactly 1 input, got " + std::to_string(session_->GetInputCount());
      return Status::kDimMismatch;
    }
    const auto in_name = session_->GetInputNameAllocated(0, al);
    if (std::string(in_name.get()) != kInputName) {
      error_ = std::string("input name is '") + in_name.get() + "', expected '" + kInputName + "'";
      return Status::kDimMismatch;
    }
    if (const Status s = check_tensor(session_->GetInputTypeInfo(0), cfg_.obs_dim, "obs");
      s != Status::kOk)
    {
      return s;
    }

    const std::size_t want_outputs = cfg_.require_latent ? 2u : 1u;
    if (session_->GetOutputCount() < want_outputs) {
      error_ = "model exposes " + std::to_string(session_->GetOutputCount()) +
        " outputs, need " + std::to_string(want_outputs);
      return Status::kDimMismatch;
    }
    const auto a_name = session_->GetOutputNameAllocated(0, al);
    if (std::string(a_name.get()) != kActionName) {
      error_ = std::string("output 0 is '") + a_name.get() + "', expected '" + kActionName + "'";
      return Status::kDimMismatch;
    }
    if (const Status s = check_tensor(session_->GetOutputTypeInfo(0), cfg_.action_dim, "action");
      s != Status::kOk)
    {
      return s;
    }

    if (cfg_.require_latent) {
      const auto l_name = session_->GetOutputNameAllocated(1, al);
      if (std::string(l_name.get()) != kLatentName) {
        error_ = std::string("output 1 is '") + l_name.get() + "', expected '" + kLatentName +
          "'. The reliability shield needs a latent tap; this model has none.";
        return Status::kDimMismatch;
      }
      if (const Status s = check_tensor(session_->GetOutputTypeInfo(1), cfg_.latent_dim, "latent");
        s != Status::kOk)
      {
        return s;
      }
    }
    return Status::kOk;
  }

  // Note the deliberate two-step: GetTensorTypeAndShapeInfo() returns an
  // unowned view into the TypeInfo, so binding the TypeInfo to a named local
  // first is required. Chaining the calls reads fine and yields garbage.
  Status check_tensor(Ort::TypeInfo && info, std::size_t expect_last, const char * what)
  {
    Ort::TypeInfo held = std::move(info);
    const auto t = held.GetTensorTypeAndShapeInfo();
    if (t.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      error_ = std::string(what) + " is not float32";
      return Status::kDimMismatch;
    }
    const auto shape = t.GetShape();
    if (shape.empty()) {
      error_ = std::string(what) + " has no shape";
      return Status::kDimMismatch;
    }
    const std::int64_t last = shape.back();
    if (last != static_cast<std::int64_t>(expect_last)) {
      error_ = std::string(what) + " width is " + std::to_string(last) + ", expected " +
        std::to_string(expect_last);
      return Status::kDimMismatch;
    }
    return Status::kOk;
  }

  ModelConfig cfg_;
  bool ready_ = false;
  bool external_data_ = false;
  std::string error_;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::MemoryInfo> mem_info_;

  std::vector<float> obs_buf_, action_buf_, latent_buf_;
  std::vector<std::int64_t> in_shape_, action_shape_, latent_shape_;
  std::vector<Ort::Value> inputs_, outputs_;
};

}  // namespace

std::unique_ptr<InferenceEngine> make_ort_engine()
{
  return std::make_unique<OrtEngine>();
}

bool has_ort_backend() noexcept {return true;}

}  // namespace phoenix_core

#else  // PHOENIX_WITH_ORT

namespace phoenix_core
{
std::unique_ptr<InferenceEngine> make_ort_engine() {return nullptr;}
bool has_ort_backend() noexcept {return false;}
}  // namespace phoenix_core

#endif  // PHOENIX_WITH_ORT
