// Copyright 2026 Yusuf Guenena. MIT License.
//
// Policy inference abstraction and its ONNX Runtime backend.
//
// Design constraints, each traceable to something the audit found:
//
// * The session is built once in initialize() and never rebuilt per tick.
// * Input and output metadata are verified against the expected contract at
//   initialize() time, so a model whose observation width or output names
//   drifted fails at startup rather than producing plausible garbage on the
//   robot.
// * Session creation is PATH-based, never buffer-based. deploy/*.onnx keeps its
//   weights in a sibling .onnx.data file, and a buffer-based session cannot
//   resolve that sidecar (audit risk R3).
// * infer() is noexcept, allocation-free after initialize(), and writes into
//   caller-provided storage. ORT throws; the exception boundary is here, not
//   in the control loop.
// * A failed inference must NOT leave the caller looking at the previous
//   tick's action. On any failure the outputs are poisoned with NaN, so a
//   caller that ignores the status still fails closed at the finiteness gate
//   rather than actuating stale commands.
// * Non-finite outputs are detected and reported.
//
// There is deliberately NO normalizer in this library. Normalization is baked
// into the exported graph for some checkpoints and absent from others, so a
// native normalizer driven by a config flag would double-normalize one of them
// (audit risk R1). The engine feeds raw observations and that is the whole
// policy: nothing here can double-apply what it does not implement.
#ifndef PHOENIX_CORE__INFERENCE_HPP_
#define PHOENIX_CORE__INFERENCE_HPP_

#include <cstdint>
#include <memory>
#include <string>

#include "phoenix_core/types.hpp"

namespace phoenix_core
{

struct ModelConfig
{
  // Path to the .onnx file. If the graph uses external data, the sibling
  // .onnx.data must be present; it is resolved relative to this path.
  std::string model_path;

  std::size_t obs_dim = kObsDim;
  std::size_t action_dim = kNumJoints;

  // Expected latent width. The reliability shield consumes this output; set
  // require_latent=false for a model that does not emit one.
  std::size_t latent_dim = 384;
  bool require_latent = true;

  // Pinned to 1 and sequential by default. This is not a performance choice:
  // thread count and execution mode change reduction order, and therefore
  // change results at the bit level. Cross-language parity is only meaningful
  // when both sides run the same configuration, so the deterministic setting
  // is the default and any change is explicit.
  int intra_op_threads = 1;
  int inter_op_threads = 1;
};

struct InferenceResult
{
  Status status = Status::kOk;
  // Wall time for the Run() call, from a monotonic clock. Measured here rather
  // than by the caller so it excludes observation assembly and covers exactly
  // the model evaluation.
  std::int64_t duration_ns = 0;
  // True if any element of any output was non-finite. status is set to
  // kNonFinite in that case; the field exists so telemetry can distinguish a
  // NaN-producing model from a failed Run().
  bool non_finite_output = false;
};

class InferenceEngine
{
public:
  virtual ~InferenceEngine() = default;

  virtual Status initialize(const ModelConfig & config) = 0;

  // Evaluate the policy. `obs` must be obs_dim long; `action_out` action_dim;
  // `latent_out` latent_dim (may be null/0 when the model has no latent or
  // the caller does not want it).
  //
  // On any non-kOk status the outputs are filled with NaN rather than left
  // untouched, so a caller that ignores the return value still fails closed.
  virtual InferenceResult infer(
    const float * obs, std::size_t obs_len,
    float * action_out, std::size_t action_len,
    float * latent_out, std::size_t latent_len) noexcept = 0;

  // Human-readable description of the last initialize() failure. Off the hot
  // path; safe to allocate.
  virtual const std::string & last_error() const noexcept = 0;
};

// Construct the ONNX Runtime backend. Returns null only if the library was
// built without ONNX Runtime support.
std::unique_ptr<InferenceEngine> make_ort_engine();

// True if this build has an ONNX Runtime backend compiled in.
bool has_ort_backend() noexcept;

}  // namespace phoenix_core

#endif  // PHOENIX_CORE__INFERENCE_HPP_
