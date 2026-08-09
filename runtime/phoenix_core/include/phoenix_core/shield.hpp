// Copyright 2026 Yusuf Guenena. MIT License.
//
// The reliability shield: Mahalanobis monitor + Simplex arbiter.
//
// Port of phoenix/reliability/deploy.py (DeployMonitor, DeployShield) and
// phoenix/reliability/arbiter.py (SimplexArbiter). Four things in here are
// easy to get wrong in a way that still produces a robot that walks:
//
// R17  A non-finite score must count as ABOVE TRIP and never toward CLEAR.
//      The Python writes `(not isfinite(score)) or score > trip`. The obvious
//      C++ transcription, `score > trip`, is FALSE for NaN, which inverts the
//      fail-closed convention precisely when the monitor is most suspicious.
//      The audit calls this the single most dangerous naive-port bug in the
//      file, so it is the first thing tested.
//
// R15  The deploy shield applies NO temporal filtering. The offline
//      ShieldRuntime does EWMA smoothing; DeployShield sets
//      filtered_score = raw. Porting the offline one would silently move the
//      operating point the artifact was calibrated at.
//
// R16  The arming window is counted in TICKS, not milliseconds. At a jittery
//      50 Hz, 15 ticks is not 300 ms, and the artifact records control_dt_s
//      separately.
//
// R12  The score is computed in float32 (matching numpy's float32 buffers) and
//      widened to double for the threshold comparison, because the thresholds
//      come from the artifact as doubles. Comparing a float32-rounded score
//      against a double threshold is a different predicate near the boundary.
//
// The shield is ADVISORY in the current deploy path: it returns a blend weight
// and cannot latch, abort, or suppress any hard predicate above it in the gate
// ladder. That is a deliberate property, not an omission. See
// docs/NATIVE_RUNTIME_PLAN.md §3.
#ifndef PHOENIX_CORE__SHIELD_HPP_
#define PHOENIX_CORE__SHIELD_HPP_

#include <cstdint>
#include <vector>

#include "phoenix_core/types.hpp"

namespace phoenix_core
{

// Wire encoding is pinned here and must not be reordered: the Python
// telemetry encodes state as an index into
// ("nominal", "handoff", "fallback", "recovering") independently of its own
// enum declaration order (audit risk R22).
enum class ShieldState : std::uint8_t
{
  kNominal = 0,     // learned policy in control
  kHandoff = 1,     // ramping learned -> fallback
  kFallback = 2,    // safe controller in control
  kRecovering = 3,  // ramping fallback -> learned
};

const char * shield_state_name(ShieldState s) noexcept;

struct ArbiterConfig
{
  double trip_threshold = 0.0;
  double clear_threshold = 0.0;
  int trip_persistence = 3;
  int clear_persistence = 10;
  int handoff_ticks = 10;
  int recover_ticks = 25;
  int min_fallback_ticks = 20;
  bool latch = false;

  // Mirrors SimplexArbiterCfg.__post_init__. The hysteresis invariant
  // (clear < trip) is a correctness property, not a style check: equal or
  // inverted thresholds make the arbiter chatter every tick.
  bool valid() const noexcept;
};

struct ArbiterOutput
{
  ShieldState state = ShieldState::kNominal;
  double blend = 0.0;  // weight on the FALLBACK controller, [0, 1]

  bool engaged() const noexcept {return state != ShieldState::kNominal;}
};

class SimplexArbiter
{
public:
  explicit SimplexArbiter(const ArbiterConfig & cfg) noexcept;

  // Advance one tick. A non-finite score is maximal evidence of trouble: it
  // counts toward tripping and never toward clearing (R17).
  ArbiterOutput update(double score) noexcept;
  void reset() noexcept;

  ShieldState state() const noexcept {return state_;}
  double blend() const noexcept {return blend_;}

private:
  void enter_handoff() noexcept;

  ArbiterConfig cfg_;
  ShieldState state_ = ShieldState::kNominal;
  double blend_ = 0.0;
  int ramp_ = 0;
  int fallback_ticks_ = 0;
  int over_trip_ = 0;
  int under_clear_ = 0;
};

// Squared Mahalanobis distance against a fitted latent distribution.
//
// Constants are copied in at initialize() and every working buffer is sized
// then, so score_one() performs no allocation.
class DeployMonitor
{
public:
  // mean is dim long; whitener is dim*dim, row-major. Rejects a non-finite
  // constant outright: a monitor fitted on garbage cannot fail safe later.
  Status initialize(const float * mean, const float * whitener, std::size_t dim) noexcept;

  std::size_t dim() const noexcept {return dim_;}

  // Returns +inf for a non-finite difference, matching the Python. Combined
  // with R17 in the arbiter, that pushes the shield toward fallback rather
  // than silently past it.
  double score_one(const float * latent, std::size_t n) const noexcept;

private:
  std::size_t dim_ = 0;
  std::vector<float> mean_;
  std::vector<float> whitener_;
  mutable std::vector<float> diff_;
  mutable std::vector<float> proj_;
};

struct ShieldDecision
{
  double blend = 0.0;
  ShieldState state = ShieldState::kNominal;
  double raw_score = 0.0;
  // Equal to raw_score by construction. The field exists so the telemetry
  // schema matches the Python and so nobody "restores" the EWMA (R15).
  double filtered_score = 0.0;
  bool armed = false;
};

class DeployShield
{
public:
  Status initialize(
    const float * mean, const float * whitener, std::size_t dim,
    const ArbiterConfig & cfg, int arming_ticks) noexcept;

  // One control tick. While disarmed the true score is reported but the
  // arbiter is held at NOMINAL and is NOT advanced.
  ShieldDecision step(const float * latent, std::size_t n) noexcept;

  void reset() noexcept;
  bool armed() const noexcept {return ticks_ >= arming_ticks_;}
  std::size_t dim() const noexcept {return monitor_.dim();}

private:
  DeployMonitor monitor_;
  SimplexArbiter arbiter_{ArbiterConfig{}};
  int arming_ticks_ = 0;
  int ticks_ = 0;
  bool ready_ = false;
};

}  // namespace phoenix_core

#endif  // PHOENIX_CORE__SHIELD_HPP_
