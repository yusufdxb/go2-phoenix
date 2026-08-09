// Copyright 2026 Yusuf Guenena. MIT License.
// See shield.hpp. Port of reliability/deploy.py and reliability/arbiter.py.
#include "phoenix_core/shield.hpp"

#include <cmath>
#include <limits>

namespace phoenix_core
{

const char * shield_state_name(ShieldState s) noexcept
{
  switch (s) {
    case ShieldState::kNominal: return "nominal";
    case ShieldState::kHandoff: return "handoff";
    case ShieldState::kFallback: return "fallback";
    case ShieldState::kRecovering: return "recovering";
  }
  return "nominal";
}

bool ArbiterConfig::valid() const noexcept
{
  if (!(clear_threshold < trip_threshold)) {
    return false;  // hysteresis invariant
  }
  if (trip_persistence < 1 || clear_persistence < 1 || handoff_ticks < 1 || recover_ticks < 1) {
    return false;
  }
  return min_fallback_ticks >= 0;
}

// ---------------------------------------------------------------------------
// SimplexArbiter (arbiter.py:106-161)
// ---------------------------------------------------------------------------

SimplexArbiter::SimplexArbiter(const ArbiterConfig & cfg) noexcept
: cfg_(cfg) {reset();}

void SimplexArbiter::reset() noexcept
{
  state_ = ShieldState::kNominal;
  blend_ = 0.0;
  ramp_ = 0;
  fallback_ticks_ = 0;
  over_trip_ = 0;
  under_clear_ = 0;
}

void SimplexArbiter::enter_handoff() noexcept
{
  state_ = ShieldState::kHandoff;
  ramp_ = 0;
  blend_ = 0.0;
}

ArbiterOutput SimplexArbiter::update(double score) noexcept
{
  // R17. The isfinite test MUST come first and MUST be an or: a bare
  // `score > trip` is false for NaN, which would let a garbage frame read as
  // healthy at exactly the moment the monitor is most suspicious. The
  // symmetric point matters too: below_clear requires isfinite, so a NaN can
  // never contribute to releasing the shield.
  const bool above_trip = !std::isfinite(score) || score > cfg_.trip_threshold;
  const bool below_clear = std::isfinite(score) && score < cfg_.clear_threshold;

  // Persistence counters reset the moment the streak breaks.
  over_trip_ = above_trip ? over_trip_ + 1 : 0;
  under_clear_ = below_clear ? under_clear_ + 1 : 0;

  switch (state_) {
    case ShieldState::kNominal:
      blend_ = 0.0;
      if (over_trip_ >= cfg_.trip_persistence) {
        enter_handoff();
      }
      break;

    case ShieldState::kHandoff: {
      // Committed to the ramp: the score is ignored mid-handoff so the
      // transition cannot chatter.
      ++ramp_;
      const double frac = static_cast<double>(ramp_) / static_cast<double>(cfg_.handoff_ticks);
      blend_ = frac < 1.0 ? frac : 1.0;
      if (ramp_ >= cfg_.handoff_ticks) {
        state_ = ShieldState::kFallback;
        blend_ = 1.0;
        fallback_ticks_ = 0;
      }
      break;
    }

    case ShieldState::kFallback: {
      blend_ = 1.0;
      ++fallback_ticks_;
      const bool eligible = !cfg_.latch && fallback_ticks_ >= cfg_.min_fallback_ticks;
      if (eligible && under_clear_ >= cfg_.clear_persistence) {
        state_ = ShieldState::kRecovering;
        ramp_ = 0;
      }
      break;
    }

    case ShieldState::kRecovering:
      // Safety dominates: a fresh trip aborts recovery back to full fallback.
      if (over_trip_ >= cfg_.trip_persistence) {
        state_ = ShieldState::kFallback;
        blend_ = 1.0;
        fallback_ticks_ = 0;
      } else {
        ++ramp_;
        const double v =
          1.0 - static_cast<double>(ramp_) / static_cast<double>(cfg_.recover_ticks);
        blend_ = v > 0.0 ? v : 0.0;
        if (ramp_ >= cfg_.recover_ticks) {
          state_ = ShieldState::kNominal;
          blend_ = 0.0;
        }
      }
      break;
  }

  return ArbiterOutput{state_, blend_};
}

// ---------------------------------------------------------------------------
// DeployMonitor (deploy.py:184-214)
// ---------------------------------------------------------------------------

Status DeployMonitor::initialize(
  const float * mean, const float * whitener, std::size_t dim) noexcept
{
  if (mean == nullptr || whitener == nullptr || dim == 0) {
    return Status::kDimMismatch;
  }
  for (std::size_t i = 0; i < dim; ++i) {
    if (!std::isfinite(mean[i])) {
      return Status::kNonFinite;
    }
  }
  for (std::size_t i = 0; i < dim * dim; ++i) {
    if (!std::isfinite(whitener[i])) {
      return Status::kNonFinite;
    }
  }

  dim_ = dim;
  mean_.assign(mean, mean + dim);
  whitener_.assign(whitener, whitener + dim * dim);
  diff_.assign(dim, 0.0f);
  proj_.assign(dim, 0.0f);
  return Status::kOk;
}

double DeployMonitor::score_one(const float * latent, std::size_t n) const noexcept
{
  if (dim_ == 0 || latent == nullptr || n != dim_) {
    // Fail toward the fallback rather than reporting a healthy zero.
    return std::numeric_limits<double>::infinity();
  }

  for (std::size_t i = 0; i < dim_; ++i) {
    diff_[i] = latent[i] - mean_[i];
  }
  // Matches the Python: the finiteness test is on the DIFFERENCE, after the
  // subtraction, so an infinite latent that cancels is still caught.
  for (std::size_t i = 0; i < dim_; ++i) {
    if (!std::isfinite(diff_[i])) {
      return std::numeric_limits<double>::infinity();
    }
  }

  // proj = whitener @ diff, row-major, accumulated in float32 to match the
  // width of numpy's float32 buffers.
  for (std::size_t r = 0; r < dim_; ++r) {
    const float * row = &whitener_[r * dim_];
    float acc = 0.0f;
    for (std::size_t c = 0; c < dim_; ++c) {
      acc += row[c] * diff_[c];
    }
    proj_[r] = acc;
  }

  float sq = 0.0f;
  for (std::size_t i = 0; i < dim_; ++i) {
    sq += proj_[i] * proj_[i];
  }
  return static_cast<double>(sq);
}

// ---------------------------------------------------------------------------
// DeployShield (deploy.py:240-274)
// ---------------------------------------------------------------------------

Status DeployShield::initialize(
  const float * mean, const float * whitener, std::size_t dim,
  const ArbiterConfig & cfg, int arming_ticks) noexcept
{
  ready_ = false;
  if (arming_ticks < 0 || !cfg.valid()) {
    return Status::kDimMismatch;
  }
  if (const Status s = monitor_.initialize(mean, whitener, dim); s != Status::kOk) {
    return s;
  }
  arbiter_ = SimplexArbiter(cfg);
  arming_ticks_ = arming_ticks;
  ticks_ = 0;
  ready_ = true;
  return Status::kOk;
}

void DeployShield::reset() noexcept
{
  arbiter_.reset();
  ticks_ = 0;
}

ShieldDecision DeployShield::step(const float * latent, std::size_t n) noexcept
{
  ShieldDecision d;
  if (!ready_) {
    // No artifact: report the disarmed, fully-nominal decision rather than a
    // fabricated score. The caller decides whether a shield-less run is
    // acceptable; that is a config question, not a runtime one.
    return d;
  }

  const double raw = monitor_.score_one(latent, n);
  d.raw_score = raw;
  // R15: no temporal filtering at deploy. Equal by construction.
  d.filtered_score = raw;

  // R16: the window is counted in ticks, and `armed` is evaluated BEFORE the
  // increment, so arming_ticks=15 yields exactly 15 disarmed calls.
  if (!armed()) {
    ++ticks_;
    d.blend = 0.0;
    d.state = ShieldState::kNominal;
    d.armed = false;
    return d;
  }
  ++ticks_;

  const ArbiterOutput out = arbiter_.update(raw);
  d.blend = out.blend;
  d.state = out.state;
  d.armed = true;
  return d;
}

}  // namespace phoenix_core
