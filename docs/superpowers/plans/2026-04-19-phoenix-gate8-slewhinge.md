# Phoenix Gate 8 Slew-Hinge Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `phoenix-flat-slewhinge` — a v3b replacement that clears `slew_saturation_pct < 0.05` on flat.yaml with tracking intact — by adding a per-motor squared-hinge reward term that directly penalizes motors approaching the ±0.175 rad/step slew clip.

**Architecture:** Phase 2b's one code change is a new Phoenix-owned reward module (`src/phoenix/sim_env/rewards.py`) that contributes `slew_sat_hinge_l2`, wired into `go2_env_cfg.py` through a new `_NEW_TERM_FACTORIES` dispatcher parallel to the existing `_REWARD_TERM_MAP` (which only handles upstream Isaac Lab term names). A fresh env/train config pair fine-tunes v3b for 500 iters with the new term at `w=-50`, with a weight-sweep fallback to `{-500, -5}` on gate failure.

**Tech Stack:** Python 3.12, Isaac Lab 3.x (`~/IsaacLab`), PyTorch, rsl_rl PPO, ONNX Runtime, pytest, OmegaConf. GPU: RTX 5070. Branch: `audit-fixes-2026-04-16`.

**Spec:** `docs/superpowers/specs/2026-04-19-phoenix-gate8-slewhinge-design.md` (commit `6198234`).

**Preconditions:** Phase 0 of the 2026-04-19 retrain plan has already landed (commits `fa9ce5e`→`3b82f86`). `_apply_rewards` + `_REWARD_TERM_MAP` exist in `src/phoenix/sim_env/go2_env_cfg.py`; reward wiring is confirmed working by the 2026-04-19 A/B smoke.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/phoenix/sim_env/rewards.py` | create | Phoenix-owned reward functions, first being `slew_sat_hinge_l2`. |
| `tests/test_slew_sat_hinge.py` | create | 5 pure-torch unit tests for the reward function. |
| `src/phoenix/sim_env/go2_env_cfg.py` | modify | Add `_NEW_TERM_FACTORIES` dispatcher; extend `_apply_rewards` with new-term branch. |
| `tests/test_go2_env_cfg_rewards.py` | modify | +1 integration test: YAML with `slew_sat_hinge` key attaches the RewTerm to env_cfg.rewards. |
| `configs/env/flat_slewhinge.yaml` | create | New env overlay — v3b + `slew_sat_hinge: -50.0`. |
| `configs/train/ppo_flat_slewhinge.yaml` | create | PPO fine-tune config (clone of `ppo_flat_v3b_ft.yaml` with slewhinge env). |
| `checkpoints/phoenix-flat-slewhinge/<ts>/` | produce | Training artifacts (gitignored, rsync'd to T7). |
| `checkpoints/phoenix-flat/gate8/` | produce | Staged ONNX for the future Gate 8 deploy (sibling of `phoenix-flat/policy.onnx` which remains stand-v2). |
| `docs/rollout_flat_slewhinge_<flat|stand>_2026-04-19.json` | produce | Gate evidence. |
| `docs/retrain_flat_slewhinge_2026-04-19.md` | produce | Post-mortem (before/after vs v3b). |

---

## Task 1: Add `slew_sat_hinge_l2` reward function + tests

**Files:**
- Create: `src/phoenix/sim_env/rewards.py`
- Create: `tests/test_slew_sat_hinge.py`

- [ ] **Step 1: Write the failing tests.** Create `tests/test_slew_sat_hinge.py`:

```python
"""Tests for phoenix.sim_env.rewards.slew_sat_hinge_l2.

Pure-torch tests; no Isaac Lab dependency. Uses a trivial stand-in
`env` object whose only interface is `env.action_manager.action` and
`env.action_manager.prev_action` — matches how upstream Isaac Lab
action-rate rewards access actions (see
IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:action_rate_l2).
"""

from __future__ import annotations

import pytest
import torch

from phoenix.sim_env.rewards import slew_sat_hinge_l2


class _FakeActionManager:
    def __init__(self, action: torch.Tensor, prev: torch.Tensor):
        self.action = action
        self.prev_action = prev


class _FakeEnv:
    def __init__(self, action: torch.Tensor, prev: torch.Tensor):
        self.action_manager = _FakeActionManager(action, prev)


def test_zero_below_threshold() -> None:
    # All deltas = 0.1 (< 0.15 threshold)
    prev   = torch.zeros(2, 12)
    action = torch.full((2, 12), 0.1)
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env)
    assert torch.allclose(r, torch.zeros(2))


def test_squared_above_threshold() -> None:
    # Env 0: one motor at 0.175 (at clip), rest zero
    prev = torch.zeros(1, 12)
    action = torch.zeros(1, 12)
    action[0, 3] = 0.175
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env)
    # excess = 0.025, squared = 6.25e-4
    assert torch.allclose(r, torch.tensor([0.025 ** 2]), atol=1e-8)


def test_sums_across_motors() -> None:
    # 3 motors at 0.175 -> 3 * (0.025)^2
    prev = torch.zeros(1, 12)
    action = torch.zeros(1, 12)
    action[0, 0] = 0.175
    action[0, 5] = 0.175
    action[0, 11] = 0.175
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env)
    assert torch.allclose(r, torch.tensor([3 * 0.025 ** 2]), atol=1e-8)


def test_per_env_independent() -> None:
    # Env 0: all quiet. Env 1: one motor at clip.
    prev = torch.zeros(2, 12)
    action = torch.zeros(2, 12)
    action[1, 0] = 0.175
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env)
    assert r.shape == (2,)
    assert torch.allclose(r[0], torch.tensor(0.0))
    assert torch.allclose(r[1], torch.tensor(0.025 ** 2), atol=1e-8)


def test_threshold_parameter() -> None:
    # Motor at 0.175. With threshold=0.175 (strict >), penalty must be 0.
    prev = torch.zeros(1, 12)
    action = torch.zeros(1, 12)
    action[0, 0] = 0.175
    env = _FakeEnv(action, prev)
    r = slew_sat_hinge_l2(env, threshold=0.175)
    assert torch.allclose(r, torch.zeros(1))
```

- [ ] **Step 2: Run tests and verify failure.**

```
cd /home/yusuf/workspace/go2-phoenix
PYTHONPATH=src pytest tests/test_slew_sat_hinge.py -v
```

Expected: `ImportError: cannot import name 'slew_sat_hinge_l2' from 'phoenix.sim_env.rewards'` (module does not exist yet).

- [ ] **Step 3: Create `src/phoenix/sim_env/rewards.py`.**

```python
"""Phoenix-owned reward functions not provided by upstream Isaac Lab.

Added 2026-04-19 (Phase 2b retrain, spec
`docs/superpowers/specs/2026-04-19-phoenix-gate8-slewhinge-design.md`)
as a template for custom reward terms. Each function here follows
the upstream Isaac Lab signature `func(env, **params) -> Tensor[E]`.
"""

from __future__ import annotations

import torch

# 85% of MAX_DELTA_PER_STEP_RAD=0.175, the per-step slew clip enforced
# in phoenix.sim2real.safety and on the GO2 hardware deploy path.
_DEFAULT_HINGE_THRESHOLD = 0.15


def slew_sat_hinge_l2(
    env,
    threshold: float = _DEFAULT_HINGE_THRESHOLD,
) -> torch.Tensor:
    """Per-motor squared-hinge penalty on action deltas approaching
    the hardware slew clip.

    For each env at each control step, compute
    ``|a_t^i - a_{t-1}^i|`` per motor ``i``, apply a hinge at
    ``threshold``, square, and sum across motors. Returns a
    positive-magnitude tensor; caller applies a negative weight via
    ``RewTerm``.

    Targets the same failure mode that ``slew_saturation_pct`` in
    ``phoenix.training.evaluate`` measures: any single motor hitting
    the clip is sufficient to activate the penalty, unlike
    ``action_rate_l2`` which is an L2 norm across all motors and can
    stay small while individual motors saturate.

    Args:
        env: IsaacLab ``ManagerBasedRLEnv`` (duck-typed; tests use a
            stand-in with ``action_manager.action`` /
            ``action_manager.prev_action``).
        threshold: Hinge threshold in radians. Motors with
            ``|delta| <= threshold`` contribute 0.
    """
    action = env.action_manager.action            # [E, num_actions]
    prev = env.action_manager.prev_action         # [E, num_actions]
    delta = torch.abs(action - prev)
    excess = torch.clamp(delta - threshold, min=0.0)
    return (excess ** 2).sum(dim=-1)              # [E]
```

- [ ] **Step 4: Run tests and verify they pass.**

```
PYTHONPATH=src pytest tests/test_slew_sat_hinge.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit.**

```bash
git add src/phoenix/sim_env/rewards.py tests/test_slew_sat_hinge.py
git commit -m "sim_env: add slew_sat_hinge_l2 reward function + tests"
```

---

## Task 2: Extend `_apply_rewards` with new-term factory dispatcher

**Files:**
- Modify: `src/phoenix/sim_env/go2_env_cfg.py`
- Modify: `tests/test_go2_env_cfg_rewards.py`

**Purpose:** `_apply_rewards` currently only handles YAML keys whose upstream Isaac Lab term already exists (via `_REWARD_TERM_MAP` + `setattr`). For Phoenix-owned terms like `slew_sat_hinge`, we need to **construct** a new `RewTerm` and attach it. Adding a parallel `_NEW_TERM_FACTORIES` dispatcher keeps the two code paths clean without touching the existing upstream-term path.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_go2_env_cfg_rewards.py`:

```python
def test_apply_rewards_new_term_factory_attaches_reward() -> None:
    """YAML key `slew_sat_hinge` should trigger _NEW_TERM_FACTORIES
    dispatch, constructing a RewTerm via the factory and attaching
    it to env_cfg.rewards.slew_sat_hinge_l2 with the given weight."""
    from phoenix.sim_env.go2_env_cfg import _NEW_TERM_FACTORIES

    # Sanity: the new factory is registered under the YAML key.
    assert "slew_sat_hinge" in _NEW_TERM_FACTORIES

    # env_cfg.rewards starts without the new term attached.
    env_cfg = _FakeEnvCfg(_FakeRewards())
    _apply_rewards(env_cfg, {"slew_sat_hinge": -50.0})

    # After _apply_rewards, the attribute exists and has correct weight.
    assert hasattr(env_cfg.rewards, "slew_sat_hinge_l2")
    assert env_cfg.rewards.slew_sat_hinge_l2.weight == -50.0


def test_apply_rewards_new_term_factory_mixed_with_upstream() -> None:
    """Mixing upstream (setattr) and new (factory) keys in one call
    should produce both effects correctly."""
    env_cfg = _FakeEnvCfg(
        _FakeRewards(action_rate_l2=_FakeRewardTerm(-0.01)),
    )
    _apply_rewards(
        env_cfg,
        {"action_rate": -0.5, "slew_sat_hinge": -50.0},
    )
    assert env_cfg.rewards.action_rate_l2.weight == -0.5
    assert env_cfg.rewards.slew_sat_hinge_l2.weight == -50.0
```

- [ ] **Step 2: Run tests and verify failure.**

```
PYTHONPATH=src pytest tests/test_go2_env_cfg_rewards.py::test_apply_rewards_new_term_factory_attaches_reward -v
```

Expected: `ImportError: cannot import name '_NEW_TERM_FACTORIES'`.

- [ ] **Step 3: Extend `src/phoenix/sim_env/go2_env_cfg.py`.**

At the top of the file, add to the existing imports block:

```python
# Isaac Lab's RewardTermCfg — lazy-import guard so this module can
# still be imported in CI without isaaclab. The actual usage is
# gated behind _NEW_TERM_FACTORIES, which only fires at env build.
try:  # pragma: no cover - exercised only on machines with Isaac Lab
    from isaaclab.managers import RewardTermCfg as _RewTerm
except ImportError:  # pragma: no cover
    _RewTerm = None  # type: ignore[assignment]

from phoenix.sim_env.rewards import slew_sat_hinge_l2
```

Below `_REWARD_TERM_MAP`, add:

```python
# Factories for Phoenix-owned reward terms — not in upstream
# UnitreeGo2RoughEnvCfg.rewards. When a YAML key lands here,
# _apply_rewards constructs a RewTerm via the factory and setattrs
# it onto env_cfg.rewards. Keys must not collide with
# _REWARD_TERM_MAP — see _apply_rewards dispatch.
_NEW_TERM_FACTORIES: dict[str, tuple[str, callable]] = {
    "slew_sat_hinge": (
        "slew_sat_hinge_l2",  # attribute name on env_cfg.rewards
        lambda weight: _RewTerm(
            func=slew_sat_hinge_l2,
            weight=float(weight),
            params={"threshold": 0.15},
        ),
    ),
}
```

Then modify `_apply_rewards` from:

```python
def _apply_rewards(env_cfg: Any, rewards: dict[str, Any]) -> None:
    if not rewards:
        return
    for yaml_key, weight in rewards.items():
        if yaml_key not in _REWARD_TERM_MAP:
            raise KeyError(...)
        term_name = _REWARD_TERM_MAP[yaml_key]
        term = getattr(env_cfg.rewards, term_name, None)
        if term is None:
            raise AttributeError(...)
        term.weight = float(weight)
```

to:

```python
def _apply_rewards(env_cfg: Any, rewards: dict[str, Any]) -> None:
    """Apply YAML reward overrides to Isaac Lab env cfg.

    Upstream-term keys (in ``_REWARD_TERM_MAP``) reweight an existing
    ``env_cfg.rewards.<term>`` by setting its ``weight``.

    Phoenix-owned-term keys (in ``_NEW_TERM_FACTORIES``) construct a
    new ``RewTerm`` via the factory and attach it to
    ``env_cfg.rewards`` under the factory's attribute name.

    Unknown keys raise ``KeyError`` — this is deliberate, to prevent
    the silent-no-op drift that motivated adding this helper (see
    :mod:`phoenix.sim_env.go2_env_cfg` module docstring, 2026-04-19).
    """
    if not rewards:
        return
    for yaml_key, weight in rewards.items():
        if yaml_key in _REWARD_TERM_MAP:
            term_name = _REWARD_TERM_MAP[yaml_key]
            term = getattr(env_cfg.rewards, term_name, None)
            if term is None:
                raise AttributeError(
                    f"Reward term {term_name!r} (YAML key {yaml_key!r}) not present on "
                    f"{type(env_cfg.rewards).__name__}. Either the upstream task omits "
                    f"this term, or _REWARD_TERM_MAP is stale."
                )
            term.weight = float(weight)
        elif yaml_key in _NEW_TERM_FACTORIES:
            attr_name, factory = _NEW_TERM_FACTORIES[yaml_key]
            setattr(env_cfg.rewards, attr_name, factory(weight))
        else:
            raise KeyError(
                f"Unknown reward key {yaml_key!r} — add it to _REWARD_TERM_MAP, "
                f"add it to _NEW_TERM_FACTORIES, or remove from YAML. "
                f"Known upstream keys: {sorted(_REWARD_TERM_MAP)}; "
                f"known phoenix keys: {sorted(_NEW_TERM_FACTORIES)}"
            )
```

(The upstream-term branch preserves existing behavior exactly — the only structural change is adding the `elif` for factory-based terms and updating the `KeyError` message.)

- [ ] **Step 4: Patch the fake classes in the test to track `setattr` properly.**

The existing `_FakeRewards` class in `tests/test_go2_env_cfg_rewards.py` uses `setattr` in `__init__`, so assigning new attrs later works by default. But since the new test passes a `_RewTerm`-shaped value (the factory's return), and `_RewTerm` comes from `isaaclab.managers` which isn't importable in CI, the factory will dereference `_RewTerm is None` and fail.

Patch: in `tests/test_go2_env_cfg_rewards.py`, add a test-local stub and monkey-patch `_RewTerm` for the integration tests.

At the top of `tests/test_go2_env_cfg_rewards.py`, add:

```python
from unittest.mock import patch
```

Modify the two new tests to wrap the `_apply_rewards` call in a patch:

```python
def test_apply_rewards_new_term_factory_attaches_reward() -> None:
    from phoenix.sim_env.go2_env_cfg import _NEW_TERM_FACTORIES

    assert "slew_sat_hinge" in _NEW_TERM_FACTORIES

    class _StubRewTerm:
        def __init__(self, *, func, weight, params):
            self.func = func
            self.weight = weight
            self.params = params

    env_cfg = _FakeEnvCfg(_FakeRewards())
    with patch("phoenix.sim_env.go2_env_cfg._RewTerm", _StubRewTerm):
        _apply_rewards(env_cfg, {"slew_sat_hinge": -50.0})

    assert hasattr(env_cfg.rewards, "slew_sat_hinge_l2")
    assert env_cfg.rewards.slew_sat_hinge_l2.weight == -50.0
    # Sanity: the factory passed threshold=0.15 per the spec.
    assert env_cfg.rewards.slew_sat_hinge_l2.params == {"threshold": 0.15}


def test_apply_rewards_new_term_factory_mixed_with_upstream() -> None:
    class _StubRewTerm:
        def __init__(self, *, func, weight, params):
            self.func = func
            self.weight = weight
            self.params = params

    env_cfg = _FakeEnvCfg(
        _FakeRewards(action_rate_l2=_FakeRewardTerm(-0.01)),
    )
    with patch("phoenix.sim_env.go2_env_cfg._RewTerm", _StubRewTerm):
        _apply_rewards(
            env_cfg,
            {"action_rate": -0.5, "slew_sat_hinge": -50.0},
        )
    assert env_cfg.rewards.action_rate_l2.weight == -0.5
    assert env_cfg.rewards.slew_sat_hinge_l2.weight == -50.0
```

- [ ] **Step 5: Run the full rewards test file.**

```
PYTHONPATH=src pytest tests/test_go2_env_cfg_rewards.py -v
```

Expected: 10 passed (was 8 — adds 2 new factory tests).

- [ ] **Step 6: Run full CI suite as regression check.**

```
PYTHONPATH=src pytest tests/ --ignore=tests/test_sim_integration.py -q
```

Expected: 160 passed (was 158 before the 2 new factory tests).

- [ ] **Step 7: Commit.**

```bash
git add src/phoenix/sim_env/go2_env_cfg.py tests/test_go2_env_cfg_rewards.py
git commit -m "sim_env: wire slew_sat_hinge RewTerm via _NEW_TERM_FACTORIES"
```

---

## Task 3: Add env + train configs for `phoenix-flat-slewhinge`

**Files:**
- Create: `configs/env/flat_slewhinge.yaml`
- Create: `configs/train/ppo_flat_slewhinge.yaml`

- [ ] **Step 1: Create `configs/env/flat_slewhinge.yaml`.**

```yaml
# Flat-slewhinge: v3b + direct per-motor slew-clip penalty.
#
# See docs/superpowers/specs/2026-04-19-phoenix-gate8-slewhinge-design.md
# for why L2 action_rate was insufficient (flat-v3b-ft negative result).
# This overlay adds slew_sat_hinge_l2 — a per-motor squared hinge that
# penalizes motors approaching the ±0.175 rad/step hardware slew clip
# (threshold 0.15 rad = 85% of clip).
#
# Scope: walking-only. cmd_vel=0 is handled by a deploy-layer
# mode-switch to stand-v2 (separate spec). rel_standing_envs stays at
# base 0.02; action_rate / joint_acc stay at base (0.05 / -2.5e-7) so
# any delta vs v3b is attributable to the new term.
#
# Evaluation criteria (from retrain spec Phase 2b gates):
# - slew_saturation_pct < 0.05 on flat.yaml (nonzero cmds)
# - mean_lin_vel_err <= 0.10 m/s on flat.yaml
# - mean_ang_vel_err <= 0.10 rad/s on flat.yaml
# - 32/32 success @ 20 s on flat.yaml

defaults:
  - flat

reward:
  slew_sat_hinge: -50.0
```

- [ ] **Step 2: Smoke-load the new env config.**

```
PYTHONPATH=src python3 -c "
from phoenix.sim_env.config_loader import load_layered_config
cfg = load_layered_config('configs/env/flat_slewhinge.yaml').to_container()
print('reward.slew_sat_hinge =', cfg['reward']['slew_sat_hinge'])
print('reward.action_rate =', cfg['reward']['action_rate'])
print('command.rel_standing_envs =', cfg['command']['rel_standing_envs'])
print('env.task_name =', cfg['env']['task_name'])
"
```

Expected output:
```
reward.slew_sat_hinge = -50.0
reward.action_rate = -0.05
command.rel_standing_envs = 0.02
env.task_name = Isaac-Velocity-Flat-Unitree-Go2-v0
```

(`reward.action_rate` / `rel_standing_envs` come from the `flat` → `base` defaults chain, confirming the overlay only adds the new term.)

- [ ] **Step 3: Commit env config.**

```bash
git add configs/env/flat_slewhinge.yaml
git commit -m "configs: add flat_slewhinge env overlay (v3b + slew_sat_hinge=-50)"
```

- [ ] **Step 4: Create `configs/train/ppo_flat_slewhinge.yaml`.**

```yaml
# PPO fine-tune config for flat-slewhinge.
#
# Fine-tunes from phoenix-flat/2026-04-16_21-39-16/model_999.pt (v3b).
# Invoke with --resume pointed at that path. 500 iters matches the
# budget used by flat-v3b-ft; reward signal is now per-motor and much
# more targeted. entropy_coef stays at 0.005 (v4's 0.01 was a mistake).

run:
  name: "phoenix-flat-slewhinge"
  output_dir: "checkpoints"
  log_interval: 1
  save_interval: 50
  max_iterations: 500
  seed: 42
  device: "cuda:0"

env:
  config: "configs/env/flat_slewhinge.yaml"

algorithm:
  class_name: "PPO"
  value_loss_coef: 1.0
  use_clipped_value_loss: true
  clip_param: 0.2
  entropy_coef: 0.005
  num_learning_epochs: 5
  num_mini_batches: 4
  learning_rate: 1.0e-3
  schedule: "adaptive"
  gamma: 0.99
  lam: 0.95
  desired_kl: 0.01
  max_grad_norm: 1.0

policy:
  class_name: "ActorCritic"
  init_noise_std: 0.1
  actor_hidden_dims: [512, 256, 128]
  critic_hidden_dims: [512, 256, 128]
  activation: "elu"

runner:
  num_steps_per_env: 24
  empirical_normalization: true

logging:
  tensorboard: true
  wandb: false
  wandb_project: "go2-phoenix"
```

- [ ] **Step 5: Commit train config.**

```bash
git add configs/train/ppo_flat_slewhinge.yaml
git commit -m "configs: add ppo_flat_slewhinge fine-tune config"
```

---

## Task 4: Train + gate + export + stage `phoenix-flat-slewhinge` (seed 42, w=-50)

**GPU task — not dispatchable to a subagent.** Run interactively on mewtwo. Steps reference env vars `RUN_DIR` set in Step 2 for later steps.

**Files:**
- Produce: `checkpoints/phoenix-flat-slewhinge/<timestamp>/`
- Produce: `docs/rollout_flat_slewhinge_flat_2026-04-19.json`
- Produce: `docs/rollout_flat_slewhinge_stand_2026-04-19.json`
- Produce: `checkpoints/phoenix-flat/gate8/policy.{onnx,onnx.data,pt}`

- [ ] **Step 1: Confirm baseline checkpoint exists.**

```bash
ls -l checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt
```

Expected: ~4.5 MB file, present.

- [ ] **Step 2: Launch training.**

```bash
cd ~/workspace/go2-phoenix
./scripts/train.sh configs/train/ppo_flat_slewhinge.yaml \
    --resume checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt \
    2>&1 | tee /tmp/phoenix-flat-slewhinge-seed42.log
```

Expected: ~15 min wall on RTX 5070. Watch for `slew_sat_hinge_l2 | -50.0` in Isaac Lab's reward-term table at iter 0 (confirms wiring).

- [ ] **Step 3: Record the output dir.**

```bash
export RUN_DIR=$(ls -dt checkpoints/phoenix-flat-slewhinge/2026-* | head -1)
echo "RUN_DIR=$RUN_DIR"
ls "$RUN_DIR"/model_*.pt | tail -3
```

Expected: `model_499.pt` (final), `model_450.pt`, `model_50.pt`.

- [ ] **Step 4: Reward curve sanity check.**

```bash
source ~/isaac-sim-venv/bin/activate
python3 - <<PY
from tensorboard.backend.event_processing import event_accumulator
import glob
ef = sorted(glob.glob(f"$RUN_DIR/events.out.tfevents.*"))[0]
ea = event_accumulator.EventAccumulator(ef, size_guidance={event_accumulator.SCALARS: 10000})
ea.Reload()
for tag in ["Train/mean_reward", "Episode_Reward/slew_sat_hinge_l2",
            "Episode_Reward/action_rate_l2", "Train/mean_episode_length"]:
    if tag not in ea.Tags()["scalars"]:
        print(f"MISSING {tag}")
        continue
    s = ea.Scalars(tag)
    print(f"{tag}: n={len(s)}, first={s[0].value:.4f}, last={s[-1].value:.4f}, min={min(p.value for p in s):.4f}, max={max(p.value for p in s):.4f}")
PY
```

Pass criteria:
- `Train/mean_reward` last > first (or at least not a cliff — flat-v3b-ft saw an exploration dip that recovered).
- `Episode_Reward/slew_sat_hinge_l2` magnitude decreases over training (policy learning to avoid the hinge).
- `Train/mean_episode_length` reaches >900 by iter 400.

If any sanity gate fails, **STOP** — inspect log for training instability before running eval gates.

- [ ] **Step 5: Run flat.yaml rollout (G1 + G2 + G3 + G4).**

```bash
cd ~/workspace/go2-phoenix
source ~/isaac-sim-venv/bin/activate
export ISAACLAB_PATH="$HOME/IsaacLab" OMNI_KIT_ACCEPT_EULA=YES
PYTHONPATH="$PWD/src:${PYTHONPATH:-}" \
"$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.training.evaluate \
    --checkpoint "$RUN_DIR/model_499.pt" \
    --env-config configs/env/flat.yaml \
    --num-envs 16 --num-episodes 32 \
    --slew-saturation-max 0.05 \
    --metrics-out docs/rollout_flat_slewhinge_flat_2026-04-19.json
```

Expected exit 0. Inspect:

```bash
python3 -c "import json; d=json.load(open('docs/rollout_flat_slewhinge_flat_2026-04-19.json')); print(d)"
```

Required:
- `success_rate == 1.0`  (G1)
- `mean_episode_length_s == 20.0`  (G1)
- `slew_saturation_pct < 0.05`  (G2; note `--slew-saturation-max 0.05` already enforced exit=0)
- `mean_lin_vel_error <= 0.10`  (G3)
- `mean_ang_vel_error <= 0.10`  (G4)

If any fails: skip Step 6-8, proceed to Task 5 (weight-sweep fallback).

- [ ] **Step 6: Run stand.yaml rollout (informational only).**

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH:-}" \
"$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.training.evaluate \
    --checkpoint "$RUN_DIR/model_499.pt" \
    --env-config configs/env/stand.yaml \
    --num-envs 16 --num-episodes 32 \
    --metrics-out docs/rollout_flat_slewhinge_stand_2026-04-19.json
```

No gate — just record for the post-mortem. Expect `slew_saturation_pct` to be *worse* than stand-v2's 0.003 (this policy isn't optimized for cmd=0) but survival should be high.

- [ ] **Step 7: Commit metrics JSONs.**

```bash
git add docs/rollout_flat_slewhinge_flat_2026-04-19.json \
        docs/rollout_flat_slewhinge_stand_2026-04-19.json
git commit -m "flat-slewhinge seed 42 w=-50: rollout metrics (G1-G4 PASS)"
```

- [ ] **Step 8: Export ONNX with self-verify (G5 part 1).**

```bash
PYTHONPATH=src python3 -m phoenix.sim2real.export \
    --checkpoint "$RUN_DIR/model_499.pt" \
    --output "$RUN_DIR/policy.onnx" \
    --verify
```

Expected: exit 0, log line `Max torch<->onnx abs diff: <1e-5>`.

- [ ] **Step 9: Stage at `phoenix-flat/gate8/` (not over stand-v2's policy.onnx).**

```bash
mkdir -p checkpoints/phoenix-flat/gate8
cp "$RUN_DIR/policy.onnx"      checkpoints/phoenix-flat/gate8/policy.onnx
cp "$RUN_DIR/policy.onnx.data" checkpoints/phoenix-flat/gate8/policy.onnx.data
cp "$RUN_DIR/policy.pt"        checkpoints/phoenix-flat/gate8/policy.pt
```

Verify stand-v2 is untouched at `phoenix-flat/policy.onnx`:

```bash
ls -l checkpoints/phoenix-flat/policy.onnx checkpoints/phoenix-flat/gate8/policy.onnx
```

Both files should exist with distinct mtimes.

- [ ] **Step 10: G5 parity gate — need temporary deploy.yaml pointing at the gate8 ONNX.**

Create a transient deploy cfg sibling:

```bash
cat > /tmp/deploy_gate8.yaml <<'EOF'
policy:
  onnx_path: "checkpoints/phoenix-flat/gate8/policy.onnx"
  obs_pad_zeros: 0
EOF
# fold in the rest of configs/sim2real/deploy.yaml by appending (keeps
# joint_order, control.default_joint_pos, etc.)
python3 - <<PY
import yaml
base = yaml.safe_load(open("configs/sim2real/deploy.yaml"))
over = yaml.safe_load(open("/tmp/deploy_gate8.yaml"))
base["policy"] = over["policy"]
open("/tmp/deploy_gate8.yaml", "w").write(yaml.safe_dump(base))
PY

PYTHONPATH=src python3 -m phoenix.sim2real.verify_deploy \
    --parquet data/failures/synth_slippery_trained.parquet \
    --deploy-cfg /tmp/deploy_gate8.yaml \
    --tol 1e-4 \
    --max-steps 200
rm /tmp/deploy_gate8.yaml
```

Expected: `Parity: ... -> PASS` with max_diff < 1e-5.

- [ ] **Step 11: Rsync to T7.**

```bash
mkdir -p "/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat-slewhinge/$(basename $RUN_DIR)"
rsync -avL "$RUN_DIR/" \
    "/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat-slewhinge/$(basename $RUN_DIR)/"

mkdir -p "/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat/gate8"
rsync -avL checkpoints/phoenix-flat/gate8/ \
    "/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat/gate8/"
```

Expected: both rsyncs succeed, no errors.

---

## Task 5 (conditional): Weight-sweep fallback

**Only run if Task 4 Step 5 failed a gate.**

Decision tree per spec §Seed/weight policy:
- G2 fail (slew > 5%) → retry at **w=-500**
- G3 or G4 fail (tracking regressed) → retry at **w=-5**
- Both → prioritize **w=-500** first (primary gate is G2)

- [ ] **Step 1: Create sibling env config.** Copy `configs/env/flat_slewhinge.yaml` → `configs/env/flat_slewhinge_w<N>.yaml` where `<N>` ∈ `{500, 5}`. Change only the `slew_sat_hinge:` value.

- [ ] **Step 2: Create sibling train config.** Copy `configs/train/ppo_flat_slewhinge.yaml` → `configs/train/ppo_flat_slewhinge_w<N>.yaml`. Change `run.name` to `"phoenix-flat-slewhinge-w<N>"` and `env.config` to the new env path.

- [ ] **Step 3: Commit the sibling configs.**

```bash
git add configs/env/flat_slewhinge_w<N>.yaml configs/train/ppo_flat_slewhinge_w<N>.yaml
git commit -m "configs: add flat_slewhinge_w<N> fallback (weight sweep)"
```

- [ ] **Step 4: Re-run Task 4 Steps 2–11** with the new train config and `RUN_DIR` pointing at the new run's timestamp dir. Rollout JSONs use filename `docs/rollout_flat_slewhinge_w<N>_{flat,stand}_2026-04-19.json`. Staging path is still `checkpoints/phoenix-flat/gate8/` (overwrites prior attempt's gate8 stage only if this run passes).

- [ ] **Step 5: If G2 passes but tracking fails:** pivot to `w=-5`. Repeat Step 1-4 with that weight.

- [ ] **Step 6: If none of `{-50, -500, -5}` passes all gates:** STOP. Document outcome in the post-mortem (Task 6). Do not ship. Open a follow-on spec — candidate next moves per spec §Known risks §1, §2: widen hinge threshold to 0.17 or double iters to 1000.

---

## Task 6: Post-mortem + vault + memory

**Files:**
- Create: `docs/retrain_flat_slewhinge_2026-04-19.md`
- Modify: `Vault/Projects/go2-phoenix/RETRAIN_2026-04-19.md`
- Modify: `Vault/Projects/go2-phoenix/status.md`
- Modify: `~/.claude/projects/-home-yusuf/memory/project_go2_phoenix.md`

- [ ] **Step 1: Write `docs/retrain_flat_slewhinge_2026-04-19.md`.**

Follow the structure of `docs/retrain_stand_v2_2026-04-19.md`:

- Summary (baseline, config, seed, iters, accepted weight, status: SHIPPED or FAILED).
- Run dir.
- Gates table with thresholds and actuals from JSON.
- v3b baseline comparison (mean_episode_return, lin_vel_err, ang_vel_err on flat.yaml from commit `61fae38`; slew_saturation_pct NEW from this rollout).
- Training curve highlights (mean_reward trajectory, slew_sat_hinge_l2 magnitude over time).
- Staging decision (`phoenix-flat/gate8/` NOT `phoenix-flat/policy.onnx`).
- What's out of scope (deploy-layer mode-switch, separate spec).

Fill the `{}` placeholders from the JSON outputs.

- [ ] **Step 2: Commit post-mortem.**

```bash
git add docs/retrain_flat_slewhinge_2026-04-19.md
git commit -m "docs: flat-slewhinge post-mortem (2026-04-19, Gate 8 candidate)"
```

- [ ] **Step 3: Append Phase 2b section to `Vault/Projects/go2-phoenix/RETRAIN_2026-04-19.md`.**

Replace the existing "Phase 2b — Gate 8 follow-up (planned, next session)" block with a Phase 2b actual-outcome block. Include: accepted weight; G1-G5 table; before/after vs v3b; Gate 8 staging decision.

- [ ] **Step 4: Update `Vault/Projects/go2-phoenix/status.md`.**

Current-state line should flip to: "Gate 7 READY (stand-v2); Gate 8 candidate produced (flat-slewhinge); deploy-layer mode-switch is next blocker before Gate 8 lab day."

- [ ] **Step 5: Commit vault.**

```bash
cd "/home/yusuf/Documents/Obsidian Vault"
git add "Projects/go2-phoenix/RETRAIN_2026-04-19.md" \
        "Projects/go2-phoenix/status.md"
git commit -m "vault: phoenix retrain 2026-04-19 Phase 2b — Gate 8 candidate"
```

- [ ] **Step 6: Update memory.**

Edit `~/.claude/projects/-home-yusuf/memory/project_go2_phoenix.md`:
- Frontmatter `description`: add "Gate 8 candidate `phoenix-flat-slewhinge` at `checkpoints/phoenix-flat/gate8/`; deploy mode-switch spec pending".
- Append a "Phase 2b" paragraph at the end covering: accepted weight, key metric deltas vs v3b, staging path, next blocker (mode-switch spec).

Update `MEMORY.md` index line for `project_go2_phoenix.md` with one-line summary of Gate 8 candidate shipped.

---

## Self-review checklist

- [ ] **Spec coverage.** Phase 2b reward function → Task 1. `_NEW_TERM_FACTORIES` dispatcher → Task 2. Env/train configs → Task 3. Training + all 5 gates + staging + T7 sync → Task 4. Weight-sweep fallback → Task 5. Post-mortem + vault + memory → Task 6. All spec sections (§Implementation, §Run spec, §Gates, §Seed/weight policy, §Artifacts & staging, §Reporting) have tasks. ✓
- [ ] **Placeholder scan.** Only `{}` placeholder in Task 6 Step 1 (JSON values filled at execution time, by design). No TBD/TODO. No "add appropriate error handling". ✓
- [ ] **Type consistency.** `_REWARD_TERM_MAP: dict[str, str]` (upstream-term map) vs `_NEW_TERM_FACTORIES: dict[str, tuple[str, callable]]` (phoenix factory map) — distinct by design, no name collision. `slew_sat_hinge_l2` function is consistently named across Task 1, Task 2, YAML reference, and env_cfg attribute. ✓
- [ ] **Commit discipline.** Each task ends in a commit; Task 4 splits into three commits (metrics + post-mortem separately from training artifacts which are gitignored). ✓
