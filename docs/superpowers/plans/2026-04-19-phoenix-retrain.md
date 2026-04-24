# Phoenix Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two hardware-safe GO2 checkpoints — a real stand specialist (`phoenix-stand-v2`) and a smoothness-tuned flat policy (`phoenix-flat-v3b-ft`) — that both clear <5% slew-saturation at cmd=0 in sim. Prerequisite: wire the currently-ignored `reward` YAML section into the env cfg.

**Architecture:** Phase 0 fixes a silent no-op in `src/phoenix/sim_env/go2_env_cfg.py` where YAML `reward.*` overrides are listed in `_UNWIRED_TOP_LEVEL` and logged as warnings instead of applied. Once wired, Phase 1 fine-tunes `phoenix-stand` for 500 iters on `stand_v2.yaml` (+10× `action_rate` and +4× `joint_acc` over `base.yaml`, init_noise 0.1), and Phase 2 fine-tunes `phoenix-flat` v3b for 500 iters on a new `flat_v3b_ft.yaml` (same reward knobs + `rel_standing_envs` 0.02→0.10). Each accepted run is ONNX-exported, parity-gated, rsync'd to T7. stand-v2 is staged as the Gate 7 candidate; flat-v3b-ft is the Gate 8 candidate.

**Tech Stack:** Python 3.12, Isaac Lab 3.x (`~/IsaacLab`), rsl_rl PPO, PyTorch, ONNX Runtime, pytest, OmegaConf, git on branch `audit-fixes-2026-04-16`. GPU: RTX 5070 on mewtwo.

**Spec:** `docs/superpowers/specs/2026-04-19-phoenix-retrain-design.md` (commits `67e29da` + `7f05679`).

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/phoenix/sim_env/go2_env_cfg.py` | modify | Add `_REWARD_TERM_MAP` + `_apply_rewards`; remove `"reward"` from `_UNWIRED_TOP_LEVEL`; call `_apply_rewards` from `build_env_cfg` |
| `configs/env/base.yaml` | modify | Strip unmappable reward keys (`base_height`, `joint_vel`, `collision`, `termination`) — they were aspirational / unmapped in upstream `UnitreeGo2RoughEnvCfg` |
| `tests/test_go2_env_cfg_rewards.py` | create | Three regression tests: mapping table, unknown-key raise, wired warning no longer fires for `reward` |
| `configs/env/flat_v3b_ft.yaml` | create | New env config: `defaults: [flat]`, reward knobs (+10× action_rate, +4× joint_acc), `rel_standing_envs: 0.10` |
| `configs/train/ppo_flat_v3b_ft.yaml` | create | New PPO fine-tune config — mirror of `ppo_stand_v2.yaml` with flat env + `phoenix-flat-v3b-ft` run name |
| `checkpoints/phoenix-stand-v2/<timestamp>/` | produce | Phase 1 training artifacts (gitignored, rsync'd to T7) |
| `checkpoints/phoenix-flat-v3b-ft/<timestamp>/` | produce | Phase 2 training artifacts (gitignored, rsync'd to T7) |
| `checkpoints/phoenix-flat/policy.onnx` | update | Staged stand-v2 ONNX for Gate 7 lab day |
| `checkpoints/phoenix-flat/v3b-ft/` | create | Staged flat-v3b-ft ONNX for Gate 8 |
| `docs/retrain_stand_v2_2026-04-19.md` | create | Phase 1 post-mortem (before/after metrics, seeds used, staging decision) |
| `docs/retrain_flat_v3b_ft_2026-04-19.md` | create | Phase 2 post-mortem |
| `Vault/Projects/go2-phoenix/RETRAIN_2026-04-19.md` | create | Vault-side session summary |

---

## Phase 0 — Wire the `reward` section

### Task 0.1: Add failing test for reward term mapping

**Files:**
- Create: `tests/test_go2_env_cfg_rewards.py`

- [ ] **Step 1: Write the failing test.** Create `tests/test_go2_env_cfg_rewards.py` with:

```python
"""Regression tests for reward-section wiring in go2_env_cfg.

These tests run in CI (non-sim). They exercise the pure-Python helpers
added in Phase 0 of the 2026-04-19 phoenix retrain plan.
"""

from __future__ import annotations

import logging

import pytest

from phoenix.sim_env.go2_env_cfg import (
    _REWARD_TERM_MAP,
    _apply_rewards,
    _unwired_sections_present,
)


def test_reward_term_map_covers_phoenix_base_keys() -> None:
    """Every reward key we keep in base.yaml must map to an upstream
    Isaac Lab reward term name."""
    expected = {
        "track_lin_vel_xy": "track_lin_vel_xy_exp",
        "track_ang_vel_z": "track_ang_vel_z_exp",
        "lin_vel_z": "lin_vel_z_l2",
        "ang_vel_xy": "ang_vel_xy_l2",
        "joint_torque": "dof_torques_l2",
        "joint_acc": "dof_acc_l2",
        "action_rate": "action_rate_l2",
        "feet_air_time": "feet_air_time",
    }
    assert _REWARD_TERM_MAP == expected
```

- [ ] **Step 2: Run test and verify it fails.**

```
cd ~/workspace/go2-phoenix
PYTHONPATH=src pytest tests/test_go2_env_cfg_rewards.py::test_reward_term_map_covers_phoenix_base_keys -v
```

Expected: `ImportError: cannot import name '_REWARD_TERM_MAP' from 'phoenix.sim_env.go2_env_cfg'`.

- [ ] **Step 3: Add the `_REWARD_TERM_MAP` constant.** Edit `src/phoenix/sim_env/go2_env_cfg.py`, adding after `_UNWIRED_ROBOT_SUB` (~line 42):

```python
# YAML reward key -> upstream Isaac Lab RewardsCfg term attribute name.
# Upstream term names live at
# IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/
#   velocity/velocity_env_cfg.py (class RewardsCfg).
# Only terms supported by UnitreeGo2RoughEnvCfg are listed. Keys in YAML
# not present here raise KeyError in _apply_rewards — we do NOT want
# silent drift reappearing.
_REWARD_TERM_MAP: dict[str, str] = {
    "track_lin_vel_xy": "track_lin_vel_xy_exp",
    "track_ang_vel_z": "track_ang_vel_z_exp",
    "lin_vel_z": "lin_vel_z_l2",
    "ang_vel_xy": "ang_vel_xy_l2",
    "joint_torque": "dof_torques_l2",
    "joint_acc": "dof_acc_l2",
    "action_rate": "action_rate_l2",
    "feet_air_time": "feet_air_time",
}
```

- [ ] **Step 4: Run test and verify it passes.**

```
PYTHONPATH=src pytest tests/test_go2_env_cfg_rewards.py::test_reward_term_map_covers_phoenix_base_keys -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tests/test_go2_env_cfg_rewards.py src/phoenix/sim_env/go2_env_cfg.py
git commit -m "sim_env: add _REWARD_TERM_MAP for reward-section wiring"
```

---

### Task 0.2: Add `_apply_rewards` helper with unknown-key protection

**Files:**
- Modify: `src/phoenix/sim_env/go2_env_cfg.py`
- Modify: `tests/test_go2_env_cfg_rewards.py`

- [ ] **Step 1: Write the failing test.** Append to `tests/test_go2_env_cfg_rewards.py`:

```python
class _FakeRewardTerm:
    """Stand-in for Isaac Lab RewardTermCfg — only `.weight` is exercised."""
    def __init__(self, weight: float):
        self.weight = weight


class _FakeRewards:
    """Attribute-access container matching RewardsCfg's term-as-attr pattern."""
    def __init__(self, **terms):
        for k, v in terms.items():
            setattr(self, k, v)


class _FakeEnvCfg:
    def __init__(self, rewards):
        self.rewards = rewards


def test_apply_rewards_sets_weights() -> None:
    env_cfg = _FakeEnvCfg(
        _FakeRewards(
            action_rate_l2=_FakeRewardTerm(-0.01),
            dof_acc_l2=_FakeRewardTerm(-2.5e-7),
        )
    )
    _apply_rewards(env_cfg, {"action_rate": -0.5, "joint_acc": -1.0e-6})
    assert env_cfg.rewards.action_rate_l2.weight == -0.5
    assert env_cfg.rewards.dof_acc_l2.weight == -1.0e-6


def test_apply_rewards_unknown_key_raises() -> None:
    env_cfg = _FakeEnvCfg(_FakeRewards())
    with pytest.raises(KeyError, match="bogus_term"):
        _apply_rewards(env_cfg, {"bogus_term": -1.0})


def test_apply_rewards_empty_dict_is_noop() -> None:
    env_cfg = _FakeEnvCfg(_FakeRewards(action_rate_l2=_FakeRewardTerm(-0.01)))
    _apply_rewards(env_cfg, {})
    assert env_cfg.rewards.action_rate_l2.weight == -0.01
```

- [ ] **Step 2: Run tests and verify failure.**

```
PYTHONPATH=src pytest tests/test_go2_env_cfg_rewards.py::test_apply_rewards_sets_weights -v
```

Expected: `ImportError: cannot import name '_apply_rewards'`.

- [ ] **Step 3: Add `_apply_rewards` helper.** In `src/phoenix/sim_env/go2_env_cfg.py`, add after `_apply_commands` (~line 142):

```python
def _apply_rewards(env_cfg: Any, rewards: dict[str, Any]) -> None:
    """Apply YAML reward overrides to Isaac Lab env cfg reward term weights.

    For each key in ``rewards``, look up the upstream term name in
    ``_REWARD_TERM_MAP`` and set ``env_cfg.rewards.<term>.weight``. An
    unknown key raises KeyError — this is deliberate, to prevent the
    silent-no-op drift that motivated adding this helper (see
    :mod:`phoenix.sim_env.go2_env_cfg` module docstring, 2026-04-19).
    """
    if not rewards:
        return
    for yaml_key, weight in rewards.items():
        if yaml_key not in _REWARD_TERM_MAP:
            raise KeyError(
                f"Unknown reward key {yaml_key!r} — add it to _REWARD_TERM_MAP "
                f"or remove from YAML. Known keys: {sorted(_REWARD_TERM_MAP)}"
            )
        term_name = _REWARD_TERM_MAP[yaml_key]
        term = getattr(env_cfg.rewards, term_name)
        term.weight = float(weight)
```

- [ ] **Step 4: Run tests and verify they pass.**

```
PYTHONPATH=src pytest tests/test_go2_env_cfg_rewards.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit.**

```bash
git add src/phoenix/sim_env/go2_env_cfg.py tests/test_go2_env_cfg_rewards.py
git commit -m "sim_env: add _apply_rewards helper (raises on unknown keys)"
```

---

### Task 0.3: Wire `_apply_rewards` into `build_env_cfg` and unwire `reward`

**Files:**
- Modify: `src/phoenix/sim_env/go2_env_cfg.py`
- Modify: `tests/test_go2_env_cfg_rewards.py`

- [ ] **Step 1: Write failing test for unwired-warning behavior.** Append to `tests/test_go2_env_cfg_rewards.py`:

```python
def test_reward_no_longer_in_unwired_top_level() -> None:
    """Phase 0 of the 2026-04-19 retrain removes 'reward' from the
    unwired list. 'termination' and the robot sub-keys stay unwired
    (separate PRs)."""
    from phoenix.sim_env.go2_env_cfg import _UNWIRED_TOP_LEVEL

    assert "reward" not in _UNWIRED_TOP_LEVEL
    assert "termination" in _UNWIRED_TOP_LEVEL  # intentionally unchanged


def test_unwired_sections_does_not_flag_reward() -> None:
    unwired = _unwired_sections_present({"reward": {"action_rate": -0.5}})
    assert unwired == []


def test_unwired_sections_still_flags_termination() -> None:
    unwired = _unwired_sections_present(
        {"termination": {"pitch_threshold_rad": 0.8}}
    )
    assert unwired == ["termination"]
```

- [ ] **Step 2: Run tests and verify failure.**

```
PYTHONPATH=src pytest tests/test_go2_env_cfg_rewards.py::test_reward_no_longer_in_unwired_top_level -v
```

Expected: FAIL — `reward` is still in `_UNWIRED_TOP_LEVEL`.

- [ ] **Step 3: Remove `"reward"` from `_UNWIRED_TOP_LEVEL` and wire the call in `build_env_cfg`.** In `src/phoenix/sim_env/go2_env_cfg.py`:

Change line 41:
```python
_UNWIRED_TOP_LEVEL = ("reward", "termination")
```
to:
```python
_UNWIRED_TOP_LEVEL = ("termination",)
```

In `build_env_cfg` (~line 179), after the `_apply_perturbation` call, add:
```python
    _apply_rewards(cfg, data.get("reward", {}))
```

- [ ] **Step 4: Update module docstring (lines 11-22) to reflect the new wired list.** Change:

```
Wired → override upstream defaults:
    env, command, domain_randomization, perturbation, seed

Present in ``base.yaml`` but NOT wired (upstream Go2 defaults win):
    reward, observation.noise, termination, robot.init_state, robot.actuator
```

to:

```
Wired → override upstream defaults:
    env, command, domain_randomization, perturbation, reward, seed

Present in ``base.yaml`` but NOT wired (upstream Go2 defaults win):
    observation.noise, termination, robot.init_state, robot.actuator

Reward wiring added 2026-04-19 (retrain spec Phase 0); prior to this,
YAML reward.* overrides were silent no-ops. This change invalidates
v3b as a reproducible baseline — v3b checkpoint stays as the frozen
reference for comparisons but cannot be re-created from its config.
```

- [ ] **Step 5: Run the full new test file.**

```
PYTHONPATH=src pytest tests/test_go2_env_cfg_rewards.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit.**

```bash
git add src/phoenix/sim_env/go2_env_cfg.py tests/test_go2_env_cfg_rewards.py
git commit -m "sim_env: wire reward section; remove from _UNWIRED_TOP_LEVEL"
```

---

### Task 0.4: Clean up `base.yaml` — remove keys unmappable to upstream

**Files:**
- Modify: `configs/env/base.yaml`

**Why:** Once `_apply_rewards` raises on unknown keys, any `load_layered_config` that hits `base.yaml` will fail because `base.yaml` currently contains 4 reward keys with no upstream term: `base_height`, `joint_vel` (weight is `-0.0`, no-op anyway), `collision`, `termination`. These were aspirational per the existing module docstring. They come out of `base.yaml`; if we later want them, they land in a follow-on PR with matching wiring.

- [ ] **Step 1: Edit `configs/env/base.yaml` reward block (lines 85-98).** Replace:

```yaml
reward:
  # Standard legged-gym style reward shaping.
  track_lin_vel_xy: 1.0
  track_ang_vel_z: 0.5
  base_height: -1.0
  ang_vel_xy: -0.05
  lin_vel_z: -2.0
  joint_acc: -2.5e-7
  joint_vel: -0.0
  joint_torque: -0.0002
  action_rate: -0.05
  feet_air_time: 1.0
  collision: -1.0
  termination: -10.0
```

with:

```yaml
reward:
  # Applied via go2_env_cfg._apply_rewards. Only upstream
  # UnitreeGo2RoughEnvCfg reward term names are accepted — see
  # phoenix.sim_env.go2_env_cfg._REWARD_TERM_MAP. Removed 2026-04-19:
  # base_height, joint_vel, collision, termination (no upstream term;
  # aspirational pre-wiring).
  track_lin_vel_xy: 1.0
  track_ang_vel_z: 0.5
  ang_vel_xy: -0.05
  lin_vel_z: -2.0
  joint_acc: -2.5e-7
  joint_torque: -0.0002
  action_rate: -0.05
  feet_air_time: 1.0
```

- [ ] **Step 2: Run the existing config-loader tests to confirm no regression.**

```
PYTHONPATH=src pytest tests/test_config_loader.py -v
```

Expected: all pass.

- [ ] **Step 3: Run the reward-wiring tests again.**

```
PYTHONPATH=src pytest tests/test_go2_env_cfg_rewards.py -v
```

Expected: 7 passed.

- [ ] **Step 4: Run the entire CI-compatible test suite.** Verify no other test depended on the stripped keys.

```
PYTHONPATH=src pytest tests/ -v --ignore=tests/test_sim_integration.py
```

Expected: all pass. `test_sim_integration` is sim-gated and will be covered by Task 0.5.

- [ ] **Step 5: Commit.**

```bash
git add configs/env/base.yaml
git commit -m "configs: strip unmappable reward keys from base.yaml

base_height, joint_vel, collision, termination have no upstream
UnitreeGo2RoughEnvCfg term. They were aspirational pre-wiring; now
that _apply_rewards raises on unknown keys, they'd break config
load. Removal is part of the reward-section wiring (Phase 0 of the
2026-04-19 retrain plan)."
```

---

### Task 0.5: Sim smoke — 1-iter retrain proves weights propagate

**Files:**
- Produce: `checkpoints/phoenix-stand-v2/<timestamp>/` (1-iter smoke, kept only until verified)

**Why:** Unit tests use fakes. This task runs the actual training loop for 1 iter with a known extreme `action_rate` weight and checks the TensorBoard log shows the weight took effect in the reward stream.

- [ ] **Step 1: Create a smoke-override YAML** at `/tmp/ppo_stand_v2_smoke.yaml`:

```yaml
defaults:
  - ppo_stand_v2

run:
  name: "phoenix-stand-v2-smoke"
  max_iterations: 1
  save_interval: 1
```

Then run 1-iter with the standard weight. (Use `stand_v2.yaml`'s `action_rate: -0.5`.)

```bash
cd ~/workspace/go2-phoenix
./scripts/train.sh configs/train/ppo_stand_v2.yaml \
    --resume checkpoints/phoenix-stand/2026-04-16_22-04-28/model_999.pt \
    --max-iterations 1 \
    --run-name phoenix-stand-v2-smoke-A
```

(If `train.sh` / `ppo_runner` does not accept `--run-name` / `--max-iterations` as CLI overrides, instead copy `ppo_stand_v2.yaml` → `/tmp/ppo_stand_v2_smoke_A.yaml` and edit `max_iterations: 1`, `run.name: phoenix-stand-v2-smoke-A`.)

Expected: finishes in <60 s, writes `checkpoints/phoenix-stand-v2-smoke-A/<ts>/events.out.tfevents.*`.

- [ ] **Step 2: Repeat the smoke with a dialled-down reward weight.** Copy `configs/train/ppo_stand_v2.yaml` → `/tmp/ppo_stand_v2_smoke_B.yaml`, edit:
- `env.config: /tmp/stand_v2_smoke.yaml`
- `run.name: "phoenix-stand-v2-smoke-B"`
- `run.max_iterations: 1`

Create `/tmp/stand_v2_smoke.yaml`:

```yaml
defaults:
  - {{repo}}/configs/env/stand

reward:
  action_rate: -0.01   # upstream default
  joint_acc: -2.5e-7   # upstream default
```

(Substitute the absolute repo path for `{{repo}}`.) Run:

```bash
./scripts/train.sh /tmp/ppo_stand_v2_smoke_B.yaml \
    --resume checkpoints/phoenix-stand/2026-04-16_22-04-28/model_999.pt
```

- [ ] **Step 3: Verify TensorBoard scalars differ between runs.**

```bash
python3 - <<'PY'
from pathlib import Path
import struct, sys

def read_scalar(event_file: Path, tag: str) -> list[tuple[int, float]]:
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(
        str(event_file.parent),
        size_guidance={event_accumulator.SCALARS: 1000},
    )
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        print(f"MISSING TAG {tag} in {event_file.parent}")
        print("Available:", ea.Tags()["scalars"][:40])
        sys.exit(2)
    return [(s.step, s.value) for s in ea.Scalars(tag)]

for run in ["phoenix-stand-v2-smoke-A", "phoenix-stand-v2-smoke-B"]:
    p = next(Path(f"checkpoints/{run}").glob("*/events.out.tfevents.*"))
    print(run, read_scalar(p, "Train/mean_reward_action_rate_l2"))
PY
```

Expected: the per-step mean reward contribution for `action_rate_l2` in smoke-A (weight -0.5) is ≥5× smoke-B's absolute value. If not, weights did not propagate — stop and debug `_apply_rewards`.

- [ ] **Step 4: Clean up smoke artifacts.**

```bash
rm -rf checkpoints/phoenix-stand-v2-smoke-A checkpoints/phoenix-stand-v2-smoke-B
rm /tmp/ppo_stand_v2_smoke_B.yaml /tmp/stand_v2_smoke.yaml
```

- [ ] **Step 5: Record result in vault.** Append to `Vault/Projects/go2-phoenix/RETRAIN_2026-04-19.md` (create the file with the block below if it does not yet exist):

```markdown
---
project: go2-phoenix
type: retrain
date: 2026-04-19
---

# Phoenix Retrain 2026-04-19

## Phase 0 — reward-section wiring
- _apply_rewards + _REWARD_TERM_MAP added in sim_env/go2_env_cfg.py
- base.yaml stripped of 4 unmappable keys
- 1-iter smoke: weight propagation confirmed. action_rate mean reward
  contribution smoke-A (w=-0.5) vs smoke-B (w=-0.01) ratio = <X>.
- Tests: 7/7 reward-wiring + full CI suite green.
```

(Fill in the ratio from Step 3 output.)

- [ ] **Step 6: Commit the vault note.** (Run in the vault directory.)

```bash
cd "/home/yusuf/Documents/Obsidian Vault"
git add "Projects/go2-phoenix/RETRAIN_2026-04-19.md"
git commit -m "vault: phoenix retrain 2026-04-19 — Phase 0 smoke note"
```

(No commit to the main repo yet; Phase 0 is code-complete after Task 0.4.)

---

## Phase 1 — `phoenix-stand-v2` fine-tune

### Task 1.1: Run stand-v2 fine-tune (seed 42)

**Files:**
- Produce: `checkpoints/phoenix-stand-v2/<timestamp>/`

- [ ] **Step 1: Confirm baseline checkpoint exists.**

```bash
ls -l checkpoints/phoenix-stand/2026-04-16_22-04-28/model_999.pt
```

Expected: file present, ~4.5 MB.

- [ ] **Step 2: Launch training in a logged foreground session.**

```bash
cd ~/workspace/go2-phoenix
./scripts/train.sh configs/train/ppo_stand_v2.yaml \
    --resume checkpoints/phoenix-stand/2026-04-16_22-04-28/model_999.pt \
    2>&1 | tee /tmp/phoenix-stand-v2-seed42.log
```

Expected: ~15 min wall clock on RTX 5070 at 10240 envs. Final line lists the output dir under `checkpoints/phoenix-stand-v2/<timestamp>/`.

- [ ] **Step 3: Record the output dir.**

```bash
RUN_DIR=$(ls -dt checkpoints/phoenix-stand-v2/2026-* | head -1)
echo "RUN_DIR=$RUN_DIR"
ls "$RUN_DIR"/model_*.pt | tail -3
```

Save `RUN_DIR` into the session log — later tasks reference it.

- [ ] **Step 4: Quick sanity check — reward curve didn't collapse.**

```bash
python3 - <<PY
from tensorboard.backend.event_processing import event_accumulator
import pathlib, os
run = os.environ["RUN_DIR"]
ea = event_accumulator.EventAccumulator(run, size_guidance={event_accumulator.SCALARS: 10000})
ea.Reload()
rew = [s.value for s in ea.Scalars("Train/mean_reward")]
print(f"iters={len(rew)}, first={rew[0]:.3f}, last={rew[-1]:.3f}, min={min(rew):.3f}, max={max(rew):.3f}")
PY
```

Expected: `last` ≥ `first` (or at least not a cliff). If reward collapsed, fail-closed and go to Task 1.2-alt (seed sweep) rather than exporting a regressed policy.

---

### Task 1.2: Phase 1 gates G1 + G3 (sim survival + slew saturation)

**Files:**
- Produce: `docs/rollout_stand_v2_2026-04-19.json`

- [ ] **Step 1: Run the stand.yaml rollout with slew gate.**

```bash
cd ~/workspace/go2-phoenix
python3 -m phoenix.training.evaluate \
    --checkpoint "$RUN_DIR/model_999.pt" \
    --env-config configs/env/stand.yaml \
    --num-envs 16 --num-episodes 32 \
    --slew-saturation-max 0.05 \
    --metrics-out docs/rollout_stand_v2_2026-04-19.json
```

Expected exit 0. Inspect JSON output:

```bash
python3 -c "import json; d=json.load(open('docs/rollout_stand_v2_2026-04-19.json')); print(d)"
```

Required fields:
- `success_rate == 1.0` (32/32)
- `mean_episode_length_s == 20.0`
- `slew_saturation_pct < 0.05`

If any fail, go to Task 1.5 (seed fallback). Do NOT export a failed policy.

- [ ] **Step 2: Commit the metrics JSON.**

```bash
git add docs/rollout_stand_v2_2026-04-19.json
git commit -m "stand-v2 seed 42: rollout metrics (stand.yaml, G1+G3 PASS)"
```

---

### Task 1.3: Phase 1 gate G6 — export ONNX + parity

**Files:**
- Produce: `$RUN_DIR/policy.onnx`, `$RUN_DIR/policy.onnx.data`, `$RUN_DIR/policy.pt`

- [ ] **Step 1: Export ONNX.**

```bash
python3 -m phoenix.sim2real.export_onnx \
    --checkpoint "$RUN_DIR/model_999.pt" \
    --out "$RUN_DIR/policy.onnx"
```

Expected: `$RUN_DIR/policy.onnx` + `policy.onnx.data` + `policy.pt` written. `policy.onnx` mtime is current minute.

- [ ] **Step 2: Run parity gate against an existing parquet.**

```bash
python3 -m phoenix.sim2real.verify_deploy \
    --parquet data/failures/synth_slippery_trained.parquet \
    --deploy-cfg configs/sim2real/deploy.yaml \
    --onnx-path "$RUN_DIR/policy.onnx" \
    --tol 1e-4 \
    --max-steps 200
```

(If `verify_deploy` does not accept `--onnx-path`, instead temporarily patch `configs/sim2real/deploy.yaml` `policy.onnx_path` to the new path for this step, then restore.)

Expected: exit 0, max_diff <1e-4.

If parity fails, ONNX is divergent — do not ship. Re-export and re-run; if still fails, flag as a blocker.

---

### Task 1.4: Stage stand-v2 as the Gate 7 policy + T7 sync

**Files:**
- Modify: `checkpoints/phoenix-flat/policy.onnx`, `checkpoints/phoenix-flat/policy.onnx.data`, `checkpoints/phoenix-flat/policy.pt` (Gate 7 staging slot)

- [ ] **Step 1: Stage stand-v2 ONNX as the Gate 7 candidate.** The deploy.yaml hardcodes `checkpoints/phoenix-flat/policy.onnx`; staging = file copy, no config change.

```bash
cp "$RUN_DIR/policy.onnx"      checkpoints/phoenix-flat/policy.onnx
cp "$RUN_DIR/policy.onnx.data" checkpoints/phoenix-flat/policy.onnx.data
cp "$RUN_DIR/policy.pt"        checkpoints/phoenix-flat/policy.pt
```

- [ ] **Step 2: Re-run the parity gate against the staged path.**

```bash
python3 -m phoenix.sim2real.verify_deploy \
    --parquet data/failures/synth_slippery_trained.parquet \
    --deploy-cfg configs/sim2real/deploy.yaml \
    --tol 1e-4 \
    --max-steps 200
```

Expected: PASS. This confirms the staged copies are byte-identical to what was parity-gated in Task 1.3.

- [ ] **Step 3: Rsync to T7 (exFAT — use `-L` to dereference any symlinks).**

```bash
rsync -avL \
    "$RUN_DIR/" \
    "/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-stand-v2/$(basename $RUN_DIR)/"

rsync -avL \
    checkpoints/phoenix-flat/policy.onnx \
    checkpoints/phoenix-flat/policy.onnx.data \
    checkpoints/phoenix-flat/policy.pt \
    "/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat/"
```

Expected: both rsyncs show "sent XX bytes, received YY bytes", no errors. No `--delete` on the run dir — keep prior runs on T7 as history.

- [ ] **Step 4: Write Phase 1 post-mortem `docs/retrain_stand_v2_2026-04-19.md`:**

```markdown
# stand-v2 fine-tune — 2026-04-19

## Summary
Fine-tuned `phoenix-stand/2026-04-16_22-04-28/model_999.pt` on
`configs/env/stand_v2.yaml` (+10× action_rate, +4× joint_acc, init
noise 0.1) for 500 iters, seed 42. Exported ONNX staged as the Gate 7
candidate at `checkpoints/phoenix-flat/policy.onnx`.

## Run dir
`<$RUN_DIR>`

## Gates
| # | Gate | Threshold | Result |
|---|---|---|---|
| G1 | 32/32 stand.yaml | 20.0 s | {from JSON} |
| G3 | slew_saturation_pct @ cmd=0 | <0.05 | {from JSON} |
| G6 | ONNX parity | <1e-4 | {from verify_deploy} |

## Baseline comparison (stand-v1 2026-04-16_22-04-28 vs stand-v2)
|                              | stand-v1 | stand-v2 |
|------------------------------|---|---|
| slew_saturation_pct          | {sim, or "not measured"} | {JSON} |
| mean_episode_return          | {prior JSON if available} | {JSON} |

## Staging
- stand-v2 ONNX staged at `checkpoints/phoenix-flat/policy.onnx`
  (Gate 7 candidate)
- rsync'd to T7 at the same path
- v3b artifacts retained at `checkpoints/phoenix-flat/v3b/`
```

Fill in the `{}` fields from the JSON + verify_deploy output.

- [ ] **Step 5: Commit post-mortem.**

```bash
git add docs/retrain_stand_v2_2026-04-19.md
git commit -m "docs: stand-v2 fine-tune post-mortem (2026-04-19)"
```

---

### Task 1.5 (conditional): Seed fallback if Task 1.2 failed

**Only run if Task 1.2 gates failed on seed 42.**

**Files:**
- Produce: `checkpoints/phoenix-stand-v2/<ts_seed7>/`, `<ts_seed123>/`
- Produce: `docs/rollout_stand_v2_seed<n>_2026-04-19.json`

- [ ] **Step 1: Copy `configs/train/ppo_stand_v2.yaml` → `configs/train/ppo_stand_v2_seed7.yaml`.** Change `run.name: "phoenix-stand-v2-seed7"`, `run.seed: 7`.

- [ ] **Step 2: Train seed 7.** (Same invocation as Task 1.1 Step 2 but the seed-7 config.)

- [ ] **Step 3: Run Task 1.2 gates on seed 7.** Write metrics to `docs/rollout_stand_v2_seed7_2026-04-19.json`.

- [ ] **Step 4: If seed 7 passes, use it for staging (Tasks 1.3–1.4).** Otherwise repeat for seed 123.

- [ ] **Step 5: If none of {42, 7, 123} passes all gates, STOP.** Do not ship. Update `Vault/Projects/go2-phoenix/RETRAIN_2026-04-19.md` with the failure and reconsider the design (likely the reward weights need further tuning — revisit in a new spec).

---

## Phase 2 — `phoenix-flat-v3b-ft` fine-tune

### Task 2.1: Create `flat_v3b_ft.yaml` env config

**Files:**
- Create: `configs/env/flat_v3b_ft.yaml`

- [ ] **Step 1: Write the env config.** Create `configs/env/flat_v3b_ft.yaml`:

```yaml
# Flat-v3b fine-tune — smoothness recipe + modest cmd=0 exposure.
#
# v3b (checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt) passed
# sim but saturated ±0.175 rad/step slew clip on 30.23% of motor steps
# at cmd_vel=0 on GO2 hardware (2026-04-18). This fine-tune mirrors the
# stand_v2 action-smoothness knobs and bumps rel_standing_envs
# 0.02 -> 0.10 so the policy actually sees cmd=0 during training.
#
# This is NOT the v4 over-reach: v4 went to rel_standing_envs=0.15 AND
# doubled init_noise AND doubled entropy, which wrecked yaw tracking
# (ang_vel_error +67% vs v3b). See commit 61fae38 for v4 post-mortem.
#
# Evaluation criterion (from 2026-04-19 retrain spec Phase 2, gate A):
# - slew_saturation_pct < 0.05 at cmd=0 on stand.yaml
# - lin_vel_err <= 0.10 m/s on flat.yaml
# - ang_vel_err <= 0.10 rad/s on flat.yaml
# - 32/32 success on both stand.yaml and flat.yaml

defaults:
  - flat

reward:
  action_rate: -0.5     # was -0.05 (base.yaml); 10x stronger
  joint_acc: -1.0e-6    # was -2.5e-7 (base.yaml); 4x stronger

command:
  # v3b trained with base's 0.02 (effectively no cmd=0 samples).
  # v4 jumped to 0.15 and over-corrected. 0.10 = 5x more cmd=0 exposure
  # than v3b without flooding the distribution.
  rel_standing_envs: 0.10
```

- [ ] **Step 2: Confirm it loads.**

```bash
PYTHONPATH=src python3 -c "
from phoenix.sim_env.config_loader import load_layered_config
cfg = load_layered_config('configs/env/flat_v3b_ft.yaml').to_container()
print('reward.action_rate =', cfg['reward']['action_rate'])
print('reward.joint_acc =', cfg['reward']['joint_acc'])
print('command.rel_standing_envs =', cfg['command']['rel_standing_envs'])
"
```

Expected output:
```
reward.action_rate = -0.5
reward.joint_acc = -1.0e-06
command.rel_standing_envs = 0.1
```

- [ ] **Step 3: Commit.**

```bash
git add configs/env/flat_v3b_ft.yaml
git commit -m "configs: add flat_v3b_ft env overlay (+10x action_rate, rel_standing 0.10)"
```

---

### Task 2.2: Create `ppo_flat_v3b_ft.yaml` train config

**Files:**
- Create: `configs/train/ppo_flat_v3b_ft.yaml`

- [ ] **Step 1: Write the train config.** Create `configs/train/ppo_flat_v3b_ft.yaml`:

```yaml
# PPO fine-tune config for flat-v3b-ft.
#
# Fine-tunes from checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt
# (v3b). Invoke with --resume pointed at that path. 500 iters is
# generous for a smoothness refinement; v3b plateaued on tracking at
# ~iter 100 with the same hyperparameters.
#
# entropy_coef stays at 0.005 — v4's bump to 0.01 was a mistake.

run:
  name: "phoenix-flat-v3b-ft"
  output_dir: "checkpoints"
  log_interval: 1
  save_interval: 50
  max_iterations: 500
  seed: 42
  device: "cuda:0"

env:
  config: "configs/env/flat_v3b_ft.yaml"

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
  # Quieter mean-action baseline for fine-tuning — v3b already knows
  # how to track velocities; we're tightening smoothness, not
  # re-exploring.
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

- [ ] **Step 2: Commit.**

```bash
git add configs/train/ppo_flat_v3b_ft.yaml
git commit -m "configs: add ppo_flat_v3b_ft fine-tune config"
```

---

### Task 2.3: Run flat-v3b-ft fine-tune (seed 42)

**Files:**
- Produce: `checkpoints/phoenix-flat-v3b-ft/<timestamp>/`

- [ ] **Step 1: Verify v3b baseline.**

```bash
ls -l checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt
```

Expected: ~4.5 MB file.

- [ ] **Step 2: Launch training.**

```bash
cd ~/workspace/go2-phoenix
./scripts/train.sh configs/train/ppo_flat_v3b_ft.yaml \
    --resume checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt \
    2>&1 | tee /tmp/phoenix-flat-v3b-ft-seed42.log
```

Expected: ~15 min on RTX 5070.

- [ ] **Step 3: Record the output dir.**

```bash
FT_RUN_DIR=$(ls -dt checkpoints/phoenix-flat-v3b-ft/2026-* | head -1)
echo "FT_RUN_DIR=$FT_RUN_DIR"
ls "$FT_RUN_DIR"/model_*.pt | tail -3
```

- [ ] **Step 4: Reward curve sanity check.** (Same as Task 1.1 Step 4 but against `$FT_RUN_DIR`.)

---

### Task 2.4: Phase 2 gates G1-G5

**Files:**
- Produce: `docs/rollout_flat_v3b_ft_stand_2026-04-19.json`
- Produce: `docs/rollout_flat_v3b_ft_flat_2026-04-19.json`

- [ ] **Step 1: Rollout on stand.yaml (G1 survival + G3 slew).**

```bash
python3 -m phoenix.training.evaluate \
    --checkpoint "$FT_RUN_DIR/model_999.pt" \
    --env-config configs/env/stand.yaml \
    --num-envs 16 --num-episodes 32 \
    --slew-saturation-max 0.05 \
    --metrics-out docs/rollout_flat_v3b_ft_stand_2026-04-19.json
```

Expected: exit 0, `success_rate == 1.0`, `mean_episode_length_s == 20.0`, `slew_saturation_pct < 0.05`.

- [ ] **Step 2: Rollout on flat.yaml (G2 survival + G4 + G5 tracking).**

```bash
python3 -m phoenix.training.evaluate \
    --checkpoint "$FT_RUN_DIR/model_999.pt" \
    --env-config configs/env/flat.yaml \
    --num-envs 16 --num-episodes 32 \
    --metrics-out docs/rollout_flat_v3b_ft_flat_2026-04-19.json
```

Inspect:
```bash
python3 -c "import json; d=json.load(open('docs/rollout_flat_v3b_ft_flat_2026-04-19.json')); print(d)"
```

Required:
- `success_rate == 1.0`
- `mean_episode_length_s == 20.0`
- `mean_lin_vel_error <= 0.10`
- `mean_ang_vel_error <= 0.10`

If any gate fails, go to Task 2.7 (seed fallback) — do not export.

- [ ] **Step 3: Commit metrics JSONs.**

```bash
git add docs/rollout_flat_v3b_ft_stand_2026-04-19.json \
        docs/rollout_flat_v3b_ft_flat_2026-04-19.json
git commit -m "flat-v3b-ft seed 42: rollout metrics (G1-G5 PASS)"
```

---

### Task 2.5: Phase 2 gate G6 — export ONNX + parity

**Files:**
- Produce: `$FT_RUN_DIR/policy.onnx`, `policy.onnx.data`, `policy.pt`

- [ ] **Step 1: Export ONNX.**

```bash
python3 -m phoenix.sim2real.export_onnx \
    --checkpoint "$FT_RUN_DIR/model_999.pt" \
    --out "$FT_RUN_DIR/policy.onnx"
```

- [ ] **Step 2: Parity gate.**

```bash
python3 -m phoenix.sim2real.verify_deploy \
    --parquet data/failures/synth_slippery_trained.parquet \
    --deploy-cfg configs/sim2real/deploy.yaml \
    --onnx-path "$FT_RUN_DIR/policy.onnx" \
    --tol 1e-4 \
    --max-steps 200
```

(Same fallback as Task 1.3 Step 2 if `--onnx-path` is not a CLI option — temporarily patch + restore `deploy.yaml`.)

Expected: max_diff <1e-4.

---

### Task 2.6: Stage flat-v3b-ft as Gate 8 candidate + T7 sync

**Files:**
- Create: `checkpoints/phoenix-flat/v3b-ft/policy.onnx`, `.onnx.data`, `.pt`

- [ ] **Step 1: Stage in a sibling dir (do not overwrite stand-v2 staging).**

```bash
mkdir -p checkpoints/phoenix-flat/v3b-ft
cp "$FT_RUN_DIR/policy.onnx"      checkpoints/phoenix-flat/v3b-ft/policy.onnx
cp "$FT_RUN_DIR/policy.onnx.data" checkpoints/phoenix-flat/v3b-ft/policy.onnx.data
cp "$FT_RUN_DIR/policy.pt"        checkpoints/phoenix-flat/v3b-ft/policy.pt
```

- [ ] **Step 2: Update `latest.pt` symlink** to the new flat fine-tune's final checkpoint (the symlink currently points at v3b's `model_999.pt`).

```bash
cd checkpoints/phoenix-flat
ln -sfn "$(realpath --relative-to=. "$FT_RUN_DIR/model_999.pt")" latest.pt
ls -l latest.pt
cd -
```

Expected: `latest.pt -> phoenix-flat-v3b-ft/<timestamp>/model_999.pt`.

- [ ] **Step 3: Rsync new artifacts to T7.**

```bash
rsync -avL \
    "$FT_RUN_DIR/" \
    "/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat-v3b-ft/$(basename $FT_RUN_DIR)/"

rsync -avL \
    checkpoints/phoenix-flat/v3b-ft/ \
    "/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat/v3b-ft/"
```

- [ ] **Step 4: Write Phase 2 post-mortem `docs/retrain_flat_v3b_ft_2026-04-19.md`** — same structure as `docs/retrain_stand_v2_2026-04-19.md` but with both rollout JSONs and the v3b-vs-flat-v3b-ft side-by-side table (mirror the format from commit `61fae38`).

- [ ] **Step 5: Commit post-mortem.**

```bash
git add docs/retrain_flat_v3b_ft_2026-04-19.md
git commit -m "docs: flat-v3b-ft fine-tune post-mortem (2026-04-19)"
```

---

### Task 2.7 (conditional): Seed fallback if Task 2.4 failed

**Only run if Task 2.4 gates failed on seed 42.**

Same structure as Task 1.5 but for flat-v3b-ft: copy `ppo_flat_v3b_ft.yaml` → `_seed7.yaml` / `_seed123.yaml`, retrain, re-gate, use first passing seed for staging. If none of {42, 7, 123} pass, STOP.

---

## Finalization

### Task 3.1: Update vault + memory

**Files:**
- Modify: `Vault/Projects/go2-phoenix/status.md`
- Modify: `Vault/Projects/go2-phoenix/RETRAIN_2026-04-19.md`
- Modify: `~/.claude/projects/-home-yusuf/memory/project_go2_phoenix.md`
- Create: `Vault/Projects/go2-phoenix/SESSION_2026-04-19.md` (optional)

- [ ] **Step 1: Update `Vault/Projects/go2-phoenix/status.md`.** Flip the current-state block from "Gate 7 BLOCKED on saturation gate" to "Gate 7 ready — retrained stand-v2 specialist; hw dryrun pending at next CaresLab session." Append a Recent Changes row for 2026-04-19 with one-line summary of Phase 0/1/2 outcomes.

- [ ] **Step 2: Append Phase 1 + Phase 2 sections to `Vault/Projects/go2-phoenix/RETRAIN_2026-04-19.md`.** Copy the before/after table from both post-mortems; note the Gate 7 staging decision (stand-v2 at `phoenix-flat/policy.onnx`, flat-v3b-ft at `phoenix-flat/v3b-ft/`).

- [ ] **Step 3: Update `project_go2_phoenix.md` memory.** Change status line to "stand-v2 + flat-v3b-ft retrained 2026-04-19 (phoenix-flat policy = stand-v2 ONNX); Gate 7 ready for CaresLab."

- [ ] **Step 4: Commit vault.**

```bash
cd "/home/yusuf/Documents/Obsidian Vault"
git add "Projects/go2-phoenix/status.md" "Projects/go2-phoenix/RETRAIN_2026-04-19.md"
git commit -m "vault: phoenix retrain 2026-04-19 complete — Gate 7 ready"
```

---

### Task 3.2: Final report to user

- [ ] **Step 1: Print a summary to the session log.** Table per the 2026-04-18 status.md format (one row per run, columns: G1, G2, G3, G4, G5, G6 — PASS/FAIL + the numeric value).

- [ ] **Step 2: Point at `PHOENIX_NEXT_STEPS.md §1` as the next actionable step on lab day.**

---

## Phase 0 self-review checklist

- [ ] Spec coverage: Phase 0 (4 tasks + smoke) → Task 0.1-0.5. Phase 1 Runs → Task 1.1-1.4 + fallback 1.5. Phase 2 Runs → Task 2.1-2.6 + fallback 2.7. Finalization → 3.1-3.2. Gates G1-G6 mapped to concrete evaluate/verify_deploy invocations with numeric thresholds. ✓
- [ ] No placeholders: exact file paths everywhere, full YAML bodies, exact commands, expected outputs. One `{}` template in Task 1.4 step 4 (post-mortem table) — meant to be filled in at execution time from the JSON outputs, not a plan gap. ✓
- [ ] Type consistency: `_REWARD_TERM_MAP` / `_apply_rewards` names consistent across 0.1/0.2/0.3. `$RUN_DIR` / `$FT_RUN_DIR` env vars consistent from Task 1.1 onward. ✓
- [ ] Commit discipline: each task ends in a commit; phase 0 splits into 4 small commits rather than one big blob. ✓
