# go2-phoenix

**Closed-loop sim-to-real learning for the Unitree GO2 quadruped.**

For on-robot deployment, see [`go2_phoenix_deploy_prompt.txt`](./go2_phoenix_deploy_prompt.txt).

The Phoenix loop trains a locomotion policy in simulation, deploys it to the
real robot, captures the failures that happen on hardware, replays those
failures in simulation under a randomized physics sweep, and fine-tunes the
policy on that failure-seeded distribution. The improved policy goes back
to the robot. Every stage of the loop is a concrete Python module with its
own CLI, configuration, and (where possible) unit tests.

![Phoenix architecture](docs/architecture.svg)

```
SIM train  ──▶  ONNX export  ──▶  ROS 2 deploy  ──▶  GO2 hardware
    ▲                                                      │
    │                                                      ▼
    │                                        failure detector + parquet log
    │                                                      │
    └────── fine-tune (failure curriculum) ◀── replay w/ Halton variations
```

## Current results (2026-04-19)

### Hardware-candidate policies

Three policies are currently exported + parity-gated for live deployment.
The two-policy deploy-layer mode switch (shipped 2026-04-19) lets the
ROS 2 node route between them on a `cmd_vel` magnitude threshold, so
the robot no longer has to pick one.

| Candidate | Task | Config | Status |
|---|---|---|---|
| `phoenix-stand-v2/latest.pt` | stand-in-place (cmd=0) | `configs/train/ppo_stand_v2.yaml` | **Gate 7 candidate**, sim slew 0.00254 @ cmd=0 (89× under the 2026-04-18 hardware failure) |
| `phoenix-flat/v3b/` | flat-terrain velocity tracking | `configs/train/ppo_flat.yaml` | **Gate 8 walking half**, sim lin_err 0.091 / ang_err 0.087 |
| `mode_switch` runtime | stand-v2 + v3b, hysteresis + 25-tick blend | `configs/sim2real/deploy.yaml` `policy.mode_switch` | **Gate 8 path B, shipped** — opt-in flag, zero retraining, 179 unit tests green |

The 2026-04-18 hardware dryrun saturated the per-step slew clip at
30.23% specifically when `cmd_vel = (0, 0, 0)`. Stand-v2 solves that
directly in sim; the mode switch preserves v3b for all nonzero commands
while delegating the zero-cmd regime to stand-v2.

### 2026-04-19 session summary — how the single-policy path got exhausted

Four attempts at a single v3b replacement, all with `slew_sat_hinge_l2`
as the new reward term, all sharing the same tracking-collapse outcome:

| run | init | iters | slew_sat | lin_err | ang_err | verdict |
|---|---|---:|---:|---:|---:|---|
| v3b baseline | scratch (pre-hinge) | 1000 | 0.302 hw | **0.091** | **0.087** | reference |
| `flat-v3b-ft` | fine-tune v3b | 500 | 0.341 | 0.619 | 0.435 | ❌ negative |
| `flat-slewhinge` (w=-50) | fine-tune v3b | 500 | 0.123 | 0.623 | 0.607 | ❌ negative |
| `flat-slewhinge-w5` (w=-5) | fine-tune v3b | 500 | 0.186 | 0.572 | 0.576 | ❌ negative |
| `flat-scratch` (w=-50) | scratch | 2500 | **0.00254** | 0.579 | 0.658 | ❌ negative |

The flat-scratch run disproved the fine-tune-destabilization
hypothesis: scratch vs fine-tune, init_noise_std 0.1 vs 0.5, 500 vs
2500 iters all converge to the same ~0.57–0.66 m/s lin_err band with
`slew_sat_hinge @ w=-50`. Root cause is **reward-landscape dominance**
(expected return from minimizing motion exceeds expected return from
tracking), not init conditioning. Full analysis at
[`docs/retrain_flat_scratch_2026-04-19.md`](docs/retrain_flat_scratch_2026-04-19.md);
the mode-switch design that followed lives at
[`docs/superpowers/specs/2026-04-19-phoenix-gate8-mode-switch-design.md`](docs/superpowers/specs/2026-04-19-phoenix-gate8-mode-switch-design.md).

### Pre-lab gates — phoenix-stand (2026-04-17)

| Gate | Metric | Result |
|---|---|---:|
| 0a — sim rollout | success @ 20.0 s mean length | **16 / 16** |
| 0b — ONNX staging | hashes match deploy path | ✓ |
| 0c — verify_deploy parity | max torch↔ort abs-diff | **3.8e-06** (26× under 1e-4 tol) |

Metrics serialized at `docs/pre_lab_gates_2026-04-17.md` +
`docs/pre_lab_stand_rollout_2026-04-17.json`. Lab-day §1–§6 (Jetson
offline gates, three-bridge bringup, fail-closed dry-run, Gate 7 live
stand ×3) are the remaining hardware steps.

### Flat-v0 training — v3b (final, shipped) vs v4 (negative, not shipped)

`phoenix.training.evaluate` on `configs/env/flat.yaml`, 16 envs × 32
episodes, after the 2026-04-18 warp-array tracking-error fix:

| Metric | v3b (Gate 8 candidate) | v4 (superseded) |
|---|---:|---:|
| mean_episode_return | **39.50** | 34.50 |
| mean_lin_vel_error (m/s) | **0.091** | 0.110 |
| mean_ang_vel_error (rad/s) | **0.087** | 0.145 |
| success_rate | 32 / 32 | 32 / 32 |

v4 tried higher entropy (0.005 → 0.01), wider init-noise (0.5 → 1.0),
and a stronger canonical-stand attractor (rel_standing_envs 0.02 → 0.15)
at 5000 iters / 10240 envs. Training converged cleanly but the policy
traded velocity-tracking quality for no measurable robustness gain.
Full post-mortem at
[`checkpoints/phoenix-flat-v4/NEGATIVE_RESULT.md`](checkpoints/phoenix-flat-v4/NEGATIVE_RESULT.md).

### Historical — baseline + fine-tune adaptation (2026-04-14)

| Policy | Terrain | Mean return | Success | Episodes |
|---|---|---:|---:|---:|
| `model_499.pt` (final, rough-v0, 500 iters) | rough    | 18.95 | 100% | 16 |
| `phoenix-base/latest.pt`                    | slippery | 15.90 | 90.6%  | 64 |
| `phoenix-adapt/latest.pt`                   | slippery | **16.64** | **100%** | 64 |
| `phoenix-adapt/latest.pt`                   | rough    | 17.56 | 96.9% | 64 |

The rough-v0 baseline showed 99.5% per-step slew saturation in the
2026-04-14 hardware-adjacent dryrun (235-obs pipeline, proprio + height
scan) and was retired from the deploy path in favor of flat-v0.

> **What the adapt result is and isn't.** The adaptation is plain
> warm-start PPO on the ``slippery.yaml`` overlay (low friction). It is
> **not** a failure-curriculum result — ``configs/train/adaptation.yaml``
> ships with ``failure_sample_fraction: 0.0`` because the failure
> reset-bridge has no hardware-captured parquets to validate against
> yet. A smoke config (``configs/train/adaptation_smoke.yaml``)
> exercises the curriculum plumbing with synthesized parquets but is
> not the producer of the numbers above. See "Known limitations" below.

### Sim-to-sim artifacts

* 500-iter PPO rough-v0 baseline on `Isaac-Velocity-Rough-Unitree-Go2-v0`,
  4096 envs, RTX 5070, ~28 min; 200-iter fine-tune adds ~11 min.
* 2500-iter flat-v0 (v3b) on `Isaac-Velocity-Flat-Unitree-Go2-v0`,
  4096 envs, RTX 5070, ~45 min; serves as Gate 8 candidate.
* Stand-only policy on `configs/train/ppo_stand.yaml`, safer first
  hardware task; Gate 7 candidate.
* Every deployed ONNX passes the `verify_deploy` parity gate
  (torch↔onnxruntime max abs-diff < 1e-4 on 200 sim steps). Current
  stand candidate: 3.8e-06.
* Failure parquet synthesizer produces 200-step rollouts with
  attitude/collapse/slip flags. ``replay/reconstruct.py`` spawns N sim
  envs from the logged initial state and applies a Halton-sampled
  per-env perturbation (mass, push velocity, push yaw) — exercised in
  unit tests against the pure-numpy translation in
  ``replay/apply_variations.py``; the Isaac Sim hand-off itself is
  sim-only, so it cannot run in CI.
* Side-by-side demo videos:
  [`media/side_by_side.mp4`](media/side_by_side.mp4) (training progress)
  and [`media/side_by_side_adapt.mp4`](media/side_by_side_adapt.mp4)
  (baseline on slippery | baseline on rough | adapted on slippery).

---

## Why this repo exists

Most open-source quadruped RL projects stop at "trained in sim, deployed
once." Phoenix is explicitly about the *loop that happens after the first
deployment*: reproducing real failures in sim, using them as training
seeds, and shipping a better policy. The full pipeline is automated by
five shell scripts and driven by YAML configs.

---

## Repository layout

```
configs/            layered YAML: env + train + adapt + replay + deploy
scripts/            train.sh · deploy.sh · replay.sh · adapt.sh · demo.sh
src/phoenix/
    sim_env/        GO2 env factory on top of Isaac Lab's rough-terrain task
    training/       PPO (rsl_rl) + evaluation rollouts
    sim2real/       ONNX export (with parity check), ROS 2 policy node
    real_world/     rule-based failure detector, Parquet trajectory logger
    replay/         Halton variation sampler + Isaac Sim reconstruction
    adaptation/     failure-curriculum fine-tuning
    demo/           side-by-side video pipeline (ffmpeg)
tests/              unit tests (pure-python pieces; run in CI)
docs/               architecture diagram + design notes
docker/             CPU-only testbox for CI
```

---

## Quick start

```bash
# 1. Install Isaac Lab 3.0+ (https://isaac-sim.github.io/IsaacLab/).
export ISAACLAB_PATH=$HOME/IsaacLab

# 2. Train a baseline policy (~4 h on RTX 5070 at 4096 envs)
./scripts/train.sh configs/train/ppo.yaml

# 3. Export to ONNX and deploy on the GO2
./scripts/deploy.sh checkpoints/phoenix-base/latest.pt

# 4. After recording failures on the real robot, replay one in sim
./scripts/replay.sh data/failures/attitude_2026_04_12.parquet

# 5. Fine-tune with the failure curriculum
./scripts/adapt.sh configs/train/adaptation.yaml

# 6. Record the side-by-side demo video
./scripts/demo.sh \
    checkpoints/phoenix-base/latest.pt \
    checkpoints/phoenix-adapt/latest.pt \
    media/real_clip.mp4
```

---

## Python environment boundary

Two Python contexts, one filesystem:

| Context | Installed in | Modules that import from it |
|---|---|---|
| **Isaac Lab Python** | `$ISAACLAB_PATH/isaaclab.sh -p` | `sim_env`, `training`, `replay`, `adaptation`, `demo.benchmark`, `sim2real.export` |
| **System Python + ROS 2** | `/opt/ros/humble` + `pip install .[real]` | `sim2real.ros2_policy_node`, `real_world`, `demo.video_compose` |

Data crosses the boundary as files: `*.onnx`, `*.parquet`, `*.mp4`. No
module imports `torch` *and* `rclpy` — the two contexts never share a
process.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests -m "not sim and not ros"
```

132 unit tests cover the config loader, observation builder, failure
detector, trajectory logger, Parquet round-trip, Halton variation
sampler, curriculum scheduler, ffmpeg escape helper, per-env variation
translation feeding ``reconstruct.py``, the fail-closed estop / sensor
freshness predicates used by the deploy nodes, the projected-gravity
helper consistency between the policy node and the parity gate, the
`verify_deploy` parity gate itself, the `reset_bridge` quat/pose
conversion, and the lowcmd bridge config builder. Sim and ROS tests
are out of CI scope by design — they run manually on the hardware.

Isaac-Lab integration tests live at `tests/test_sim_integration.py`
(marked `@pytest.mark.sim`). They instantiate the Phoenix env cfg
against real Isaac Lab and assert that the friction / mass / perturbation
overrides land on the right event terms — run them locally with
`pytest tests -m sim` before trusting a YAML change.

Policy evaluation (`phoenix.training.evaluate`) reports per-term reward
contributions in addition to success/return, so two candidate policies
with the same success rate can be compared on where their gradient
signal actually went (tracking vs stand vs smoothness). The 2026-04-18
pass also added a `flat_perturb` diagnostic env and fixed a warp-array
`flat_tracking_error` computation so rollouts over 16+ envs no longer
silently report tracking-error from env 0 only. These two additions
made the v3b vs v4 comparison in "Current results" possible.

---

## Safety semantics on the deploy path

The real-robot side fails closed by default. ``ros2_policy_node`` and
``lowcmd_bridge_node`` both treat a stale ``/phoenix/estop`` heartbeat
as an asserted estop, not as "OK to keep going." Every gate is a free
function in ``src/phoenix/sim2real/safety.py`` and is unit-tested in
``tests/test_safety.py``:

* **Startup is locked.** The policy node refuses to run inference or
  publish a policy-derived command until it has received a fresh
  ``/phoenix/estop`` heartbeat with ``data == False`` AND fresh
  ``/imu/data`` AND fresh ``/joint_states``. During cold startup with
  any precondition unmet, the node stays SILENT — the bridge's own
  fail-closed watchdog (also fed by ``estop_is_active``) holds the
  motors with the conservative ``hold_kp`` / ``hold_kd`` gains.
* **Past the grace window**, an unmet precondition latches the abort
  with a specific reason (``estop_publisher_missing``,
  ``estop_heartbeat_stale``, ``external_estop``, ``sensor_missing``,
  ``sensor_stale``); the node then publishes the safe default stand
  pose so the bridge can deliberately hold the robot upright.
* **Slew-rate cap is shared.** The policy node and the bridge both call
  ``per_step_clip_array(target, current, MAX_DELTA_PER_STEP_RAD)`` —
  the constant lives in ``safety.py`` and the cap is provably the same
  on both sides.
* **Wireless / joystick deadman**: stale input *or* released button →
  publish ``estop=True`` within one tick (no "last reported held"
  trust).

The relevant knobs live under ``safety:`` in
``configs/sim2real/deploy.yaml``; ``estop_timeout_s`` is read all the
way through to the bridge as well, with a strict resolution order
(CLI flag → YAML → 0.5 s last-resort default). Defaults are deliberate
and tighter than the upstream Unitree examples, not looser.

---

## Known limitations

v0.1 has the Phoenix-loop *architecture* in place and validated
end-to-end in sim, including a measurable warm-start fine-tune on
the slippery-terrain regime. What's still left for v0.2:

* **Real-robot deployment.** `sim2real.ros2_policy_node` ran against
  a live GO2 on 2026-04-18 end-to-end; the deploy chain (bridges,
  estop, parity gates) is hardware-green. Gate 7 (10 s live stand
  ×3) is pending with `phoenix-stand-v2` staged at
  `configs/sim2real/deploy.yaml:policy.onnx_path`. The 2026-04-19
  two-policy mode switch adds a `policy.mode_switch.enabled` flag
  that loads `stand-v2 + v3b` both and routes on `cmd_vel` magnitude;
  flip on at the lab bench per
  [`docs/deploy_mode_switch_runbook.md`](docs/deploy_mode_switch_runbook.md).
  Flat-v0 (obs_dim=48, no scanner) replaced the rough-v0 baseline
  after the latter saturated the per-step slew clip on 99.5% of
  motor-steps in the 2026-04-14 dryrun.
* **Failure-curriculum adaptation.** The `reset_bridge` that re-seeds
  selected envs from real failure parquets is wired (env-origin-relative
  poses, xyzw→wxyz quat conversion) and unit-tested. The headline
  adaptation result above does **not** use it: `adaptation.yaml` ships
  with `failure_sample_fraction: 0.0` until a hardware-captured parquet
  exists. The pipeline can be flipped on with `adaptation_smoke.yaml`,
  but those numbers are plumbing-only.
* **Replay variation application is local-only.** The pure-Python
  variation translation in `replay/apply_variations.py` is unit-tested
  in CI; the Isaac Sim mass / friction / initial-velocity application
  in `replay/reconstruct.py` is exercised on `mewtwo` but does not have
  a hardware-rollout comparison yet.
* **rsl_rl 3.0 iter-0 logging artifact.** Fine-tune from a trained
  baseline now uses `init_at_random_ep_len=False`; without that,
  `runner.learn` emits iter-0 "mean reward ≈ 0" even with byte-exact
  warm-start, because only envs whose pre-seeded episode length is
  near `max_episode_length` actually terminate inside the 24-step
  first rollout. See
  [`docs/adapt_load_debug.md`](docs/adapt_load_debug.md) for the
  diagnostic trail.

---

## Configuration model

YAML files under `configs/` support a Hydra-style `defaults:` chain:

```yaml
# configs/env/slippery.yaml
defaults:
  - base
domain_randomization:
  friction_range: [0.05, 0.4]   # overrides base
```

All configs are serialized into each run's log directory as
`train.yaml` / `env.yaml` so a rollout is fully reproducible from the
artifact alone.

---

## License

MIT — see [LICENSE](LICENSE).
