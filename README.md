# go2-phoenix

**Closed-loop sim-to-real learning for the Unitree GO2 quadruped.**

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

## Current results (2026-04-14)

### Baseline training — 500 iters, rough terrain

| Policy | Terrain | Mean return | Success | Episodes |
|---|---|---:|---:|---:|
| `model_100.pt` (early) | rough     | 14.22 | 81.2% | 16 |
| `model_499.pt` (final) | rough     | **18.95** | **100%** | 16 |

### Fine-tune adaptation — 200 iters, warm-started from baseline

| Policy | Terrain | Mean return | Success | Episodes |
|---|---|---:|---:|---:|
| `phoenix-base/latest.pt`  | slippery | 15.90 | 90.6%  | 64 |
| `phoenix-adapt/latest.pt` | slippery | **16.64** | **100%** | 64 |
| `phoenix-adapt/latest.pt` | rough    | 17.56 | 96.9% | 64 |

Baseline gets rough right (100%) but slips on slippery (90.6%). The
adapted policy trades ~3 pp of rough-terrain success for closing the
entire slip gap on slippery terrain — 100% success, +0.74 in return.
Exactly the shape of improvement the Phoenix loop is designed to
produce.

### Sim-to-sim artifacts

* 500-iter PPO baseline trained on `Isaac-Velocity-Rough-Unitree-Go2-v0`,
  4096 parallel envs, RTX 5070, ~28 min wall time; 200-iter fine-tune
  adds ~11 min.
* ONNX export passes parity check (max torch↔onnxruntime abs-diff 2.98e-6).
* Failure parquet synthesizer produces 200-step rollouts with
  attitude/collapse/slip flags; replay reconstruction runs N perturbed
  variations from the logged initial state.
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

39 unit tests cover the config loader, observation builder, failure
detector, trajectory logger, Parquet round-trip, Halton variation
sampler, curriculum scheduler, and ffmpeg escape helper. Sim and ROS
tests are out of CI scope by design — they run manually on the hardware.

Isaac-Lab integration tests live at `tests/test_sim_integration.py`
(marked `@pytest.mark.sim`). They instantiate the Phoenix env cfg
against real Isaac Lab and assert that the friction / mass / perturbation
overrides land on the right event terms — run them locally with
`pytest tests -m sim` before trusting a YAML change.

---

## Known limitations

v0.1 has the full Phoenix-loop *architecture* in place and validated
end-to-end in sim, including a measurable fine-tune improvement on
the slippery-terrain regime. What's still left for v0.2:

* **Real-robot deployment.** `sim2real.ros2_policy_node` runs but has
  not been exercised against a live GO2 this cycle. The baseline
  policy observes 235 dims (includes rough-terrain height scan); the
  sim2real pipeline will need a real height-scan source or a flat-task
  variant for genuine deployment.
* **Failure-parquet-driven curriculum.** The `reset_bridge` that
  re-seeds selected envs from real failure parquets is wired (env-
  origin-relative poses, xyzw→wxyz quat conversion) but
  `failure_sample_fraction` ships at `0.0` because we do not yet have
  a hardware-captured parquet to avoid contaminating the adapt run
  with synthesized failures. It is an opt-in knob once real GO2
  failures are recorded — see `data/failures/README.md`.
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
