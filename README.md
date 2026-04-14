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
