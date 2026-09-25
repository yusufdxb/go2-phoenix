<div align="center">

# go2-phoenix

**Closed-loop sim-to-real learning for the Unitree GO2 quadruped.**

[![CI](https://github.com/yusufdxb/go2-phoenix/actions/workflows/ci.yml/badge.svg)](https://github.com/yusufdxb/go2-phoenix/actions/workflows/ci.yml)
&nbsp;![Python](https://img.shields.io/badge/python-3.10%2B-blue)
&nbsp;[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
&nbsp;![Robot](https://img.shields.io/badge/robot-Unitree%20GO2-orange)

</div>

---

The **Phoenix loop** is designed to train a locomotion policy in simulation, deploy it to
the real robot, captures the failures that happen on hardware, replays those
failures in simulation under a randomized physics sweep, and fine-tunes the
policy on that failure-seeded distribution, then return the improved policy to
the robot. That full loop has not completed on hardware. Every stage is a Python module with its own CLI,
configuration, and (where possible) unit tests.

<p align="center">
  <img src="docs/architecture.svg" alt="Phoenix architecture" width="90%">
</p>

## Demo

A 60-second walkthrough: thousands of robots training in parallel in Isaac
Sim, the trained policy tracking velocity commands with a live telemetry
overlay, and where the project stands.

<p align="center">
  <a href="https://youtu.be/Nu0oWyJJbEM">
    <img src="https://img.youtube.com/vi/Nu0oWyJJbEM/sddefault.jpg" alt="Watch the Phoenix demo on YouTube" width="640">
  </a>
</p>

## Project status

As of 2026-09-25, Phoenix has two policy tracks. The H25 stand policy reached
live GO2 control for 0.56 s during the September stage F1 attempt and then
latched on a joint-limit fault. No stable loaded stand or robot walking run has
been established. The walking velocity policy has been trained and evaluated
in Isaac Lab and MuJoCo, with no checkpoint cleared for hardware. See the
[walking result](docs/walk_v1_results.md) and the [evidence ledger](EVIDENCE.md).

| Stage | Current evidence |
|---|---|
| H25 stand on the robot | Stage F1 ended after 0.56 s of policy authority; no stable stand |
| Walking training | Early seed 46 checkpoints completed Isaac Lab evaluation; late training collapsed |
| MuJoCo walking gate | Gate v3 fails on the early candidates' manifest and command envelope |
| Walking on the robot | Not attempted with the new policy |
| Failure replay loop | Code and offline tests exist; hardware-seeded improvement has not been demonstrated |

The separate Phoenix V2 actuator-adaptation study stopped at its simulation
gates. Its result is on the `research/phoenix-hardware-adaptation-v2` branch.

## Why this repo exists

Most open-source quadruped RL projects stop at "trained in sim, deployed
once." Phoenix is explicitly about the loop *after* the first deployment:
reproducing real failures in sim, using them as training seeds, and shipping
a better policy. The full pipeline is driven by YAML configs and ~10 shell
entry points.

## Quick start

```bash
# Install Isaac Lab 3.0+ (https://isaac-sim.github.io/IsaacLab/).
export ISAACLAB_PATH=/path/to/IsaacLab

# Train a baseline policy (~4h on NVIDIA (Blackwell) consumer GPU at 4096 envs)
./scripts/train.sh configs/train/ppo.yaml

# Export to ONNX, bench it, and print the Jetson bringup steps
./scripts/deploy.sh checkpoints/phoenix-base/latest.pt

# After recording a failure on the real robot, replay it in sim
./scripts/replay.sh data/failures/attitude_2026_04_12.parquet

# Fine-tune with the failure curriculum
./scripts/adapt.sh configs/train/adaptation.yaml
```

For a full layout map, see [`docs/structure.md`](docs/structure.md).

## Two Python contexts, one filesystem

Phoenix runs in two Python environments that never share a process.
Data crosses the boundary as files (`*.onnx`, `*.parquet`, `*.mp4`);
no module imports `torch` *and* `rclpy`.

| Context | Where | Optional extra |
|---|---|---|
| Isaac Lab Python | `$ISAACLAB_PATH/isaaclab.sh -p` | `pip install -e ".[sim]"` |
| System Python + ROS 2 | `/opt/ros/humble` + venv | `pip install -e ".[real]"` |

## Tests

```bash
pip install -e ".[dev]"
pytest tests -m "not sim and not ros"
```

The offline tests cover the
config loader, observation builder, failure detector, trajectory logger,
Parquet round-trip, Halton variation sampler, curriculum scheduler, per-env
variation translation, the fail-closed estop / sensor-freshness predicates,
the projected-gravity helper, the `verify_deploy` parity gate, the ONNX-export
observation-normalizer reconstruction, the `reset_bridge` quat/pose
conversion, the lowcmd bridge config builder, the sweep runner, and the
lab-day harness.

Isaac Lab and ROS 2 paths run manually on the hardware:

```bash
pytest tests -m sim    # requires Isaac Lab + GPU
pytest tests -m ros    # requires a running ROS 2 environment
```

## Safety semantics on the deploy path

The real-robot side fails closed by default. `ros2_policy_node` and
`lowcmd_bridge_node` both treat a stale `/phoenix/estop` heartbeat as an
asserted estop, not as "OK to keep going." Every gate is a pure function in
`src/phoenix/sim2real/safety.py` and is unit-tested in `tests/test_safety.py`.

- **Startup is locked.** The policy node refuses to publish until it has
  received a fresh `/phoenix/estop` heartbeat with `data == False` AND fresh
  `/imu/data` AND fresh `/joint_states`. During cold startup with any
  precondition unmet, the node stays silent; the bridge's own fail-closed
  watchdog holds the motors with conservative `hold_kp` / `hold_kd` gains.
- **Past the grace window**, an unmet precondition latches the abort with a
  specific reason (`estop_publisher_missing`, `estop_heartbeat_stale`,
  `external_estop`, `sensor_missing`, `sensor_stale`); the node sends one
  abort notice and goes silent, and the bridge holds the MEASURED posture. It
  no longer drives toward the stand pose on abort.
- **Actions and targets are bounded.** The walking path reads its action clip
  from the checkpoint manifest. The policy node and bridge then apply torque
  and hard-position limits; sustained clipping latches a fault. Earlier H25
  evidence used a measured-position slew limit, so its clip percentages are
  not directly comparable with the walking path.
- **Wireless / joystick deadman**: stale input *or* released button publishes
  `estop=True` within one tick.
- **The LowCmd bridge is the final authority.** `lowcmd_bridge_node` is a thin
  shell around the pure `phoenix.sim2real.actuator_gate`: hard GO2 joint limits
  from Unitree's own URDF, LowState freshness (hold, then damping), rejection of
  NaN, wrong joint order or wire version, a real-deadman requirement when live,
  and one telemetry line per tick (`phoenix.sim2real.bridge_telemetry`), all
  covered by `tests/test_actuator_gate.py`.
- **The H25 contract is stand-only.** A config with
  `base_lin_vel_source: zeros` must declare `safety.stand_only: true`; a nonzero
  velocity command latches an abort (`phoenix.sim2real.deploy_contract`). The
  new walking policy uses a separate 45-D observation and manifest contract.
  It has not been cleared for hardware.
- **Staged hardware gates.** `scripts/harness_preflight.sh` records GO / NO-GO
  evidence for stages A (offline) through H (10 s stand, three attempts) against
  one commit and one artifact lock, and never moves between motor-off and live
  stages on its own. The September stage F1 attempt failed; the
  [Evidence index](EVIDENCE.md) records that path.

The legacy H25 knobs live under `safety:` in `configs/sim2real/deploy.yaml`.
Walking uses the checkpoint manifest for its action clip and gains.

## Configuration model

YAML files under `configs/` support a Hydra-style `defaults:` chain:

```yaml
# configs/env/slippery.yaml
defaults:
  - base
domain_randomization:
  friction_range: [0.05, 0.4]   # overrides base
```

All configs are serialized into each run's log directory as `train.yaml` /
`env.yaml`, so a rollout is fully reproducible from the artifact alone.

## Known limitations

- **Failure-curriculum adaptation.** The `reset_bridge` is wired
  (env-origin-relative poses, xyzw to wxyz quat conversion, configurable
  seed-row and opt-in velocity write) and unit-tested. `adaptation.yaml`
  still ships with `failure_sample_fraction: 0.0` until enough
  hardware-captured parquets exist to validate against. The opt-in velocity
  write passes body-frame velocities into Isaac Lab's world-frame
  `write_root_velocity_to_sim` unrotated; for a failure seeded at a
  non-trivial orientation the injected velocity points the wrong way. It is
  off by default; a proper fix rotates by the base quaternion first.
- **Replay variation application is local-only.** The pure-Python variation
  translation in `replay/apply_variations.py` is unit-tested in CI; the
  Isaac Sim hand-off in `replay/reconstruct.py` is sim-only.
- **rsl_rl 3.0 iter-0 logging artifact.** Fine-tune from a trained baseline
  uses `init_at_random_ep_len=False`; without it, `runner.learn` reports an
  iter-0 "mean reward near 0" even with a byte-exact warm-start. Cosmetic
  only, the warm-start itself is correct.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
