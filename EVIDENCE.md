# Evidence Index

Last reviewed: 2026-04-28 (test count refreshed 2026-05-21). This page exists so a reviewer can see, at a glance,
what is **verified by reproducible artifact**, what is **inferred from
indirect evidence**, and what is **not yet validated** in this repo. If a
claim in the README is not listed here as Verified, treat it as Inferred or
Not validated until proven otherwise.

## Verified

Claims with a reproducible artifact in this repo or a captured log.

- **235 unit tests green in CI**: `pytest tests -m "not sim and not ros"`. Coverage
  listed in [README §Tests](README.md#tests). CI configured to lazy-import torch
  (commit `f235171`).
- **ONNX↔torch parity gate** — `verify_deploy` reports max abs-diff
  **3.8e-06** on the stand-v2 candidate, against a 1e-4 tolerance. Serialized
  at [`docs/pre_lab_gates_2026-04-17.md`](docs/pre_lab_gates_2026-04-17.md).
  Caveat (audit 2026-05-21): this gate compares ONNX vs TorchScript exports
  of the *same* `_ExportablePolicy` wrapper, so it verifies runtime numeric
  parity but **cannot** catch a wrong wrapper. The audit found that
  `export._load_normalizer` searched for checkpoint keys rsl_rl 3.x never
  writes (the `EmpiricalNormalization` buffers live inside `actor_state_dict`
  as `obs_normalizer._mean/_var`), so every pre-2026-05-21 export silently
  dropped observation normalization despite `empirical_normalization: true`
  in every training config. Fixed in branch `audit-fix/skeptic-2026-05-21`;
  all checkpoints must be re-exported and re-parity-checked, and the
  corrected export must be re-verified on hardware before any Gate-7 retry.
- **Stand-v2 sim rollout** — 16 / 16 success @ 20.0 s mean length, 4096-env
  PPO. Raw metrics at
  [`docs/pre_lab_stand_rollout_2026-04-17.json`](docs/pre_lab_stand_rollout_2026-04-17.json).
- **v3b flat-velocity sim eval** — 0.091 m/s lin_err, 0.087 rad/s ang_err,
  32 / 32 success on `Isaac-Velocity-Flat-Unitree-Go2-v0`, 16 envs × 32
  episodes, after the warp-array `flat_tracking_error` fix. Reproduce with
  `phoenix.training.evaluate`.
- **v4 negative result** — written up as a negative result at
  [`checkpoints/phoenix-flat-v4/NEGATIVE_RESULT.md`](checkpoints/phoenix-flat-v4/NEGATIVE_RESULT.md);
  v4 was not shipped.
- **Slew-saturation root-cause analysis** — four training runs (v3b
  fine-tune, slewhinge w=-50, slewhinge w=-5, scratch w=-50) all converge
  to the same 0.57–0.66 m/s lin_err band. Full table in README; full
  analysis at [`docs/retrain_flat_scratch_2026-04-19.md`](docs/retrain_flat_scratch_2026-04-19.md).
- **Hardware deploy chain ran live, 2026-04-18** — `ros2_policy_node`,
  `lowcmd_bridge_node`, estop, parity gates all ran on the GO2 end-to-end.
  Outcome: 30.23% per-step slew saturation specifically at `cmd_vel = 0`,
  which directly motivated stand-v2.
- **Mode-switch runtime**: `policy.mode_switch.enabled` flag, hysteresis +
  25-tick linear blend, unit-tested; runtime path is implemented
  and tested, but see "Not validated" below for hardware status.
- **Fail-closed safety semantics** — every safety gate (`estop_publisher_missing`,
  `estop_heartbeat_stale`, `external_estop`, `sensor_missing`, `sensor_stale`)
  is a free function in `src/phoenix/sim2real/safety.py` with unit tests in
  `tests/test_safety.py`. Slew cap shared between policy node and bridge via
  `per_step_clip_array(...)`.

## Inferred

Claims supported by indirect evidence but not directly measured.

- **"Phoenix loop" generalizes sim → real → fine-tune → real.** The
  *architecture* exists end-to-end (env, training, ONNX export, ROS 2
  deploy, failure detector + parquet logger, replay, adaptation). The loop
  has **not closed once on real failure data** — see "Not validated".
- **stand-v2 will hold the robot upright on hardware.** Inferred from sim
  rollout (16/16) + parity gate (3.8e-06) + the 2026-04-20 dryrun showing
  16.67% slew sat localized to rear thighs (posture-mismatch, not policy
  fault). Not yet a live 10s × 3 stand.
- **The 2026-04-19 single-policy path is exhausted.** Supported by 4
  converging negative runs, not by a formal exploration of the
  reward-weight space.
- **Mode-switch on hardware = stand-v2 at cmd=0, v3b for nonzero.** Logic is
  unit-tested in sim; behavior on hardware is inferred from the two
  sub-policies' individual behavior, not measured as a switched system.

## Not validated

Claims that require hardware time or untaken experiments. Treat as **not yet
true**.

- **Gate 7** — 10 s live stand ×3 on real GO2 in low-level mode. Pending.
  README explicitly lists this as the next step.
- **Gate 8** — flat walking on real GO2 with v3b. Not attempted.
- **Failure-curriculum adaptation against real-robot parquets.**
  `adaptation.yaml` ships with `failure_sample_fraction: 0.0`; the headline
  adaptation result (16.64 / 100% on slippery) is **plain warm-start PPO**,
  not the failure curriculum. The reset bridge is wired and unit-tested but
  has never been driven by real captures.
- **Adapt result generalization.** The `phoenix-adapt` numbers are sim-on-sim
  (slippery overlay) and have not been tested against unseen disturbances or
  on the real robot.
- **Halton replay reconstruction in Isaac Sim.** Pure-numpy translation in
  `replay/apply_variations.py` is unit-tested; the Isaac-Sim
  mass/friction/init-velocity hand-off in `replay/reconstruct.py` has been
  exercised locally but has no hardware-rollout comparison.
- **Cross-terrain transfer of v3b.** Trained and evaluated on flat-v0 only.
  Rough-v0 was retired after the 2026-04-14 dryrun showed 99.5% slew sat.

## Artifacts

- Sim eval metrics: `docs/pre_lab_gates_2026-04-17.md`,
  `docs/pre_lab_stand_rollout_2026-04-17.json`
- Negative results: `checkpoints/phoenix-flat-v4/NEGATIVE_RESULT.md`,
  `docs/retrain_flat_scratch_2026-04-19.md`
- Design specs: `docs/superpowers/specs/2026-04-19-phoenix-gate8-mode-switch-design.md`
- Deploy runbook: `docs/deploy_mode_switch_runbook.md`
- Demo videos: `media/side_by_side.mp4`, `media/side_by_side_adapt.mp4`
- Hardware logs (parquet) live on T7 portable storage, not in this repo.
