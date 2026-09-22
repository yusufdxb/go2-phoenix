# Legacy index

Phoenix v2 (2026-09-22) narrowed the project to one loop: measure a persistent change in
actuator response on the robot, target the simulator at it, retrain, gate, redeploy. The
work below is kept, still builds and is still tested, but it is not part of that story.
Nothing was deleted; the v1 narrative is intact at commit `9df76d7`, the base of the
Phoenix v2 branch. "Ashfall" is the sister project on causal failure reproduction and
policy repair for the same robot.

## Belongs conceptually to Ashfall (the failure reproduction and repair project)

| Path | What it was |
|---|---|
| `src/phoenix/replay/` (except `trajectory_reader.py`, used by `sim2real.verify_deploy`) | reconstruct hardware failures in Isaac Sim under a Halton physics sweep |
| `src/phoenix/adaptation/curriculum.py`, `reset_bridge.py`, `scenario_bridge.py` | failure-seeded curriculum fine-tuning |
| `src/phoenix/real_world/synthesize_failure.py` | synthetic failure captures |
| `scripts/replay.sh`, `scripts/adapt.sh`, `scripts/harvest_sim_failures.py`, `scripts/h0_delivery_probe.py`, `scripts/loop_closure.sh` | entry points of the v1 loop |
| `src/phoenix/reliability/hazard_gate.py`, `configs/replay/`, `configs/train/adaptation*.yaml` | hazard gating and replay/adaptation configs |

The v1 loop never closed through hardware: no real failure capture ever seeded a
reconstruction that fed a fine-tune that went back to the robot.

## Demoted research lines

| Path | What it was |
|---|---|
| `src/phoenix/reliability/` and `scripts/reliability_*.py` | the "reliability shield": OOD monitor, arbiter, replication studies, sim only |
| `reliability_eval/`, `analysis/`, `paper/` | results and drafts of that line |
| `src/phoenix/sim2real/mode_switch.py`, `docs/deploy_mode_switch_runbook.md` | stand/walk two-policy switch, never run on hardware |
| `src/phoenix/demo/` | v1 side-by-side video pipeline |
| `src/phoenix/adaptation/fine_tune.py`, `scripts/sweep_run.py`, `docs/sweep_design_2026-05-17.md` | v1 fine-tune wrapper and benchmark sweep |
| `architecture_v1.{dot,svg}` (this folder) | the v1 architecture figure |

## Superseded claims

* "32/32 success" meant "no trunk contact for 20 s"; the attitude flags attached to those
  episodes are corrupted by a quaternion-order bug (audit, H5 and H6).
* Slew percentages before 2026-09-11 used a non-deploy-equivalent metric
  (`docs/superseded_results.md`).
* The v1 demo video (https://youtu.be/Nu0oWyJJbEM) shows a velocity-tracking policy in
  simulation; the deployable policy has only ever been a stand policy.

## Still used by the v2 deploy path

`real_world/failure_detector.resolve_attitude_intervention_rad` is imported by the
deploy contract, the policy node and the preflight evaluator. It stays in place until it
moves into `sim2real/safety.py`.
