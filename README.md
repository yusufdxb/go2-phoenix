<div align="center">

# Phoenix

**Hardware-aware policy adaptation for the Unitree GO2.**

[![CI](https://github.com/yusufdxb/go2-phoenix/actions/workflows/ci.yml/badge.svg)](https://github.com/yusufdxb/go2-phoenix/actions/workflows/ci.yml)
&nbsp;![Python](https://img.shields.io/badge/python-3.10%2B-blue)
&nbsp;[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Phoenix is a real-to-sim-to-real loop around actuator change. While a learned
policy runs on the robot, Phoenix measures the difference between the joint targets
actually sent to the motors and the measured joint response, separately from anything
its own safety layer changed. A persistent deviation at one joint defines a targeted
actuator distribution in Isaac Lab. A candidate policy is fine-tuned on it and
redeployed only if it improves the changed condition without materially harming
nominal locomotion.

**RUN → DETECT → CONDITION → TRAIN → VERIFY → REDEPLOY**

Detection runs live beside the robot: `scripts/phoenix_live_monitor.py` tails the
bridge telemetry as it is written and updates the health vector every second. It has
not yet been run during a robot session. Training runs on a workstation between
deployments. It is not onboard, and it is not instantaneous learning.

> **Research question.** Can persistent actuator tracking residuals observed on a
> deployed quadruped define a targeted simulation training distribution that improves
> locomotion under the robot's changed hardware dynamics without degrading nominal
> performance? ([details](docs/research/RESEARCH_QUESTION.md))

<p align="center"><img src="docs/architecture.svg" alt="Phoenix architecture" width="95%"></p>

## Status (2026-09-22)

The loop has **not** closed through hardware. Status words follow [`EVIDENCE.md`](EVIDENCE.md).

| Stage | Status |
|---|---|
| DETECT: four-layer command accounting, residual, calibration, persistence, health vector, live tail | IMPLEMENTED, OFFLINE VERIFIED on synthetic data |
| Deployment-fidelity gate | IMPLEMENTED, OFFLINE VERIFIED on the 2026-09-21/22 GO2 logs |
| CONDITION: health report to targeted Isaac Lab overlay | IMPLEMENTED; sim event NOT YET VERIFIED (needs Isaac Lab) |
| TRAIN: warm-start PPO fine-tune, one config for every arm | IMPLEMENTED (existing runner); not yet run for v2 |
| VERIFY: candidate gate and promotion rule | IMPLEMENTED, OFFLINE VERIFIED |
| Controlled degradation on the robot (gated gain reduction on one joint) | IMPLEMENTED; NOT YET VERIFIED on hardware |
| A policy the robot executes faithfully | **not yet**: see below |
| A walking baseline that needs no downstream correction | SIM VERIFIED (W2); not deployable under the frozen limiter rule |

**The first thing Phoenix found is about Phoenix.** In the only live policy run on the
GO2 (stand-only policy, 2026-09-22), the execution layer altered 88.8 % of the joint
targets the policy requested, and the policy had 0.56 s of authority before a rear-thigh
target crossed the abort band. In simulation, in the policy's own training
distribution, the same limiter is active on 59.7 % of joint-steps (a clip-activation
rate, recorded with the earlier metric; the altered-by-more-than-1-mrad fraction has
not been measured in sim yet). No adaptation claim can be made from such runs, and the
monitor refuses them. Fixing that is phase 0 of the
[experiment](docs/research/EXPERIMENT.md). The incumbent policy is also stand-only;
walking needs a velocity policy that does not exist yet. Evidence per claim:
[`EVIDENCE.md`](EVIDENCE.md).

**Update, 2026-09-22, later (simulation only; nothing has run on the robot).** A
walking baseline now exists: trained with no soft limiter in the loop, it passes its
preregistered gate on held-out seeds (0.90 / 0.89 / 1.00) and tracks commands in both
directions. It is not deployable, because no rate bound in the frozen grid acts as a
seatbelt for it, and the smoother variants do not walk. The walking stage then stopped at
the same preregistered rule as the standing stage: the one controlled degradation this
study may apply, a single joint at half gain, moves neither policy's endpoint by the
required margin. The central question is still untested. Details:
[`docs/research/PHOENIX_V2_RESULTS.md`](docs/research/PHOENIX_V2_RESULTS.md).

**Update, 2026-09-22 (simulation only; nothing new has run on the robot).** Two
deploy-contract defects explain the rewriting: the measured-position clip (also a
4.4 N m torque cap) and a missing [-1, 1] action clamp that training always applied.
With the clamp and a 0.075 rad/step command-rate limiter, the same policy runs through
the exact deploy code in simulation with 0.14 % of targets altered. The stand stage then
stopped by its preregistered rule (the stand policy absorbs a 50 % RR_thigh gain loss),
and the first walking baseline failed its gate. Details:
[`docs/research/PHOENIX_V2_RESULTS.md`](docs/research/PHOENIX_V2_RESULTS.md).

## Experiment

Five arms are fine-tuned from the same incumbent at the same budget: continued
training, broad actuator randomisation, joint-only broad randomisation,
Phoenix-targeted (built from residuals measured on the robot) and an oracle-targeted
ablation. They are evaluated in simulation over 10 seeds under nominal, degraded,
held-out-severity and wrong-joint conditions. On the GO2, the incumbent, the broad arm
and the selected Phoenix candidate then run under a controlled, reversible software
degradation, which is the same parameter as in the simulator. The
protocol runs on standing first and on walking once a walking policy exists. Success
and failure criteria are fixed in [`docs/research/EXPERIMENT.md`](docs/research/EXPERIMENT.md)
before any run. Status: stopped before the arms were trained (see the update above).

A one-joint model ([`docs/research/TOY_MODEL.md`](docs/research/TOY_MODEL.md)) shows why a
nominal policy is brittle to gain loss and why targeting costs nominal performance. It
also shows Phoenix's mixture tying with broad randomisation on one joint.

## Demo

None yet: no hardware result exists to film.

## Reproduce

```bash
pip install -e ".[dev,real]"
PYTHONPATH=src pytest tests -m "not sim and not ros"      # torch-free, ROS-free

# the offline loop on a bridge telemetry file
PYTHONPATH=src python3 scripts/phoenix_loop.py fidelity  <bridge.jsonl>
PYTHONPATH=src python3 scripts/phoenix_loop.py calibrate <nominal runs...> --regime stand --out baseline.json
PYTHONPATH=src python3 scripts/phoenix_loop.py assess    <run> --baseline baseline.json --regime stand --out health.json
PYTHONPATH=src python3 scripts/phoenix_loop.py condition health.json --parent-env ../stand_v3_h25 --out configs/env/conditioned/x.yaml

# live, beside a running robot (read-only)
PYTHONPATH=src python3 scripts/phoenix_live_monitor.py <bridge.jsonl> --baseline baseline.json

# training (Isaac Lab, GPU)
scripts/phoenix_train_candidate.sh phoenix configs/env/conditioned/x.yaml <incumbent.pt> <seed>
```

Runbooks: [training](docs/runbooks/TRAINING.md), [deployment](docs/runbooks/DEPLOYMENT.md),
[hardware experiment](docs/runbooks/HARDWARE.md). Layout: [`docs/structure.md`](docs/structure.md).

## Limitations

* The controlled degradation scales one joint's PD gains in software. It is not motor
  damage, and results do not claim to diagnose a physical fault.
* The authority estimate `s_hat` is a response-effectiveness ratio under a matched task,
  not a motor-health percentage. It is biased near torque saturation and when the policy
  compensates the error it measures.
* No online-adaptation (RMA-style) arm is run, so conclusions are limited to policies
  without history input, and no claim is made against those methods.
* One robot; stand before walk.
* Related work that already does parts of this: [`docs/research/RELATED_WORK.md`](docs/research/RELATED_WORK.md).

## Citation and license

[`CITATION.cff`](CITATION.cff). MIT, see [`LICENSE`](LICENSE).
