# Outcome note for 2026-10-07-joint-friction-dr (written after the study; the preregistration is unchanged)

Date: 2026-10-07. This note records outcomes. It does not amend, reinterpret or replace any
rule in `2026-10-07-joint-friction-dr.md` (whose only amendment is Amendment 1, the checkpoint
name, committed before any T/C outcome was computed).

## Results as registered

| Item | Result | Evidence |
|---|---|---|
| Positive control P1 (Isaac friction in N m) | strict probe verdict FAIL on 1/16 cells (FR_hip at 1.2 N m never broke away, estimate undefined); 15/16 within the probe tolerance (worst 1.2 % error) | `results/friction_dr/isaac_probe.json` |
| Positive control P2 (friction hurts incumbent) | PASS: yaw fraction 0.267 <= 0.60 | `results/friction_dr/analysis.json` |
| E1 (yaw, fmeas=1.25) | PASS: mean paired d +0.225, bootstrap 95 % CI [0.139, 0.318], p = 1/32 | `results/friction_dr/analysis.json` |
| E2 (forward, fmeas=1.25) | PASS: mean paired d +0.208, CI [0.167, 0.247], p = 1/32 | same |
| Promotion guards | **FAIL for all 5 T seeds** (frictionless replay differs from the paired C by more than 0.10: forward on every seed, yaw on seeds 50 and 51) | `results/friction_dr/promotion_guard.json` (`scripts/friction_dr_promotion_guard.py`) |

## Consequences (binding)

- Friction hypothesis (sim replay): PASS.
- Formal promotion: FAIL. No friction-DR checkpoint is promoted. The guard is not reinterpreted
  (a one-sided reading would be a post-hoc change and is not adopted).
- `fric_T_s50` `model_3749.pt` (sha256 9948900b...) is used only as an EXPLORATORY hardware
  candidate, under the separate preregistration `2026-10-07-gantry-ab-exploratory.md`. Any
  later promotion needs a new confirmatory preregistration written before its data exist.
- Everything above is simulation evidence. No hardware was used in this study.
