# A/B/C walking trial on the physical GO2 (2026-10-20)

Comparison, preregistered on 2026-10-18, of the incumbent walking policy against the two
friction-aware retrains described in
[`../2026-10-17_friction_model/`](../2026-10-17_friction_model/README.md).
Physical Unitree GO2, one free floor, no gantry or rope.

## Arms

| Arm | Policy | Training |
|---|---|---|
| A | incumbent `model_2750` | original walking policy, no joint friction |
| B | `fm_B_s62 model_3749` | fine-tune with joint friction randomized around the values measured on the robot |
| C | `fm_C_s60 model_3749` | fine-tune with broad joint friction, 0 to 1.6 N m |

The same policy files as the simulation evidence; their hashes are in
`../2026-10-17_friction_model/candidates/*/export/phoenix_manifest.json`. Every
hardware run recorded the same ONNX hashes.

## Preregistered decision rule (2026-10-18, before the trial)

- Primary measure: tape forward displacement under both rear feet, as a
  fraction F of the commanded distance.
- An arm "walked further than A" if its mean F exceeds A's by at least 0.10,
  the exact two-sided permutation p over all relabelings is at most 0.05, and
  the arm had no safety event. With 4 trials per arm the smallest attainable p
  is 0.029.

## Protocol

- Command `vx03_4s`: 0.3 m/s forward for 4 s, 1.20 m commanded.
- 12 blocks in the preregistered balanced order A B C A C B C B A B C A, so
  each arm walks 4 times across the falling battery (55 % to 40 %, see
  `compare/trials.png`).
- Distance is measured with a tape at both rear feet. Leg odometry is logged
  as an estimate and never replaces the tape.
- STOP drills (deadman release under policy authority) in blocks 10 to 12.

## Result

| Arm | Forward fraction F (tape / commanded), mean (SD) | Tape distance | Heading drift |
|---|---|---|---|
| A | **0.677** (0.004) | about 0.81 m | 1.48 deg |
| B | **0.994** (0.007) | about 1.19 m | 1.42 deg |
| C | **1.032** (0.004) | about 1.24 m | 0.20 deg |

- B - A = +0.316 and C - A = +0.355 in F; exact two-sided permutation
  p = 0.029 for each (the smallest p possible with n = 4 per arm).
- Preregistered reading: B and C each walked further than A (difference
  >= 0.10, p <= 0.05, no arm safety event).
- C also cut heading drift (p = 0.029 against A); B did not (p = 0.80).
- MuJoCo, with the identified friction family, predicted F of 0.54 (A),
  0.84 (B) and 0.98 (C). Hardware is higher for every arm, and the ordering
  A < B < C holds.
- Leg odometry / tape: 0.853 (A), 0.860 (B), 0.873 (C). An A-only odometry
  calibration carries to B and C within about 2 %.

## Safety

- No stop condition and no discrepancy recorded. Bridge faults seen were only
  the expected ones (shutdown, e-stop and authority-window end from the STOP
  drills). No battery, temperature or collapse latch fired.
- Every one of the 43 bridge logs names `phoenix_lowcmd_bridge` as the only
  `/lowcmd` publisher.

## Valid and invalid attempts

28 attempts, 26 valid.

- Block 4, arm A, first walk: the deadman dropped for 0.5 s before the policy
  started. Repeated once, as the protocol allows; the repeat was clean.
- Block 10, arm B, STOP drill: the release came after the policy window, so
  it is invalid. An extra arm B drill after block 12 had a valid release and
  is recorded as exploratory, outside the 12 blocks.

## Deviations from the preregistration

1. Run prompts were answered by a scripted driver relaying the operator's
   answers, with the operator's authorization (`driver_relaying_operator`).
2. Tape readings were read at the robot by the operator and relayed verbally
   after each run; they are applied through `metadata_corrections.json`, and
   the raw relay is `operator_tape_readings.jsonl`. Lateral tape was recorded
   only for block 2 (5 cm).
3. `touched = false` and rope `n` were defaulted by the driver (there was no
   rope; the operator reported no touches).
4. The fixed 60 s pause between blocks was dropped from block 10 on; walks
   stayed at least 60 s apart because of the stand and re-centring.
5. Stage E needed retries (deadman not held at bridge start); a 10 s hold
   lead-in was added.
6. Bundles were re-staged at the trial commit because payload checks are
   commit-bound; hashes are identical to the preregistered arms.
7. Battery: the lab card asked for 90 %; the code floor of 30 % applied
   unchanged and the trial started at 55 %.

## Proof limits

- n = 4 per arm, one floor, one robot, one day, one forward command. The
  preregistration itself calls this exploratory.
- Tape resolution about 1.3 cm, relayed verbally, not photographed.
- The analyzer's own verdict string reads INSUFFICIENT because its two-arm
  gantry rule also requires a `yaw05_3s` preset; that rule is not part of this
  preregistration. The reading above is computed from `compare/comparison.json`.
- No turns, other speeds or other surfaces yet. B and C were each the best of
  their arm on the same MuJoCo evaluation (selection noted in PR #8).
- The trial code (`228555e`) and the raw session logs are held with the
  development branch and are not in this repository.

## Files

| Path | Content |
|---|---|
| `analysis/runs.json`, `analysis/summary.json`, `analysis/report.md` | per-run analyzer output |
| `compare/comparison.json`, `compare/comparison.md` | per-block table and per-arm statistics |
| `compare/*.png` | forward fraction, drift, odometry vs tape, trial order |
| `metadata_corrections.json`, `operator_tape_readings.jsonl` | tape corrections and the raw relay |
| `SHA256SUMS` | checksums of every file here |
