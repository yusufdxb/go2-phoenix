# MARK on the physical GO2 (2026-11-05)

MARK is a learned skill: from a stand, the robot shifts its weight onto three
legs, lifts the rear-right leg up and outward with the knee bent, puts it down
and recovers its stance. These are its
first full runs on hardware, including the first runs of the new tall stance
(0.343 m target).

Sequence per run (`stand-mark`, stop at `full`): scripted stand-up, walker
policy hold, settle gate, MARK policy, blend back, post-stand, then a scripted
sit. The phase log is identical in all three runs (STAND, WALK, STOPPING,
SETTLE_GATE, MARK, RESUME_BLEND, POST_STAND, DONE).

## Runs

| Run | Stance | Verdict | Height | Overshoot | Rear-right raised | Max raise | Max tilt |
|---|---|---|---|---|---|---|---|
| r1 control | short (h25) | GO, no fall | 0.307 m | +0.6 mm | 3.28 s | 45 mm | 4.10 deg |
| r3 tall, run 1 | tall | GO, no fall | 0.3412 m | +0.1 mm | 3.79 s | 79 mm | 4.96 deg |
| r3 tall, run 2 | tall | GO, no fall | 0.3415 m | +0.1 mm | 3.80 s | 80 mm | 4.93 deg |

- Every run ended on `authority_window_complete:DONE` with MARK done and not
  stopped, 0 control-loop overruns, and policy step latency p99 1.70 to
  1.78 ms (max 2.06 ms).
- Height and raise are estimated from joint angles, not measured externally.
- Battery 70 % at the first run, 56 % after the last.

## Remaining jerk (peak joint acceleration, rad/s^2, quiet stand 5 to 7)

| Transition | r1 control | tall run 2 |
|---|---|---|
| Stand-up to walker | 7 | 8 |
| Walker to MARK | 130 | 141 |
| MARK sequence start | 55 | 76 |
| Stop before sit | 22 | 30 |
| Sit start | 16 | 20 |

The walker-to-MARK switch is the remaining spike: at the switch a thigh target
steps by about 0.1 rad because there is no blend from the walker's last target
to MARK's first. A short blend is the next fix. On the tall stance the sit
peaks at 10.4 deg tilt before settling to 4 deg.

## Proof limits

- Three runs, one robot, one day, feet on the floor, operator at the deadman.
- The deploy code (`05d62d8`, `a87e190`) and MARK training are on development
  branches, not in this repository. Policy hashes are in each
  `runs/*_mark_rung_full.json`.
- Raw bridge logs (up to 550 MB per run) are held with the development branch.

## Files

| Path | Content |
|---|---|
| `runs/*_mark_rung_full.json` | rung verdict, reasons, code sha, walker and MARK policy hashes |
| `runs/*_composite.summary.json` | composite node summary: phases, ticks, latency, overruns |
| `SHA256SUMS` | checksums of every file here |
