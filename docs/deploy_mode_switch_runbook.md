# Mode-switch bringup runbook

Lab-day checklist for the two-policy runtime (stand-v2 @ cmd=0, v3b @ walking).
Spec: `docs/superpowers/specs/2026-04-19-phoenix-gate8-mode-switch-design.md`.

## 1. Sync T7 → Jetson

On mewtwo (host):
- Confirm `checkpoints/phoenix-flat/policy.onnx` is stand-v2 (`md5sum` matches the value in `docs/retrain_stand_v2_2026-04-19.md`).
- Confirm `checkpoints/phoenix-flat/v3b/policy.onnx` exists (export done 2026-04-19).
- `rsync -avL` both dirs to T7 under `go2-phoenix/checkpoints/`.

On CaresLab PC:
- Mount T7 at `/media/careslab/T7 Storage`.
- `scp -r` of the two dirs into the Jetson's Phoenix checkout (`~/go2-phoenix/checkpoints/phoenix-flat/` and `~/go2-phoenix/checkpoints/phoenix-flat/v3b/`).

On Jetson:
- `md5sum checkpoints/phoenix-flat/policy.onnx` and `checkpoints/phoenix-flat/v3b/policy.onnx`; both must match the mewtwo-side hashes.

## 2. Offline gates on Jetson

- `python3 -m phoenix.sim2real.verify_deploy --deploy-cfg configs/sim2real/deploy.yaml --parquet <trained-distribution parquet>` — parity gate on whichever ONNX `onnx_path` points at (defaults to stand-v2).
- Temporarily swap `onnx_path` to the v3b path and re-run verify_deploy. Both should show `max_diff < 1e-4`.
- Revert `onnx_path` back to stand-v2.
- `pytest tests/test_mode_switch.py -q` — all mode-switch unit tests green.

## 3. Edit `configs/sim2real/deploy.yaml`

Set:
```yaml
policy:
  mode_switch:
    enabled: true
```
Leave all thresholds at spec defaults for the first bringup. `stand_onnx_path` / `walk_onnx_path` already point at the correct files.

## 4. Three-bridge bringup + estop heartbeat

Unchanged from prior bringup procedure — see `docs/lab_session_2026-04-16.md` for the sequence. Mode-switch has no ROS 2 topic changes.

## 5. Hardware gates

### G7 — live stand (cmd=0 throughout)
Publish `cmd_vel = (0, 0, 0)` for 10 s × 3 attempts. Expect:
- Mode stays STAND throughout (no state transitions). Confirm via `ros2 topic echo /phoenix/estop` plus any debug logging enabled.
- No attitude abort, no collapse, no estop latch.
- Slew saturation in the parquet log < 5%.

### G8a — step 0 → 0.3 m/s forward → 0
Teleop sequence: `(0, 0, 0)` for 3 s → `(0.3, 0, 0)` for 5 s → `(0, 0, 0)` for 3 s. Expect:
- One STAND→WALK transition at the 3 s mark, completing in ~0.5 s (25 ticks).
- One WALK→STAND transition at the 8 s mark, completing in ~0.5 s.
- No attitude abort, no collapse during either transition.
- Walking-phase slew saturation in parquet < 5%.
- Stopping-phase slew saturation < 5% (this is the pattern that failed at 30% pre-mode-switch).

### G8d — combined lin + yaw
`(0, 0, 0)` for 3 s → `(0.3, 0, 0.3)` for 5 s → `(0, 0, 0)` for 3 s. Same pass criteria as G8a.

### G8b / G8c / G8e (nice-to-have)
- G8b: 0.5 m/s instead of 0.3.
- G8c: yaw only, `(0, 0, 0.5)`.
- G8e: rapid toggle 1 s walk / 1 s stop × 10 cycles. Confirms hysteresis handles noisy teleop without flutter.

## 6. On green

- Tag `v0.3.0-gate8-mode-switch`.
- Push T7 + origin.
- Update `docs/PHOENIX_NEXT_STEPS.md` — mode-switch replaces the v3b-retrain ask.
- Write `docs/gate8_mode_switch_<date>.md` with parquet evidence.

## 7. Rollback

One flag:
```yaml
policy:
  mode_switch:
    enabled: false
```
Re-run the node. `onnx_path` still points at stand-v2 → node behaves exactly as Gate 7 did on 2026-04-18. No code rollback needed.

## Troubleshooting

- **Node fails to start with `FileNotFoundError: mode_switch.walk_onnx_path missing`** → the v3b ONNX wasn't scp'd to the Jetson. Copy the `phoenix-flat/v3b/` dir over and retry.
- **Attitude abort during TRANS_TO_WALK** → likely a blend discontinuity. Disable mode switch (rollback), and open a new spec to lengthen `transition_ticks` to 50 or try a cosine blend.
- **Mode stays STAND even under `(0.5, 0, 0)` cmd** → verify the `/cmd_vel` subscriber is receiving the message (`ros2 topic echo /cmd_vel`). If yes, check `cmd_magnitude` computation: `sqrt(0.5^2) = 0.5 > 0.15`, should have triggered TRANS_TO_WALK.
- **Rapid state flutter during steady cmd** → raise `enter_walk_thresh` or lower `enter_stand_thresh` to widen the hysteresis band.
