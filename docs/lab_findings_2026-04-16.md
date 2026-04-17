# Lab findings — 2026-04-16

Branch: `deploy-run-2026-04-14`
Operator: yusufdxb
Robot: Unitree GO2 + Jetson companion (lab)
Baseline policy: `checkpoints/phoenix-flat/policy.onnx` (Flat-v0, obs_dim=48)

---

## Section 0 — Preflight sync — PASS

- **T7 mount note:** Prompt path was `/media/T7 Storage`; actual mount is `/media/cares/T7 Storage`. Lab-PC udev mounts removable drives under `/media/cares/`. Future prompts should use the corrected path.
- **Jetson IP correction:** Prompt and prior memory said `unitree@192.168.0.4`; actual is `unitree@192.168.0.2`. Memory updated.
- **Sync method change:** Original `tar | ssh "rm -rf && tar x"` would have destroyed local untracked `checkpoints/phoenix-base/` on the Jetson. Replaced with `rsync -av` (no `--delete`).
- **Pre-sync Jetson HEAD:** `c68f5b8` (wireless_estop_node). Strict ancestor of T7 HEAD `7c93986`, so fast-forward via rsync was safe.
- **Post-sync Jetson HEAD:** `7c93986 Add verify_deploy parity gate + reset_bridge tests + lab-day prompt`.
- Branch correct (`deploy-run-2026-04-14`), `policy.onnx` present, `phoenix-base/` preserved.
- 58 MB transferred.

## Section 1 — Verify deploy artifacts — PENDING

## Section 2 — Dryrun saturation gate — PENDING

## Section 3 — Low-level mode toggle — PENDING

## Section 4 — Gamepad deadman test — PENDING

## Section 5 — First live policy run (stand) — PENDING

## Section 6 — Ground run 30s — PENDING

## Section 7 — Post-run artifacts — PENDING
