# Lab findings — 2026-04-20

Branch: `deploy-run-2026-04-14` (no git commits this session — T7 only).
Operator: lab-PC SSH session, no human on the controller, no physical change
to the GO2. Session was driven from `careslab` (lab PC, mewtwo) over
WiFi `CaresLab` → Jetson at `192.168.0.2` using only ssh/scp/rsync.
Baseline from previous session: flat-v0 saturation **30.23%** on stand
dry-run, gate <5% (see `lab_findings_2026-04-18.md`).

---

## Headline

**Stand-v2 dry-run saturation: 16.67%.** Down ~2× from flat-v0 but still
over the 5% gate. **However, saturation is not distributed uniformly —
it is 100% concentrated on the two rear thigh joints, and 0% on every
other joint.** That pattern points at a `default_joint_pos` vs sport-mode
posture mismatch, not a policy-quality failure.

Gate 7 remains blocked, but the unblock path is simpler than a retrain.

---

## What was executed

1. **Discovered the stand-v2 export was a ghost.** On T7,
   `checkpoints/phoenix-stand-v2/2026-04-19_11-20-36/` contained a
   trained PPO checkpoint (`latest.pt` sha `ab7ba42e…`), but the
   `policy.onnx` / `policy.onnx.data` / `policy.pt` in the same
   directory were **byte-identical to `phoenix-flat/`** — the
   latest export had sourced `phoenix-flat/latest.pt` by mistake
   instead of stand-v2's own weights. Only the `.pt` rsl_rl checkpoint
   was genuinely stand-v2. Before discovering this we would have
   deployed flat's weights under a stand-v2 label.

2. **Re-exported stand-v2 on the Jetson** (no mewtwo touch required).
   `src/phoenix/sim2real/export.py` only imports `torch` + `onnxruntime`
   — no IsaacLab. The Jetson already had `torch 2.11.0+cu130` and
   `onnxruntime 1.18.1`; only `onnx` was missing.

   - **onnx install path on the air-gapped Jetson:** `pip download
     'onnx==1.16.2' --platform manylinux2014_aarch64 --python-version
     310 --only-binary=:all: --no-deps` on careslab, rsync the wheel to
     Jetson `/tmp/`, `pip install --user --no-deps --no-index`. Picked
     `onnx==1.16.2` on purpose: it supports `numpy<2` and `protobuf<5`,
     so it slots into the Jetson's existing pin set (numpy 1.26.4,
     protobuf 4.25.9) without collateral upgrades. Later versions would
     have force-upgraded numpy to 2.2 and broken `onnxruntime 1.18.1`.

   - **torch 2.11 default `dynamo=True` exporter needs `onnxscript`**,
     which is not available air-gapped. Monkey-patched
     `torch.onnx.export` with `dynamo=False` at call site (legacy
     TorchScript-based path) — one-off inline in the ssh command, no
     edit to `export.py`. Legacy path warned as deprecated but exports
     cleanly.

   - **Parity check:** max torch↔onnxruntime abs diff =
     **9.537e-07** (tol 1e-4). PASS. Inferred `obs_dim=48,
     action_dim=12, hidden=[512, 256, 128]`.

   - **Result** (single-file inline-weights ONNX this time, not
     `.onnx` + external `.onnx.data`):
     - `checkpoints/phoenix-stand-v2/policy.onnx` sha
       `0bf9b52d772d3878c95dc93a1ef9de7abe8bda0cf54fb6b3a58704a90cca1ebf`
     - `checkpoints/phoenix-stand-v2/policy.pt`   sha
       `b3010baa0743fec89bba95d813392edabe0c0db6c6310394b94f194642321c28`
     - `checkpoints/phoenix-stand-v2/latest.pt`   sha
       `ab7ba42ef049f379a0fa4f2722caa408a8d99b5bf98b8ace3d052fbbf2a693cb`

     These ARE different from `phoenix-flat/` (flat
     `policy.onnx.data` = `d0661fd9…`; the stand-v2 re-export uses
     inline weights so direct hash comparison isn't apples-to-apples,
     but the numerical behavior on the dryrun is different from flat —
     see section below).

3. **Authored `configs/sim2real/deploy_stand_v2.yaml`** on T7 — copy of
   `deploy.yaml` with `policy.onnx_path` +
   `policy.torchscript_path` repointed at `phoenix-stand-v2/` and
   safety values left at canonical `0.5 / 0.2 / 3.0`.
   scp'd to Jetson. **Per the 04-18 convention, the Jetson copy was
   then bumped** to `estop_timeout_s: 2.0`, `sensor_timeout_s: 1.0`,
   `startup_grace_s: 10.0` to work around cyclonedds discovery race;
   T7 copy is unchanged. The bumped Jetson yaml is archived at
   `docs/attachments/2026-04-20/jetson_deploy_stand_v2.yaml`.
   `max_runtime_s` was lowered to `25` on Jetson only for the dryrun.

4. **Ran a stand-v2 dry-run** on the Jetson with the three DRY bridges
   (`estop_publisher` heartbeat, `lowstate_bridge_node`, `lowcmd_bridge_node`
   in DRY mode publishing `/lowcmd_dry`). GO2 was powered in stock sport
   mode the whole time — motors were NEVER enabled by Phoenix because
   `--live` was not passed. Bridge topic hz during the run:

   | topic | measured | target |
   |---|---:|---:|
   | `/joint_states` | 496.6 Hz | ~500 |
   | `/imu/data`     | 500.4 Hz | ~500 |
   | `/lowcmd_dry`   | 50.25 Hz | 50 |
   | `/phoenix/estop` | 10 Hz, pub=1 sub=1 | 10 |

   Captured three parquets (`data/failures/stand_v2_dryrun_2026-04-20_*.parquet`):
   - `17-59-55.parquet` — first run, aborted on `estop_heartbeat_stale`
     because the canonical `estop_timeout_s: 0.5` + `startup_grace_s:
     3.0` lost the cyclonedds discovery race. **Pre-bump.** 512 rows /
     10 s before abort.
   - `18-00-54.parquet` — **post-bump, clean run, readable.** 1024 rows
     / 20.9 s. Shutdown was via `timeout --signal=SIGINT`. This is the
     primary source for the saturation table below.
   - `18-02-42.parquet` — self-terminated on `max_runtime_s: 25`.
     **The parquet has no Parquet footer** (`pyarrow.lib.ArrowInvalid:
     Could not open Parquet input source … Parquet magic bytes not
     found in footer`). See "Bug observed" below.

5. **Computed saturation offline** from the readable parquet, using
   the policy node's own `target = default_q + action_scale * action`
   formula and `per_step_clip_array(target, q, MAX_DELTA=0.175)`.

---

## Saturation table (parquet `stand_v2_dryrun_2026-04-20_18-00-54`)

1024 rows, 20.91 s, 49 Hz effective; 12 joints × 1024 = 12288 motor-steps.
**Overall saturation: 16.67%** (2048 / 12288 motor-steps).

| joint              | sat%   | max \|Δ\| | mean \|Δ\| | q_mean | target_mean | cmd bias |
|--------------------|-------:|------:|-------:|-------:|------------:|---------:|
| FL_hip_joint       |  0.00% | 0.1118 | 0.0983 | -0.018 |  0.081 | +0.0983 |
| FR_hip_joint       |  0.00% | 0.0550 | 0.0068 |  0.029 |  0.036 | +0.0066 |
| RL_hip_joint       |  0.00% | 0.0637 | 0.0628 |  0.005 |  0.068 | +0.0628 |
| RR_hip_joint       |  0.00% | 0.1453 | 0.1441 |  0.030 |  0.174 | +0.1441 |
| FL_thigh_joint     |  0.00% | 0.0911 | 0.0895 |  0.635 |  0.546 | -0.0895 |
| FR_thigh_joint     |  0.00% | 0.1349 | 0.1338 |  0.858 |  0.724 | -0.1338 |
| **RL_thigh_joint** | **100.00%** | **0.4291** | **0.4283** | **0.652** | **1.080** | **+0.4283** |
| **RR_thigh_joint** | **100.00%** | **0.3655** | **0.3631** | **0.733** | **1.096** | **+0.3631** |
| FL_calf_joint      |  0.00% | 0.0501 | 0.0232 | -1.529 | -1.507 | +0.0228 |
| FR_calf_joint      |  0.00% | 0.0994 | 0.0142 | -1.456 | -1.470 | -0.0142 |
| RL_calf_joint      |  0.00% | 0.0191 | 0.0175 | -1.475 | -1.458 | +0.0175 |
| RR_calf_joint      |  0.00% | 0.1392 | 0.0648 | -1.470 | -1.535 | -0.0648 |

Action magnitude (policy-output, pre-scale):
- raw range: min -1.022, max 0.701, mean 0.079, std 0.413
- vector-norm per step: mean 1.457, max 1.462

---

## What this really means — the unblock path for Gate 7

The rear thighs alone account for 100% of the saturating motor-steps.
The scaled policy output `action_scale × action` has mean |·| ≈ 0.10
across all 12 joints — comfortably under the 0.175 clip. What the clip
is hitting is **the offset between `default_joint_pos` and the
sport-mode resting posture**:

- Phoenix `default_joint_pos[R*_thigh_joint] = 1.0` rad (trained
  stand pose).
- GO2 held in stock sport mode rests the rear thighs at **0.65–0.73
  rad**.
- Delta ≈ 0.35 rad on step 1, step 2, step N — the per-step clip
  fires on every single step because the bridge is asked to move the
  rear thighs "toward 1.0" and the robot isn't going anywhere (no
  actuation, DRY).

This reframes the 2026-04-18 flat-v0 30.23% number: some unknown fraction
of that was the same baseline mismatch, not a policy that "wants to
move too aggressively." The retrain with `action_rate=-0.5`,
`joint_acc=-1.0e-6` made real progress — stand-v2's action magnitude is
more conservative than flat-v0's — but that progress is masked by the
offset baseline.

**Two ways to get an unmasked saturation reading for Gate 7 prep:**

1. **Switch the GO2 to low-level mode on the stand** (runbook Section
   4a: `ros2 run unitree_mode_ctrl mode_ctrl --ros-args -p
   target:=low`). The sport controller disengages, the robot settles
   onto the stand under gravity, and the measured joint angles will
   sit much closer to the trained default. **Requires the GO2 on a
   stand and a human operator nearby** — this is the start of Gate 7
   proper. Until then, DRY-mode saturation is measuring the wrong
   delta.

2. **Pre-align `default_joint_pos` in deploy_stand_v2.yaml to sport-mode
   resting posture** (rear thighs 1.0 → ~0.70, front thighs 0.8 →
   ~0.86, etc.). Easier to do but less principled: the policy was
   trained with `default_joint_pos[R*_thigh] = 1.0`, so changing it at
   deploy time changes what the policy observes (any `q - default_q`
   term in the obs) and where it tries to converge. Only useful as a
   diagnostic — not a real fix.

**Recommended:** plan a real Gate 7 session with the GO2 on a stand and
a human operator. The rest of the pipeline (bridges, parquet logging,
shutdown flush, safety latches) is now validated. The only remaining
question is whether stand-v2 holds a clean stand once the sport-mode
offset is removed.

---

## Bugs observed

1. **`max_runtime` shutdown path produces a footerless parquet.**
   `data/failures/stand_v2_dryrun_2026-04-20_18-02-42.parquet`
   (pulled at 152510 bytes after the process was eventually killed) is
   unreadable — `pyarrow` raises `Parquet magic bytes not found in
   footer`. The `lab_findings_2026-04-18.md` §"Real bugs surfaced" #1
   claims this was fixed, and the SIGINT path in the same session DID
   flush cleanly. So either the fix only covers the SIGINT path and
   misses the `_latch_abort → shutdown` path, or there is a regression.
   Worth checking
   `phoenix.sim2real.ros2_policy_node.shutdown()` + `_latch_abort` ordering
   around `self._logger.close()`.

2. **cyclonedds discovery race still unfixed.** Canonical
   `startup_grace_s: 3.0` + `estop_timeout_s: 0.5` + `sensor_timeout_s:
   0.2` fail every time under this Jetson's DDS stack. The lab-session
   workaround (bump to 10 / 2.0 / 1.0 on Jetson yaml only) continues to
   be the pragmatic path. The "gate startup on first-message-received
   rather than wallclock" action item from 04-18 is still open.

---

## Artifacts

On T7:
- `checkpoints/phoenix-stand-v2/policy.onnx` (re-exported on Jetson,
  inline weights, sha `0bf9b52d…`)
- `checkpoints/phoenix-stand-v2/policy.pt`   (sha `b3010baa…`)
- `checkpoints/phoenix-stand-v2/latest.pt`   (source PPO checkpoint,
  sha `ab7ba42e…` — unchanged)
- `configs/sim2real/deploy_stand_v2.yaml` (canonical safety values)
- `data/failures/stand_v2_dryrun_2026-04-20_17-59-55.parquet` (pre-bump
  abort, 512 rows)
- `data/failures/stand_v2_dryrun_2026-04-20_18-00-54.parquet`
  (**primary evidence**, 1024 rows, readable)
- `data/failures/stand_v2_dryrun_2026-04-20_18-02-42.parquet`
  (footerless, archived for bug-reproduction purposes only)
- `docs/attachments/2026-04-20/` — bridge and policy logs, plus the
  Jetson copy of `deploy_stand_v2.yaml` with the bumped timeouts.

On Jetson: same tree under `~/yusuf/go2-phoenix/`. Air-gapped onnx
wheel installed at `~/.local/lib/python3.10/site-packages/onnx/`
(version 1.16.2). Source wheel at `/tmp/onnx-1.16.2-*.whl` (not
persisted across reboot).

---

## Concrete next actions (supersede 04-18 list items 1–3)

1. **Gate 7 session — low-level mode, GO2 on stand, human operator.**
   Re-measure saturation with the sport-mode offset removed. This is
   the only honest read on whether stand-v2 is Gate-7 ready.
2. **Fix the `max_runtime` / `_latch_abort` shutdown path** so the
   parquet always flushes a valid footer. Small code change.
3. **Fix the startup-on-first-message contract** in
   `ros2_policy_node` so the Jetson-side timeout bumps are no longer
   needed. Open since 04-18.
4. Only if Gate 7 fails on sat or stability: iterate stand-v2 on
   mewtwo (action_rate even larger, or a stand-specific env yaml that
   pins `default_joint_pos` to sport-mode resting posture so the
   training target matches the deploy baseline).
