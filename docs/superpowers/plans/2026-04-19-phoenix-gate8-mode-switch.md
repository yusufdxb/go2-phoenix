# Phoenix Gate 8 — Mode-Switch Implementation Plan

> **For agentic workers:** use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship a two-policy runtime in `ros2_policy_node.py` that routes `cmd_vel ≈ 0` to stand-v2 and `cmd_vel > threshold` to v3b, with bounded-time linear-blend transitions. No retraining. All offline gates green + hardware G7 + G8a + G8d pass at CaresLab.

**Spec:** `docs/superpowers/specs/2026-04-19-phoenix-gate8-mode-switch-design.md`.

**Branch:** `audit-fixes-2026-04-16` (tip `7e54f47`, synced origin + T7).

**Preconditions:**
- `stand-v2` ONNX already staged at `checkpoints/phoenix-flat/policy.onnx` (Phase 1 2026-04-19) + T7 synced.
- v3b `.pt` checkpoint at `checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt` (local + T7).
- `phoenix.sim2real.verify_deploy` works and has tol `1e-4`.
- 161/161 pytest green on branch.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/phoenix/sim2real/mode_switch.py` | create | Pure-Python state machine (no rclpy). `compute_state(prev, cmd_vel, cfg) → (state, blend_alpha)`. |
| `tests/test_mode_switch.py` | create | 8–10 tests: thresholds, hysteresis, blend progression, boot state, abort re-entry. |
| `src/phoenix/sim2real/ros2_policy_node.py` | modify | If `mode_switch.enabled`, load two sessions, run both per tick, blend targets. |
| `tests/test_ros2_policy_parity.py` | modify or create | Replay pre-recorded cmd sequence through the two-policy path; assert no-NaN + slew clip + exactly 2 transitions. |
| `configs/sim2real/deploy.yaml` | modify | Add `mode_switch` block, default `enabled: false` at commit time. |
| `checkpoints/phoenix-flat/v3b/policy.onnx` | produce | Fresh v3b ONNX export from `2026-04-16_21-39-16/model_999.pt`. |
| `docs/deploy_mode_switch_runbook.md` | create | Lab-day bringup checklist. |

---

## Task 1: Re-export v3b ONNX to `phoenix-flat/v3b/`

- [ ] `./scripts/deploy.sh --checkpoint checkpoints/phoenix-flat/2026-04-16_21-39-16/model_999.pt --out-dir checkpoints/phoenix-flat/v3b/` (or whatever the actual `deploy.sh` CLI is — inspect first).
- [ ] Verify three files exist: `policy.onnx`, `policy.onnx.data`, `policy.pt` (or `.ptjit` / whatever export emits).
- [ ] Run `python -m phoenix.sim2real.verify_deploy --checkpoint <v3b .pt> --onnx checkpoints/phoenix-flat/v3b/policy.onnx --parquet <any trained-distribution parquet>` — expect `max_diff < 1e-4`.
- [ ] Commit: `phoenix-flat/v3b/ is gitignored via checkpoints/*/`; the v3b checkpoints stay local + T7 only. Log hashes into a `docs/v3b_reexport_2026-04-NN.md` for provenance.

## Task 2: Write `mode_switch.py` (pure Python)

- [ ] Define `State = Enum("STAND", "TRANS_TO_WALK", "WALK", "TRANS_TO_STAND")`.
- [ ] Define dataclass `ModeSwitchCfg` with: `enter_walk_thresh`, `enter_stand_thresh`, `yaw_scale`, `transition_ticks`, `initial_state`.
- [ ] Function `cmd_magnitude(cmd_vel: np.ndarray, yaw_scale: float) -> float` — `max(sqrt(vx^2 + vy^2), abs(vyaw) * yaw_scale)`.
- [ ] Function `step(state: State, ticks_in_state: int, cmd_vel: np.ndarray, cfg: ModeSwitchCfg) -> tuple[State, int, float]` — returns `(next_state, next_ticks_in_state, blend_alpha)`. `blend_alpha = 0.0` for STAND/WALK, `ticks_in_state / transition_ticks` for TRANS states.
- [ ] Boot helper `initial_state(cfg) -> tuple[State, int]` — always `(STAND, 0)` regardless of initial cmd (first safe tick publishes stand target).
- [ ] Keep everything importable without rclpy, without onnxruntime.

## Task 3: Write `test_mode_switch.py`

Test cases (all pure-numpy, no Isaac Lab):

- [ ] `test_boots_in_stand` — any starting cmd, initial state is STAND, ticks_in_state 0.
- [ ] `test_enters_walk_on_threshold_cross` — STAND + cmd magnitude 0.20 → TRANS_TO_WALK, ticks reset.
- [ ] `test_stays_in_stand_below_threshold` — STAND + cmd magnitude 0.10 → STAND (hysteresis band).
- [ ] `test_completes_transition_to_walk` — TRANS_TO_WALK for `transition_ticks` steps → WALK.
- [ ] `test_blend_alpha_monotonic` — TRANS_TO_WALK tick k → alpha k/transition_ticks, increasing.
- [ ] `test_exits_walk_on_threshold_return` — WALK + cmd magnitude 0.03 → TRANS_TO_STAND.
- [ ] `test_hysteresis_no_flutter` — cmd magnitude oscillating 0.07 ↔ 0.12 never changes state (stays in whichever mode it started in).
- [ ] `test_yaw_scale_applied` — cmd `[0, 0, 1.0]` with yaw_scale=0.3 → magnitude 0.3 (enters walk).
- [ ] `test_transition_interrupted_by_reverse_cmd` — TRANS_TO_WALK interrupted mid-blend by cmd → 0 should go to TRANS_TO_STAND (blend reverses). Acceptable alternative: complete current transition then re-evaluate. **Pick one, document which.**
- [ ] `test_blend_alpha_bounds` — alpha always in [0.0, 1.0].
- [ ] Run `pytest tests/test_mode_switch.py -q`. Expect 10/10 green.

## Task 4: Wire mode switch into `ros2_policy_node.py`

- [ ] Read `cfg["policy"]["mode_switch"]`; if `enabled: false`, skip entirely (node behaves as today, single session at `onnx_path`). Back-compat baseline.
- [ ] If `enabled: true`: load **both** sessions; maintain two separate `self._last_action_stand` and `self._last_action_walk` (both float32[12], zeroed at init).
- [ ] Per-tick in `_control_step` (only when `startup == "ready"` and not estopped):
  - Compute both obs vectors (can share proprio + use each session's own last_action).
  - Run both `session.run(["action"], {"obs": obs})`.
  - Compute both targets: `default_q + action_scale * action`.
  - Call `mode_switch.step(...)` → new state, new ticks, alpha.
  - Route: STAND → stand_target; WALK → walk_target; TRANS_* → `(1-alpha)*src + alpha*dst` blend.
  - Apply `_clip_to_limits` as today.
  - Publish.
- [ ] Update `self._last_action_stand`/`self._last_action_walk` with the active policy's action (per the per-state rules in spec §"`last_action` contract").
- [ ] Preserve all safety branches exactly as today.
- [ ] Sanity-check: if `enabled: true`, fail-fast at init if either ONNX is missing. No silent fallback.

## Task 5: Extend parity replay test

- [ ] Add or extend `tests/test_ros2_policy_parity.py` (or whatever the existing parity test file is named — search repo).
- [ ] Build a mock `_PhoenixPolicyNode`-equivalent that bypasses rclpy (construct via monkey-patched subs) OR extract the per-tick computation into a pure function first and test that.
- [ ] Replay cmd sequence `[0.0] * 100 + [0.3] * 100 + [0.0] * 100` at 50 Hz.
- [ ] Assert: no NaN in emitted targets; all targets within `default_q ± 2.0 rad`; per-step `|Δtarget| ≤ 0.175 rad`; exactly one STAND→WALK and one WALK→STAND transition, each 25 ticks; targets at start of each transition tick equal prior-state target (continuity); targets at end equal next-state target (continuity).
- [ ] `pytest tests/test_ros2_policy_parity.py -q` — all green.

## Task 6: Deploy config + runbook

- [ ] Add `mode_switch` block to `configs/sim2real/deploy.yaml` per spec §"Config changes". Default `enabled: false` at commit time.
- [ ] Write `docs/deploy_mode_switch_runbook.md` covering: T7→Jetson sync of both ONNX dirs, Jetson offline gates (verify_deploy on both), bringup sequence, how to flip `enabled: true` at bench, how to roll back.

## Task 7: Full suite + commit

- [ ] `pytest tests/ -q` — expect all prior green tests still green + new tests green.
- [ ] `ruff check . && black --check .` per repo convention.
- [ ] Commit one feature commit with message: `mode_switch: two-policy runtime (stand-v2 + v3b) with hysteresis + linear blend`.
- [ ] Push to origin + T7.

## Task 8: Hardware bringup at CaresLab (separate session)

Tracked as its own session / lab day. Outside the scope of this plan.
See spec §"Gate plan → Hardware gates".

## Risk guards during implementation

- Keep `mode_switch.enabled: false` committed in deploy.yaml. Flip to
  `true` only on the lab machine after CI-1/CI-2/CI-3 pass.
- Do not modify `phoenix-flat/policy.onnx` (Gate 7 lifeline — stays as stand-v2).
- If Task 5 parity shows a non-continuity at transition boundaries, lengthen `transition_ticks` to 50 BEFORE touching blend math — the simple fix is usually right.

## Out of scope (explicitly deferred)

- Any change to the failure detector, estop chain, lowcmd bridge, deadman node.
- Reward retraining of either policy. Shipping what we have.
- Policy distillation into a single ONNX.
- Additional modes (e.g. side-step specialist). Start with two; earn the right to add more.
