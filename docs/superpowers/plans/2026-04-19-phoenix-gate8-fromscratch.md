# Phoenix Gate 8 From-Scratch Retrain Implementation Plan

> **For agentic workers:** use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Produce `phoenix-flat-scratch` — v3b replacement trained from random init with `slew_sat_hinge_l2` active from iter 0 — that clears all Gate 8 flat.yaml gates (slew <5%, lin_vel_err ≤0.10, ang_vel_err ≤0.10, 32/32 survival).

**Spec:** `docs/superpowers/specs/2026-04-19-phoenix-gate8-fromscratch-design.md`.

**Branch:** `audit-fixes-2026-04-16` (current tip `6a789d1`, synced origin+T7).

**Preconditions:**
- `slew_sat_hinge_l2` reward + `_NEW_TERM_FACTORIES` wiring already shipped (Phase 2b, commits `52a0986`/`f5c9af9`).
- Reward section wiring in `go2_env_cfg._apply_rewards` shipped (Phase 0).
- `phoenix.training.evaluate` warp-array fix shipped (2026-04-17).
- 161/161 tests green on branch.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `configs/env/flat_scratch.yaml` | create | env overlay — flat + `slew_sat_hinge: -50.0` |
| `configs/train/ppo_flat_scratch.yaml` | create | PPO config, no resume, 2500 iters, init_noise_std=0.5 |
| `checkpoints/phoenix-flat-scratch/<ts>/` | produce | training artifacts (gitignored, rsync'd to T7 on green) |
| `docs/rollout_flat_scratch_{stand,flat}_2026-04-19.json` | produce | gate evidence |
| `docs/retrain_flat_scratch_2026-04-19.md` | produce | post-mortem (before/after vs v3b; negative result allowed) |
| `checkpoints/phoenix-flat/gate8/` | produce on pass | staged ONNX (sibling of Gate 7 stand-v2 `policy.onnx`) |

---

## Task 1: Write config pair

- [ ] Create `configs/env/flat_scratch.yaml` verbatim from spec §Design.
- [ ] Create `configs/train/ppo_flat_scratch.yaml` verbatim from spec §Design.
- [ ] Dry-run the config loader: `python -c "from phoenix.sim_env.go2_env_cfg import build_env_cfg; cfg = build_env_cfg('configs/env/flat_scratch.yaml'); print(cfg.rewards.action_rate_l2.weight, cfg.rewards.slew_sat_hinge.weight)"` (or equivalent via the test harness). Expect `-0.05` and `-50.0`.
- [ ] Run `pytest tests/ -q` — expect 161/161 still green (configs don't change wired code paths).

## Task 2: Launch training

- [ ] Verify disk: `df -h ~/workspace/go2-phoenix/checkpoints/` — expect >10 GB free. 2500 iters × save_interval=100 produces ~25 checkpoints @ ~5 MB each.
- [ ] Verify GPU free: `nvidia-smi` — no other training on cuda:0.
- [ ] Launch via `scripts/train.sh configs/train/ppo_flat_scratch.yaml` (or whichever entrypoint Phoenix uses). **This is a ~2hr commitment on cuda:0.**
- [ ] Expected run dir: `checkpoints/phoenix-flat-scratch/<YYYY-MM-DD_HH-MM-SS>/`.
- [ ] Monitor: first 100 iters should show reward > 0 and no NaNs. If either fails in the first 10 min, kill and diagnose; do not let a broken run burn the full 2hr.

## Task 3: Evaluation gates

After training completes:

- [ ] Confirm `latest.pt` symlink points at `model_2499.pt` (or closest 100-aligned save).
- [ ] **Gate G2/G3/G4/G5** — primary: `python -m phoenix.training.evaluate --env configs/env/flat.yaml --checkpoint <run>/latest.pt --num-envs 16 --num-episodes 32 --seed 42 --out docs/rollout_flat_scratch_flat_2026-04-19.json`.
  - Read back JSON; log `slew_saturation_pct`, `mean_lin_vel_err`, `mean_ang_vel_err`, `success_rate`, `mean_episode_length_s`.
- [ ] **Gate G1** — secondary: same command with `--env configs/env/stand.yaml`. Expect survival but slew may exceed 5% (OK — stand-v2 owns cmd=0).
- [ ] Write comparison table into post-mortem alongside v3b + flat-v3b-ft + flat-slewhinge rows from prior post-mortems.

## Task 4: Branch decision

**If G2 AND G3 AND G4 AND G5 all pass (scratch succeeds):**
- [ ] Export ONNX: `python -m phoenix.sim2real.export --checkpoint <run>/latest.pt --out checkpoints/phoenix-flat/gate8/policy.onnx`.
- [ ] **Gate G6** — `python -m phoenix.sim2real.verify_deploy --checkpoint <run>/latest.pt --onnx checkpoints/phoenix-flat/gate8/policy.onnx --parquet data/replay/synth_slippery_trained.parquet`. Expect `max_diff < 1e-4`.
- [ ] Commit configs + metrics + post-mortem + gate8 ONNX staging note. Do NOT touch `checkpoints/phoenix-flat/policy.onnx` (stand-v2 stays at Gate 7).
- [ ] rsync `checkpoints/phoenix-flat-scratch/` + `checkpoints/phoenix-flat/gate8/` to T7.
- [ ] Tag `v0.3.0-gate8-candidate`; push branch to origin + T7.
- [ ] Update memory (`project_go2_phoenix.md`) with the result.
- [ ] Write vault daily log entry.

**If any of G2–G5 fails (scratch is also a negative result):**
- [ ] No ONNX export. No T7 sync of checkpoints (save disk).
- [ ] Keep run dir locally for diagnosis.
- [ ] Commit configs + metrics + post-mortem explaining the failure.
- [ ] **Do not retry in-plan.** A second negative result means the whole "v3b + slew_sat_hinge" design space is exhausted. Escalate to a new spec that either (a) commits to deploy-layer mode-switch + stand-v2 only, (b) redesigns the hinge reward (different threshold? per-motor clip-aware LR?), or (c) switches to an entirely different task spec (joint-velocity command instead of base-velocity, etc.).
- [ ] Update memory.

## Task 5: Risk guards during run

- [ ] `save_interval=100` means a crash at iter 2199 still yields `model_2100.pt` — always evaluate the latest available checkpoint.
- [ ] Watch for NaN in tensorboard `Episode_Reward/slew_sat_hinge` — indicates the reward is exploding. If first 200 iters show mean hinge penalty > |reward total|, weight is too strong — kill, halve to `-25`, restart with fresh timestamp. Document the change in post-mortem.
- [ ] Do not interrupt the run to "peek." Evaluation requires the finished checkpoint + symlink consistency; interrupting mid-save can leave `latest.pt` dangling.

## Out of scope (explicitly deferred)

- Deploy-layer mode-switch (separate spec if chosen as fallback).
- Reward-weight sweep inside this plan (spec §"What is NOT in this spec").
- Architecture / observation-space changes.
- Gate 9+ (hardware dryrun on Jetson with Gate 8 ONNX) — covered by the existing lab-day playbook once G6 passes.
