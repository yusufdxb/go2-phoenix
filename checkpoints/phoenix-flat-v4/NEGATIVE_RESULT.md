# phoenix-flat-v4 — negative result, 2026-04-17

**Do not export, do not stage, do not deploy.** This directory holds a training run that was superseded by v3b before it started. Kept on disk as a reference point only.

**What this run was:** 5000 iters @ 10240 envs on `configs/env/flat_v4.yaml` + `configs/train/ppo_flat_v4.yaml`, started 2026-04-17 21:42:51, finished 2026-04-17 22:27:35 (44:44 wall).

**What changed from the v3b config (`configs/train/ppo_flat.yaml`):**
- `entropy_coef` 0.005 → 0.01
- `init_noise_std` 0.5 → 1.0
- `rel_standing_envs` 0.02 → 0.15 (via `configs/env/flat_v4.yaml`)
- `max_iterations` 2500 → 5000

**Why it was run:** Hypothesis was that v3b had converged into a basin that could be improved by more exploration (entropy) and a stronger canonical-stand attractor (rel_standing_envs).

**What happened:** Training converged cleanly — reward climbed 17 → 35, 0 crashes — but the resulting policy is meaningfully worse than v3b on the apples-to-apples `configs/env/flat.yaml` eval (16 envs × 32 episodes, `phoenix.training.evaluate` with the post-fix warp-array tracking-error):

| metric                   | v3b   | v4    |
|--------------------------|-------|-------|
| mean_episode_return      | 39.50 | 34.50 |
| mean_lin_vel_error (m/s) | 0.091 | 0.110 |
| mean_ang_vel_error (rad/s)| 0.087 | 0.145 |
| success_rate             | 32/32 | 32/32 |
| mean_episode_length_s    | 20.0  | 20.0  |

**Diagnosis:** The three knob changes traded velocity-tracking quality for no measurable robustness gain. Higher entropy made action commitments fuzzier (worst on yaw, where reward weight is 0.5 vs 1.0 for linear). `rel_standing_envs=0.15` reallocated 15% of episodes to stand, producing 15% less velocity-tracking gradient signal. Reward plateaued by iter ~1500 — the extra 3500 iters did nothing.

**Status:** v3b stays the Gate 8 candidate (exported + parity-gated at `checkpoints/phoenix-flat/v3b/`). This v4 directory is preserved as empirical evidence against re-trying the same knob combination.

**Commit:** `61fae38` on branch `audit-fixes-2026-04-16` includes the configs, metrics, and the `evaluate.py` warp-array fix that made the comparison measurable in the first place.

**Session notes:** Daily log `Daily Claude Logs/2026-04-17.md` in the Obsidian vault.
