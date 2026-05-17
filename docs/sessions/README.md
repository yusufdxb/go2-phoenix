# Sessions archive

Historical lab-day notes, retrain post-mortems, pre-lab gate metrics,
and rollout JSON dumps from the 2026-04-14 through 2026-04-22
Phoenix work cycle. Kept for traceability; not meant as current
documentation.

**Heads-up on internal links.** These files were written when they
lived at `docs/<name>.md`. Many of them reference each other as
`docs/<sibling>.md` (without the `sessions/` segment). When following
those links, mentally substitute `docs/sessions/<sibling>.md`. Files
that have moved together resolve correctly when you append `sessions/`.

## Contents

### Lab sessions
- `lab_prompt_2026-04-16.md` — prep doc for first hardware day
- `lab_session_2026-04-16.md` — narrative of the 04-16 session
- `lab_findings_2026-04-16.md`, `lab_findings_2026-04-20.md`, `lab_findings_2026-04-21.md` — per-session findings
- `LAB_CARD_2026-04-22.md` — superseded by `docs/LAB_CARD_NEXT.md`
- `deploy_session_prompt_2026-04-14.md` — Jetson-side Claude prompt for early deploy attempts

### Pre-lab gates
- `pre_lab_gates_2026-04-17.md` — Gate 0a / 0b / 0c metrics
- `pre_lab_stand_rollout_2026-04-17.json`
- `pre_lab_stand_v3_rollout_2026-04-21.json` (+ `_4096envs.json` variant)

### Retrain post-mortems (2026-04-19 sprint)
- `retrain_flat_scratch_2026-04-19.md`
- `retrain_flat_slewhinge_2026-04-19.md`
- `retrain_flat_v3b_ft_2026-04-19.md`
- `retrain_stand_v2_2026-04-19.md`

### Rollout JSON dumps
13 `rollout_*.json` files from the v3b vs v4 vs slewhinge vs scratch
comparison. Each is the structured output of
`phoenix.training.evaluate`.

### Diagnostics
- `deploy_blocked_2026-04-14.md` — early dryrun blockers
- `dryrun_findings_2026-04-14.md` — slew-saturation root cause
- `train_stand_v2_2026-04-18.log` — training log
