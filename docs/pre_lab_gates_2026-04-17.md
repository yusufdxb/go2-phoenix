# Pre-lab gates — 2026-04-17 (mewtwo)

Candidate for first hardware day: **phoenix-stand** (Option A from
`HARDWARE RUNS/updates/PHOENIX_NEXT_STEPS.md`). Checkpoint:
`checkpoints/phoenix-stand/2026-04-16_22-04-28/model_999.pt`
(symlinked as `checkpoints/phoenix-stand/latest.pt`).

## Gate 0a — sim-rollout bench (phoenix.training.evaluate)

Ran:

    PYTHONPATH=$PWD/src OMNI_KIT_ACCEPT_EULA=YES \
      ~/isaac-sim-venv/bin/python -m phoenix.training.evaluate \
        --checkpoint checkpoints/phoenix-stand/latest.pt \
        --env-config configs/env/stand.yaml \
        --num-envs 16 --num-episodes 5 \
        --metrics-out /tmp/phoenix_eval/stand_rollout.json

Result:

    {
      "num_episodes": 16,
      "mean_episode_return": 44.12958002090454,
      "mean_episode_length_s": 20.0,
      "success_rate": 1.0,
      "failure_rate": 0.0,
      "mean_lin_vel_error": 0.0,
      "mean_ang_vel_error": 0.0
    }

**PASS.** 16/16 episodes at `success_rate=1.0`, full 20 s episode length,
zero failure-detector triggers. Well above the `≥4/5` gate prescribed in
`PHOENIX_NEXT_STEPS.md`.

## Gate 0b — stage phoenix-stand ONNX into phoenix-flat deploy path

`configs/sim2real/deploy.yaml` hard-codes `checkpoints/phoenix-flat/policy.onnx`.
Staged the stand policy into that path so the Jetson side of deploy needs no
config change:

| File | Source hash | Staged hash |
|---|---|---|
| `policy.onnx` | `a66771945c116c4263aead5c817bb70b` | `a66771945c116c4263aead5c817bb70b` ✅ |
| `policy.onnx.data` | `fa2403b64d911d48a09b2eb8697daf02` | `fa2403b64d911d48a09b2eb8697daf02` ✅ |
| `policy.pt` | `a5fc8ab403ce8f19c778f1929353f3be` | `a5fc8ab403ce8f19c778f1929353f3be` ✅ |

The prior `phoenix-flat/policy.onnx` (hash `674ea7ca…`, the broken flat-v0
retrain output) is overwritten — that policy never passed the single-step
bench gate and is now superseded by the stand candidate for the first
hardware day.

## Gate 0c — pre-deploy parity gate (verify_deploy)

Ran:

    PYTHONPATH=$PWD/src python3 -m phoenix.sim2real.verify_deploy \
      --parquet data/failures/synth_slippery_trained.parquet \
      --deploy-cfg configs/sim2real/deploy.yaml \
      --tol 1e-4 \
      --max-steps 200

Result:

    Parity: steps=200 max_diff=3.815e-06 mean_diff=5.449e-07 tol=1.0e-04 -> PASS

**PASS.** Max action drift between ONNX and TorchScript across all 12 joints
is 3.8e-06 — 26× below the 1e-4 tolerance. Staged ONNX matches the
TorchScript policy.

## T7 sync

    rsync -avL \
      checkpoints/phoenix-flat/policy.onnx \
      checkpoints/phoenix-flat/policy.onnx.data \
      checkpoints/phoenix-flat/policy.pt \
      "/media/yusuf/T7 Storage/go2-phoenix/checkpoints/phoenix-flat/"

T7-side hashes match local for all three files. Jetson side of the §1 sync
(scp from CaresLab PC) is unchanged.

## Outstanding

- Lab day §1–§6 (T7 → Jetson sync, offline gates, bridge bringup, fail-closed
  dry-run, Gate 7 live stand, post-stand logging) all remain — those run in
  CaresLab on the GO2.
- `HARDWARE RUNS/updates/PHOENIX_NEXT_STEPS.md` §0a prescribes re-syncing the
  checkpoints to T7 after staging — done (above). Flat path on T7 now carries
  the stand policy.
- Jetson should `scp` the staged ONNX from CaresLab's T7 mount
  (`/media/careslab/T7 Storage/go2-phoenix/checkpoints/phoenix-flat/`).
