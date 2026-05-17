# Phoenix Lab Runbook — 2026-04-17

**Target:** Unitree GO2 + Jetson companion, ROS 2 Humble  
**Branch on T7 / Jetson clone:** `deploy-run-2026-04-14`  
**Shipped deploy artifact:** `checkpoints/phoenix-flat/policy.onnx`  
**Deploy config:** `configs/sim2real/deploy.yaml`

---

## Scope

This runbook is hardware-only. Local cleanup, training, export, replay redesign, and broad refactors are out of scope. The local `mewtwo` fixes are assumed to have landed already:

- replay variation application is partially real
- projected-gravity deploy bug is fixed
- estop / sensor freshness is fail-closed locally
- dry-run tooling resolves ONNX path from `deploy.yaml`
- docs now match the local code more closely

Tomorrow is about hardware proof only.

---

## Do Not

- Do not train, fine-tune, or export a new policy on Jetson
- Do not weaken estop, deadman, or watchdog behavior
- Do not merge to `main`
- Do not push to GitHub from Jetson
- Do not continue past a failed gate because the robot “looks fine”

If a hardware-only defect appears, make the smallest safe fix needed, document it, and keep scope tight.

---

## Success Criteria

The narrowest honest success claim for this session is:

1. `verify_deploy` passes on Jetson
2. `scripts/dry_run_policy.py` passes on Jetson
3. wireless/controller deadman proves `/phoenix/estop` behavior on the real ROS graph
4. dry-run bridge output is sane and not pathologically clip-saturated
5. one stand-only live run completes safely
6. one real hardware parquet is written and readable

Ground motion is optional and only allowed after all previous gates pass.

---

## 0. Preflight Sync

```bash
cd "/media/T7 Storage" && tar c go2-phoenix | \
  ssh unitree@192.168.0.2 "cd ~/yusuf && rm -rf go2-phoenix && tar x"

ssh unitree@192.168.0.2 "cd ~/yusuf/go2-phoenix && git status && git log -1 --oneline && git branch --show-current"
```

Expect:

- repo at `~/yusuf/go2-phoenix`
- branch `deploy-run-2026-04-14`
- HEAD newer than or equal to the flat-policy switch

Then verify the actual shipped deploy config on Jetson:

```bash
cd ~/yusuf/go2-phoenix
python3 - <<'PY'
import yaml
cfg = yaml.safe_load(open('configs/sim2real/deploy.yaml'))
print(cfg['policy']['onnx_path'])
print(cfg['policy']['torchscript_path'])
print(cfg['policy']['obs_pad_zeros'])
print(cfg['safety'])
PY
```

Required:

- `policy.onnx_path == checkpoints/phoenix-flat/policy.onnx`
- `policy.obs_pad_zeros == 0`

If not, stop and fix the repo sync first.

---

## 1. Offline Deploy Gate

Before touching the robot, confirm the shipped artifacts are internally consistent.

```bash
cd ~/yusuf/go2-phoenix
python3 -m phoenix.sim2real.verify_deploy \
  --parquet data/failures/synth_slippery_trained.parquet \
  --deploy-cfg configs/sim2real/deploy.yaml \
  --tol 1e-4 \
  --max-steps 200
```

Gate:

- must return `PASS`
- non-zero exit code means stop

---

## 2. Jetson Dry-Run Policy Gate

This checks the ROS-side policy node on Jetson with synthetic inputs.

```bash
source /opt/ros/humble/setup.bash
cd ~/yusuf/go2-phoenix
PYTHONPATH=$PWD/src python3 scripts/dry_run_policy.py --config configs/sim2real/deploy.yaml
```

Gate:

- must pass all scenarios
- if this fails, do not proceed to real ROS / hardware

---

## 3. Real ROS Topic Surface Check

Before live actuation, verify the ROS graph you actually have.

Required checks:

- `/lowstate` exists and is healthy
- `/phoenix/estop` topic exists once the estop adapter is launched
- bridge and policy topic names match `deploy.yaml`
- no stale command names or package names are being used from older prompts

Useful commands:

```bash
source /opt/ros/humble/setup.bash
ros2 topic list | sort
ros2 topic info /lowstate
ros2 topic hz --qos-profile sensor_data /lowstate
```

If the expected topic surface is missing, stop and fix the ROS bringup, not the runbook.

---

## 4. Wireless Deadman / Estop Gate

Prove the real deadman path before enabling motors.

Terminal A:

```bash
source /opt/ros/humble/setup.bash
cd ~/yusuf/go2-phoenix
PYTHONPATH=$PWD/src python3 -m phoenix.sim2real.wireless_estop_node
```

Terminal B:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /phoenix/estop
```

Gate all three:

1. hold deadman button -> `/phoenix/estop` becomes `False`
2. release deadman button -> `/phoenix/estop` becomes `True`
3. controller silence / disconnect -> `/phoenix/estop` returns `True` within timeout

If any fail, stop. No motor actuation.

---

## 5. Dry-Run Saturation Gate on Real ROS Graph

Goal: confirm the flat-v0 policy does not recreate the old rough-v0 saturation behavior.

Terminal A:

```bash
cd ~/yusuf/go2-phoenix
bash scripts/dryrun_pipeline.sh 30
```

Terminal B:

```bash
source /opt/ros/humble/setup.bash
cd ~/yusuf/go2-phoenix
python3 scripts/lowcmd_inspect.py 6 /lowcmd_dry
```

Gate:

- sane publish rates
- no incoherent thrashing
- output not pinned at the slew cap
- estop freshness still behaves fail-closed under the real ROS timing

If this fails, do not enable motors.

---

## 6. Stand-Only Live Run

Only proceed if Sections 1–5 all passed.

Bring up the live path explicitly.

Terminal A:

```bash
source /opt/ros/humble/setup.bash
cd ~/yusuf/go2-phoenix
PYTHONPATH=$PWD/src python3 -m phoenix.sim2real.lowstate_bridge_node
```

Terminal B:

```bash
source /opt/ros/humble/setup.bash
cd ~/yusuf/go2-phoenix
PYTHONPATH=$PWD/src python3 -m phoenix.sim2real.lowcmd_bridge_node --live
```

Terminal C:

```bash
source /opt/ros/humble/setup.bash
cd ~/yusuf/go2-phoenix
PYTHONPATH=$PWD/src python3 -m phoenix.sim2real.ros2_policy_node \
  --config configs/sim2real/deploy.yaml \
  --onnx checkpoints/phoenix-flat/policy.onnx \
  --log-parquet data/hardware/run_2026-04-17_stand.parquet
```

Gate:

- stand-only behavior is clean
- no unsafe snap or thrash
- deadman release / disconnect aborts cleanly in live mode
- parquet is written and readable

Do not lower the bar here. This is the main success condition.

---

## 7. Optional Ground Run

Only if stand-only live validation is clean.

Optional next step:

- limited ground motion
- operator approval first
- same deploy path
- clearly separate this result from the stand result in notes

If the robot walks safely and logs a parquet, that is valuable. If not, the stand run is still the core proof.

---

## 8. Post-Run Artifacts

Record what actually happened.

Suggested artifacts:

- `data/hardware/run_2026-04-17_stand.parquet`
- any optional ground-run parquet
- `docs/lab_findings_2026-04-17.md`

Commit only to the deploy branch on the Jetson clone:

```bash
git add data/hardware/run_2026-04-17*.parquet docs/lab_findings_2026-04-17.md
git commit -m "lab run 2026-04-17: stand validation and hardware artifacts"
```

Then sync hardware artifacts back to T7:

```bash
rsync -av --delete ~/yusuf/go2-phoenix/data/hardware/ \
  /media/T7\ Storage/LABWORK/PHOENIX/hardware_runs/
```

---

## Failure Modes To Watch

| Symptom | Likely Cause | Action |
|---|---|---|
| `verify_deploy` max diff is large | ONNX / TorchScript drift or wrong artifact | Stop. Fix on `mewtwo`. |
| Dry-run outputs saturate badly | wrong policy or config drift | Check `deploy.yaml` paths and `obs_pad_zeros: 0`. |
| `/phoenix/estop` does not flip on release/silence | deadman path broken | Stop. Do not enable motors. |
| Policy node stays silent forever | estop or sensor preconditions never satisfied | Inspect topic freshness and publisher graph. |
| Live stand run snaps or thrashes | policy/runtime mismatch or bridge issue | Abort immediately, preserve logs, stop session. |
| Parquet not written | path / permissions / node exit issue | Fix only if safe; otherwise document blocker. |

---

## End-Of-Session Checklist

- [ ] `verify_deploy` passed
- [ ] `scripts/dry_run_policy.py` passed on Jetson
- [ ] wireless/controller deadman validated on the real ROS graph
- [ ] dry-run saturation gate passed
- [ ] stand-only live run completed safely
- [ ] at least one readable hardware parquet captured
- [ ] findings written down with clear proof boundaries
- [ ] artifacts synced back to T7
