"""Harvest genuine failure trajectories from simulator rollouts.

Why this exists. The failure-curriculum pool has until now been 18 hand-written
numpy trajectories: a nominal gait row plus a scripted ramp. The H0 delivery
gate (``scripts/h0_delivery_probe.py``) showed that pool actively fights the
method it is supposed to feed. Every trajectory carries exactly 50 stable rows
before onset, failure development time varies by a factor of six across modes,
one mode's signature is a quantity the reset bridge cannot write at all, and the
synthetic gait pose sits about 10 cm below the simulator's own spawn pose, which
injects a confound into every seeded environment.

A rollout failure has none of those problems. It is produced by the policy and
the physics rather than by a generator, every field the replay pipeline needs is
observable, the pre-onset window is as long as the episode was, and the state
distribution is by construction the one the policy actually visits.

The simulator is the ground truth for whether an episode failed. Isaac's
termination manager separates ``terminated`` from ``time_out``, and every
non-timeout termination is a genuine failure whether or not the rule-based
``FailureDetector`` can see it. This script harvests ALL of them and runs the
detector as a MEASUREMENT against that ground truth, never as an inclusion
rule.

That distinction is not cosmetic. A measured run produced 148 terminations, 3
harvested trajectories, and 74 discards whose only defect was that the detector
did not fire. Detector success as a data-inclusion rule biases the pool toward
the failures the detector already understands and throws away the ones most
worth training on.

The single remaining rejection criterion is structural, not detector-driven: a
window whose onset sits closer to the start than ``--min-pre-onset-rows`` has no
usable pre-onset interval, so no seeding strategy and no reset bridge can place
the robot BEFORE the failure. Those are counted and reported separately.

Output is the standard ``phoenix.real_world.trajectory_logger`` Parquet schema,
one file per harvested failure, directly consumable by ``TrajectoryPool``, plus:

* ``<name>.meta.json`` beside each Parquet, holding the SIMULATOR termination
  terms and time separately from the detector's verdict, so the two can never
  be read as one another. The Parquet schema has no column for simulator
  provenance yet (see ``phoenix.real_world.trajectory_logger``), which is the
  only reason this lives in a sidecar.
* ``harvest_report.json`` in the output directory, holding the run-level
  counts and the measured detector recall.

``failure_flag`` in the Parquet marks rows at or after the onset used for
seeding: the detector's onset when it fired, otherwise the simulator's
termination row. ``failure_mode`` stays strictly the detector's label and is
null for a trajectory the detector missed, so a mode-subset filter cannot
silently pick up an unlabelled trajectory. ``onset_source`` in the sidecar says
which of the two was used.

Usage:
    python scripts/harvest_sim_failures.py \
        --checkpoint checkpoints/phoenix-flat-v4/latest.pt \
        --env-config configs/env/flat_perturb.yaml \
        --num-envs 64 --num-failures 24
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Provenance labels for the rows this script writes. Imported rather than
# spelled out so a sim capture cannot drift from the canonical vocabulary in
# the logger and the observation builder.
from phoenix.real_world.trajectory_logger import CAPTURE_SOURCE_SIM  # noqa: E402
from phoenix.sim2real.observation import BASE_LIN_VEL_SOURCE_SIM  # noqa: E402

logger = logging.getLogger("phoenix.harvest")

# An episode that ends by running out of clock is not a failure. Isaac Lab's
# termination manager already separates the two: `terminated` excludes time-out
# terms. Harvest only terminated episodes, and record which term fired.
TIME_OUT_TERMS = ("time_out",)

# Written into every sidecar and into harvest_report.json so a consumer can
# tell which layout it is reading.
HARVEST_SCHEMA_VERSION = "1.0"

# What the detector is asked about. Recorded with the measurement so a later
# threshold change is visible in the artifact rather than inferred.
DETECTOR_ID = "phoenix.real_world.failure_detector.FailureDetector"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-config", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument(
        "--num-failures",
        type=int,
        default=24,
        help="Stop once this many trajectories have been WRITTEN. Terminations "
        "the structural window guard rejects do not count toward it; they are "
        "still measured and reported.",
    )
    p.add_argument(
        "--pre-onset-steps",
        type=int,
        default=400,
        help="Rows of history kept before the terminal step. The H0 gate needs a "
        "pre-onset window wide enough that the seed row is not forced into the "
        "failure itself; 400 rows at 50 Hz is 8 s.",
    )
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument(
        "--push-velocity",
        type=float,
        default=1.0,
        help="Mid-episode push magnitude in m/s. This must knock the robot over "
        "SOMETIMES and late, not always and immediately: a fall at step 5 has no "
        "pre-onset window and the reset bridge rejects it outright.",
    )
    p.add_argument("--push-interval-s", type=float, nargs=2, default=(2.0, 4.0))
    p.add_argument(
        "--min-pre-onset-rows",
        type=int,
        default=100,
        help="Reject a harvested window whose onset is closer than this to the "
        "start. 100 rows at 50 Hz is 2 s of clean pre-failure behaviour. This is "
        "a STRUCTURAL criterion about the window (no pre-onset interval means no "
        "seeding strategy can use it), not a judgement about the detector; "
        "rejections are counted under their own reason in harvest_report.json.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    args = parse_args(argv)
    print(f"[harvest] args: {args}", flush=True)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    print("[harvest] app launched", flush=True)
    try:
        return _run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def _run(args: argparse.Namespace) -> int:  # noqa: ANN001
    from collections import deque
    from importlib import metadata

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from phoenix.real_world.failure_detector import FailureDetector

    # Isaac Lab buffers are torch tensors under PhysX and warp arrays under
    # Newton. Reuse the canonical converter rather than a local one: a naive
    # np.asarray on a warp array raises, and an over-eager one returns zeros.
    from phoenix.real_world.synthesize_failure import _to_numpy as to_numpy
    from phoenix.real_world.trajectory_logger import TrajectoryLogger, TrajectoryStep
    from phoenix.sim2real.export import checkpoint_has_obs_normalizer
    from phoenix.sim_env import build_env_cfg, load_layered_config
    from phoenix.training.agent_cfg import build_runner_cfg
    from phoenix.training.checkpoint import load_runner_checkpoint
    from phoenix.training.episode_outcomes import PreResetCapture, snapshot_manager_state

    env_cfg_loaded = load_layered_config(args.env_config)
    env_cfg = build_env_cfg(env_cfg_loaded)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    # The disturbance must arrive MID-EPISODE. The `perturbation:` overlay
    # modulates `base_external_force_torque`, which is a reset-mode term, so it
    # fires once at spawn: an earlier harvest run with it produced 24 falls
    # between step 5 and step 33, every one with onset at row 0 and therefore
    # rejected by the reset bridge for having no pre-onset interval. Drive the
    # interval-mode `push_robot` term instead, so the robot walks normally and
    # is then shoved.
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg

    push_range = {
        "x": (-args.push_velocity, args.push_velocity),
        "y": (-args.push_velocity, args.push_velocity),
    }
    events = env_cfg.events
    if getattr(events, "push_robot", None) is None:
        events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=tuple(args.push_interval_s),
            params={"velocity_range": push_range},
        )
    else:
        events.push_robot.mode = "interval"
        events.push_robot.interval_range_s = tuple(args.push_interval_s)
        events.push_robot.params["velocity_range"] = push_range
    print(
        f"[harvest] push_robot interval={tuple(args.push_interval_s)}s "
        f"velocity=+/-{args.push_velocity} m/s",
        flush=True,
    )

    task_name = env_cfg_loaded.to_container()["env"]["task_name"]
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    # clip_actions=1.0 matches the reliability harness. Without it the recorded
    # actions are not the actions the deployed policy would emit.
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    # Derive normalization from the checkpoint, never from a constant. A
    # hardcoded True on a checkpoint with no normalizer buffers makes rsl_rl
    # build an untrained EmpiricalNormalization whose forward is (x-0)/(1+1e-2),
    # silently shrinking every observation by 1%. That defect has already cost
    # this project one parity failure, and `evaluate.py` still carries it.
    use_norm = checkpoint_has_obs_normalizer(args.checkpoint)
    print(f"[harvest] empirical_normalization resolved from checkpoint: {use_norm}", flush=True)

    runner_yaml = {
        "run": {
            "name": "harvest",
            "output_dir": "/tmp",
            "log_interval": 1,
            "save_interval": 1,
            "max_iterations": 1,
            "seed": args.seed,
            "device": args.device,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "runner": {"num_steps_per_env": 24, "empirical_normalization": use_norm},
    }
    runner_cfg = build_runner_cfg(runner_yaml, task_name)
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    except ImportError:
        pass
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    info = load_runner_checkpoint(
        runner,
        args.checkpoint,
        load_actor=True,
        load_critic=True,
        load_optimizer=False,
        load_iteration=False,
    )
    if not info.get("actor_match", False):
        raise RuntimeError(f"Actor weights did not round-trip from {args.checkpoint}: {info}")
    policy = runner.get_inference_policy(device=args.device)

    dt_ctrl = float(env.unwrapped.step_dt)
    out_dir = args.out_dir or (REPO_ROOT / "data/failures/sim_harvest")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[harvest] dt={dt_ctrl:.4f}s  out={out_dir}", flush=True)

    n_env = args.num_envs
    window = args.pre_onset_steps
    history: list[deque] = [deque(maxlen=window) for _ in range(n_env)]
    harvested = 0
    terminations_seen = 0
    timeouts_seen = 0
    unattributed_seen = 0
    records: list[TerminationRecord] = []
    contacts_missing_warned = False
    # rsl_rl's wrapper returns either a tensor, a (obs, extras) tuple, or a
    # {group: tensor} dict depending on version. Normalize in one place so the
    # rollout loop below cannot silently feed the policy the wrong group.
    def policy_obs(raw):
        if isinstance(raw, tuple):
            raw = raw[0]
        if isinstance(raw, dict):
            raw = raw.get("policy", next(iter(raw.values())))
        return raw

    obs = policy_obs(env.get_observations())

    def snapshot():
        s = snapshot_manager_state(env.unwrapped, to_numpy)
        # snapshot_manager_state does not carry joint position; add it here so
        # the pre-reset overlay covers it too, otherwise a terminating
        # environment logs its post-reset joint pose.
        s["joint_position"] = to_numpy(env.unwrapped.scene["robot"].data.joint_pos).copy()
        return s

    # Isaac auto-resets a terminating environment INSIDE env.step(). Reading
    # state before the step is one step stale, and reading it after the step
    # returns a fresh spawn for exactly the environments that just failed.
    # PreResetCapture hooks _reset_idx and copies each terminating
    # environment's state before the reset overwrites it; overlay() then
    # substitutes those values back in. Without this the harvested window
    # silently mixes post-reset rows into a failure trajectory: an earlier run
    # produced windows holding 320 rows of a frozen 0.155 m height followed by
    # a 0.400 m spawn spike.
    capture = PreResetCapture(env.unwrapped, snapshot)

    with torch.inference_mode():
        for step_idx in range(args.max_steps):
            actions = policy(obs)
            actions_np = to_numpy(actions).copy()
            capture.begin_step()
            obs = policy_obs(env.step(actions))
            state = capture.overlay(snapshot())
            joint_pos = state["joint_position"]
            contacts = state["contacts"]
            if not np.isfinite(contacts).all() and not contacts_missing_warned:
                logger.warning("contact sensor unavailable; contact_forces will be recorded as 0")
                contacts_missing_warned = True

            for i in range(n_env):
                history[i].append(
                    {
                        "base_pos": state["position"][i].astype(np.float32),
                        # Isaac reports root_quat_w as wxyz; the Parquet schema
                        # and the replay reader are both xyzw.
                        "base_quat": np.asarray(
                            [*state["quaternion"][i][1:], state["quaternion"][i][0]],
                            dtype=np.float32,
                        ),
                        "base_lin_vel_body": state["linear"][i].astype(np.float32),
                        "base_ang_vel_body": state["angular"][i].astype(np.float32),
                        "joint_pos": joint_pos[i].astype(np.float32),
                        "joint_vel": state["joint_velocity"][i].astype(np.float32),
                        "command_vel": state["command"][i][:3].astype(np.float32),
                        "action": actions_np[i].astype(np.float32),
                        "contact_forces": (
                            np.zeros(4, dtype=np.float32)
                            if not np.isfinite(contacts[i]).all()
                            else np.asarray(contacts[i], dtype=np.float32)
                        ),
                        # Isaac's contact sensor reports true Newtons; a row
                        # that fell back to zeros says so rather than passing
                        # a structural zero off as a measurement.
                        "contact_forces_units": (
                            "unmeasured" if not np.isfinite(contacts[i]).all() else "newtons"
                        ),
                    }
                )

            terminated = state["terminated"].astype(bool)
            reasons = {
                name.split(":", 1)[1]: state[name].astype(bool)
                for name in state
                if name.startswith("termination:")
            }

            for i in range(n_env):
                if not terminated[i]:
                    continue
                terminations_seen += 1
                fired = [term for term, mask in reasons.items() if mask[i]]
                if fired and all(term in TIME_OUT_TERMS for term in fired):
                    timeouts_seen += 1
                    continue
                if not fired:
                    # `terminated` already excludes time-outs, so a termination
                    # with no named term is a genuine failure the manager did
                    # not attribute. Keep it, loudly. Dropping it would be the
                    # same class of defect as dropping a detector miss.
                    unattributed_seen += 1
                    logger.warning(
                        "env %d terminated at step %d with no termination term set; "
                        "retained as an unattributed failure",
                        i,
                        step_idx,
                    )
                    fired = ["unattributed_termination"]
                record = harvest_termination(
                    rows=list(history[i]),
                    dt_ctrl=dt_ctrl,
                    out_dir=out_dir,
                    index=harvested,
                    env_index=i,
                    step_index=step_idx,
                    terms=fired,
                    detector=FailureDetector(),
                    logger_cls=TrajectoryLogger,
                    step_cls=TrajectoryStep,
                    min_pre_onset_rows=args.min_pre_onset_rows,
                )
                records.append(record)
                if record.status == "written":
                    harvested += 1
                    verdict = (
                        f"detector={record.detector_mode}@{record.detector_onset_index}"
                        if record.detector_fired
                        else "detector=MISS"
                    )
                    print(
                        f"[harvest] {harvested}/{args.num_failures} "
                        f"env={i} step={step_idx} terms={fired} {verdict} "
                        f"-> {Path(record.path).name}",
                        flush=True,
                    )
                history[i].clear()
                if harvested >= args.num_failures:
                    break
            # Environments Isaac auto-reset must not carry pre-reset history
            # forward into the next episode's window.
            for i in range(n_env):
                if terminated[i]:
                    history[i].clear()
            if harvested >= args.num_failures:
                break

    capture.close()
    env.close()
    report = build_report(
        records,
        dt_ctrl=dt_ctrl,
        min_pre_onset_rows=args.min_pre_onset_rows,
        terminations_seen=terminations_seen,
        timeouts_seen=timeouts_seen,
        unattributed_seen=unattributed_seen,
    )
    report_path = out_dir / "harvest_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(format_report(report) + f"\n[harvest] report: {report_path}\n[harvest] out: {out_dir}",
          flush=True)
    return 0 if harvested else 1


@dataclass
class DetectorEvaluation:
    """What the rule-based detector said about one window.

    A measurement, never a filter. ``fired=False`` is a false negative against
    the simulator's ground truth, not a reason to drop the trajectory.
    """

    fired: bool
    mode: str | None = None
    onset_index: int | None = None
    onset_time_s: float | None = None


@dataclass
class TerminationRecord:
    """One genuine simulator termination, and what the detector made of it.

    The first block is simulator ground truth. The second block is detector
    output evaluated against it. They are kept apart on purpose: conflating
    them is what turned detector success into a data-inclusion rule.
    """

    env_index: int
    step_index: int
    window_rows: int
    # --- simulator ground truth -------------------------------------------
    sim_termination_terms: list[str]
    sim_termination_index: int
    sim_termination_time_s: float
    # --- detector output, measured AGAINST the ground truth above ----------
    detector_id: str = DETECTOR_ID
    detector_fired: bool = False
    detector_mode: str | None = None
    detector_onset_index: int | None = None
    detector_onset_time_s: float | None = None
    # detector_onset_time_s - sim_termination_time_s. Negative means the
    # detector fired BEFORE the simulator ended the episode (the useful case);
    # None when the detector never fired.
    detector_latency_s: float | None = None
    detector_false_negative: bool = True
    # --- what was written --------------------------------------------------
    onset_index: int = 0
    onset_source: str = "simulator_termination"
    status: str = "written"
    rejected_reason: str | None = None
    path: str | None = None
    schema_version: str = HARVEST_SCHEMA_VERSION
    field_notes: dict = field(
        default_factory=lambda: {
            "sim_*": "simulator ground truth, from the Isaac termination manager",
            "detector_*": "rule-based FailureDetector output, a measurement only",
            "onset_index": "row the Parquet failure_flag turns True on",
            "onset_source": "detector | simulator_termination",
        }
    )


def evaluate_detector(rows, dt_ctrl, detector) -> DetectorEvaluation:
    """Run the shared rule-based detector over a window and report what it saw.

    This never decides whether the window is kept. It answers one question:
    would the detector that labels real robot telemetry have seen this
    simulator failure, and if so, when and as what.
    """
    from phoenix.training.episode_telemetry import quat_wxyz_to_euler

    for row_index, row in enumerate(rows):
        quat = row["base_quat"]
        # quat_wxyz_to_euler takes wxyz; the Parquet rows carry xyzw.
        roll, pitch, _yaw = quat_wxyz_to_euler(
            np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=float)
        )
        event = detector.step(
            timestamp_s=row_index * dt_ctrl,
            pitch_rad=float(pitch),
            roll_rad=float(roll),
            base_height_m=float(row["base_pos"][2]),
            cmd_lin_vel=np.asarray(row["command_vel"], dtype=float)[:2],
            actual_lin_vel=np.asarray(row["base_lin_vel_body"], dtype=float)[:2],
        )
        if event is not None:
            return DetectorEvaluation(
                fired=True,
                mode=event.mode.value,
                onset_index=row_index,
                onset_time_s=row_index * dt_ctrl,
            )
    return DetectorEvaluation(fired=False)


def harvest_termination(
    *,
    rows,
    dt_ctrl,
    out_dir,
    index,
    env_index,
    step_index,
    terms,
    detector,
    logger_cls,
    step_cls,
    min_pre_onset_rows,
) -> TerminationRecord:
    """Record one genuine simulator termination, and write it unless unusable.

    The simulator decided this episode failed; that decision is kept whatever
    the detector says. The detector is evaluated against it and the result is
    stored separately.

    The one rejection left is structural: a window whose onset sits closer to
    the start than ``min_pre_onset_rows`` has no pre-onset interval, so no
    seeding strategy can place the robot BEFORE the failure and the reset
    bridge would refuse it. That is a property of the window, not of the
    detector, and it is counted under its own reason.
    """
    rows = list(rows)
    sim_index = len(rows) - 1
    record = TerminationRecord(
        env_index=env_index,
        step_index=step_index,
        window_rows=len(rows),
        sim_termination_terms=list(terms),
        sim_termination_index=sim_index,
        sim_termination_time_s=sim_index * dt_ctrl,
    )
    if len(rows) < 2:
        record.status = "rejected"
        record.rejected_reason = "window_shorter_than_two_rows"
        return record

    evaluation = evaluate_detector(rows, dt_ctrl, detector)
    record.detector_fired = evaluation.fired
    record.detector_mode = evaluation.mode
    record.detector_onset_index = evaluation.onset_index
    record.detector_onset_time_s = evaluation.onset_time_s
    record.detector_false_negative = not evaluation.fired
    if evaluation.fired:
        record.detector_latency_s = evaluation.onset_time_s - record.sim_termination_time_s
        record.onset_index = evaluation.onset_index
        record.onset_source = "detector"
    else:
        record.onset_index = sim_index
        record.onset_source = "simulator_termination"

    if record.onset_index < min_pre_onset_rows:
        record.status = "rejected"
        record.rejected_reason = "no_usable_pre_onset_window"
        return record

    path = out_dir / f"sim_fall_{index:04d}_env{env_index:03d}_step{step_index:06d}.parquet"
    with logger_cls(path) as writer:
        for row_index, row in enumerate(rows):
            failed = row_index >= record.onset_index
            writer.append(
                step_cls(
                    step=row_index,
                    timestamp_s=row_index * dt_ctrl,
                    failure_flag=failed,
                    # Strictly the detector's label. A trajectory the detector
                    # missed stays unlabelled rather than borrowing the
                    # simulator's termination term as if it were a mode.
                    failure_mode=record.detector_mode if failed else None,
                    capture_source=CAPTURE_SOURCE_SIM,
                    base_lin_vel_source=BASE_LIN_VEL_SOURCE_SIM,
                    **{k: v for k, v in row.items()},
                )
            )
    record.path = str(path)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(asdict(record), indent=2))
    return record


def build_report(
    records,
    *,
    dt_ctrl,
    min_pre_onset_rows,
    terminations_seen,
    timeouts_seen,
    unattributed_seen,
) -> dict:
    """Summarize the run, including detector recall as a measured quantity."""
    records = list(records)
    genuine = len(records)
    fired = [r for r in records if r.detector_fired]
    written = [r for r in records if r.status == "written"]
    by_mode: dict[str, int] = {}
    for r in fired:
        by_mode[r.detector_mode] = by_mode.get(r.detector_mode, 0) + 1
    rejected: dict[str, int] = {}
    for r in records:
        if r.status != "written":
            key = r.rejected_reason or "unspecified"
            rejected[key] = rejected.get(key, 0) + 1
    leads = [-r.detector_latency_s for r in fired if r.detector_latency_s is not None]
    return {
        "schema_version": HARVEST_SCHEMA_VERSION,
        "generated_by": "scripts/harvest_sim_failures.py",
        "control_dt_s": dt_ctrl,
        "min_pre_onset_rows": min_pre_onset_rows,
        "counts": {
            "terminations_seen": terminations_seen,
            "time_out_terminations": timeouts_seen,
            "genuine_failures": genuine,
            "unattributed_terminations_retained": unattributed_seen,
            "written": len(written),
            "rejected": rejected,
        },
        "detector": {
            "id": DETECTOR_ID,
            "genuine_failures": genuine,
            "fired": len(fired),
            # Recall over simulator ground truth. Reported, never acted on.
            "recall": (len(fired) / genuine) if genuine else None,
            "false_negatives": genuine - len(fired),
            "fired_by_mode": by_mode,
            "mean_lead_s": (sum(leads) / len(leads)) if leads else None,
            "note": (
                "Detector success is NOT a data-inclusion rule. Every genuine "
                "simulator termination above is retained; recall is a property "
                "of the detector, measured here against simulator ground truth."
            ),
        },
        "trajectories": [asdict(r) for r in records],
    }


def format_report(report: dict) -> str:
    counts = report["counts"]
    detector = report["detector"]
    recall = detector["recall"]
    rejected = counts["rejected"]
    lines = [
        "",
        f"[harvest] terminations seen: {counts['terminations_seen']}",
        f"[harvest]   of which time-out (not failures): {counts['time_out_terminations']}",
        f"[harvest]   genuine failures retained: {counts['genuine_failures']}",
        f"[harvest]   unattributed terminations kept: "
        f"{counts['unattributed_terminations_retained']}",
        f"[harvest] trajectories written: {counts['written']}",
    ]
    for reason, n in sorted(rejected.items()):
        lines.append(f"[harvest]   rejected, {reason}: {n} (structural, not detector-driven)")
    lines.append(
        f"[harvest] detector recall vs simulator ground truth: "
        f"{'n/a' if recall is None else f'{recall:.3f}'} "
        f"({detector['fired']}/{detector['genuine_failures']}), "
        f"false negatives: {detector['false_negatives']}"
    )
    lines.append(f"[harvest]   fired by mode: {detector['fired_by_mode'] or 'none'}")
    if detector["mean_lead_s"] is not None:
        lines.append(
            f"[harvest]   mean lead before simulator termination: "
            f"{detector['mean_lead_s']:.3f} s"
        )
    lines.append(
        "[harvest] detector recall is a MEASUREMENT. A missed failure is kept, "
        "not discarded."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
