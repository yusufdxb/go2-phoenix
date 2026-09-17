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

Correction under report schema 2.0: a count like that 148 is of termination
TICKS. The recorded 2026-09-12 report of the same size is 74 physical falls plus
74 post-reset artifacts, one per fall (see ``is_post_reset_artifact``). That the
earlier run's 74 discards were the same artifacts is inferred, not recomputed:
its report was not kept.

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
import hashlib
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
# tell which layout it is reading. 2.0 separates raw termination TICKS from
# physical terminations (see is_post_reset_artifact) and renames the counts
# accordingly; a 1.0 report counted every tick as a genuine failure.
HARVEST_SCHEMA_VERSION = "2.0"

# What the detector is asked about. Recorded with the measurement so a later
# threshold change is visible in the artifact rather than inferred.
DETECTOR_ID = "phoenix.real_world.failure_detector.FailureDetector"

CLASS_GENUINE = "genuine_failure"
CLASS_TIME_OUT = "time_out"
CLASS_POST_RESET_ARTIFACT = "post_reset_artifact"

# The one rule that decides whether a termination tick is a physical episode
# termination. Stored verbatim in every v2 report so the counts can be re-derived.
POST_RESET_ARTIFACT_RULE = (
    "A termination tick T on environment i is a post_reset_artifact of the previous "
    "termination tick P on the same environment iff all of: (1) T.step_index == "
    "P.step_index + 1, so T is the first control step after the auto-reset Isaac "
    "performed inside P's step; (2) T.terms == P.terms; (3) T.episode_length_steps == 1, "
    "so the episode T ended is exactly one control step old; (4) when both are known, "
    "T.episode_generation == P.episode_generation + 1. An artifact is counted and listed, "
    "attributed to the physical termination that started its chain, and never becomes a "
    "TerminationRecord: it is not shown to the detector, not written, and not a failure."
)


@dataclass(frozen=True)
class TerminationTick:
    """One control step on which Isaac's termination manager reported an env done.

    A tick is not necessarily a physical episode termination; see
    :func:`is_post_reset_artifact`.
    """

    env_index: int
    step_index: int
    terms: tuple[str, ...]
    # Control steps in the episode that ended, from Isaac's episode_length_buf
    # read before the reset. None when unknown.
    episode_length_steps: int | None = None
    # PreResetCapture.generation of the episode that ended. None when unknown.
    episode_generation: int | None = None


@dataclass(frozen=True)
class TickClassification:
    tick: TerminationTick
    classification: str
    # For a post_reset_artifact: step_index of the physical termination whose
    # reset produced it. None otherwise.
    artifact_of_step_index: int | None = None


def is_post_reset_artifact(tick: TerminationTick, previous: TerminationTick | None) -> bool:
    """Apply :data:`POST_RESET_ARTIFACT_RULE`. Pure; used by the live loop and the recompute.

    Why the rule exists. Every physical fall in the 1.0 harvest was recorded
    twice: once with its real window, and again one control step later on the
    same environment with a one-row window and the same ``base_contact`` term.
    The second tick is produced by the harvest's own instrumentation, not by the
    robot: Isaac's contact sensor fills its history lazily from the last PhysX
    step on the first ``.data`` read after it is marked outdated, ``_reset_idx``
    marks it outdated without stepping physics, and the harvest's post-step
    snapshot reads contact data exactly then. The pre-reset base contact force is
    written into the fresh history and ``illegal_contact`` fires again on the next
    tick. Measured with ``scripts/diag_post_reset_termination.py``
    (``data/failures/sim_harvest/diagnostics/post_reset_termination_2026-09-12.json``):
    reading contact data after every step produced 239 second terminations on
    the first step after the environment's own reset, each with episode length 1,
    base height at least 0.396 m and the stale force in history slot 1; reading
    it and then re-resetting the sensor for the environments reset that step
    produced none. The live loop now does the latter
    (:func:`undo_post_reset_contact_read`), so this rule should classify nothing
    in a new run. It stays as the accounting guard, and it is what recomputes the
    1.0 report.

    The rule is deliberately narrow. Every condition must hold, so a genuine
    back-to-back fall, a tick with a different term, or a tick whose episode is
    older than one control step is never merged away. When the episode length is
    unknown the tick is NOT classified as an artifact: an unknown is counted as a
    failure rather than discarded.
    """
    if previous is None or previous.env_index != tick.env_index:
        return False
    if tick.step_index != previous.step_index + 1:
        return False
    if tuple(tick.terms) != tuple(previous.terms):
        return False
    if tick.episode_length_steps != 1:
        return False
    if (
        tick.episode_generation is not None
        and previous.episode_generation is not None
        and tick.episode_generation != previous.episode_generation + 1
    ):
        return False
    return True


class TerminationLedger:
    """Classify termination ticks in step order, one environment history at a time.

    The single classifier both the live harvest loop and
    :func:`recompute_report_from_v1` go through, so the live counts and the
    recomputed historical counts cannot be derived by two different rules.
    """

    def __init__(self, time_out_terms: tuple[str, ...] = TIME_OUT_TERMS) -> None:
        self._time_out_terms = tuple(time_out_terms)
        self._last: dict[int, TerminationTick] = {}
        self._chain_root: dict[int, int] = {}
        self.classified: list[TickClassification] = []

    def observe(self, tick: TerminationTick) -> TickClassification:
        env = tick.env_index
        previous = self._last.get(env)
        if previous is not None and tick.step_index <= previous.step_index:
            raise ValueError(
                f"termination ticks must arrive in step order per environment: env {env} "
                f"step {tick.step_index} after step {previous.step_index}"
            )
        if is_post_reset_artifact(tick, previous):
            verdict = TickClassification(
                tick, CLASS_POST_RESET_ARTIFACT, artifact_of_step_index=self._chain_root[env]
            )
        else:
            self._chain_root[env] = tick.step_index
            if tick.terms and all(term in self._time_out_terms for term in tick.terms):
                verdict = TickClassification(tick, CLASS_TIME_OUT)
            else:
                verdict = TickClassification(tick, CLASS_GENUINE)
        self._last[env] = tick
        self.classified.append(verdict)
        return verdict

    def count(self, classification: str) -> int:
        return sum(1 for v in self.classified if v.classification == classification)


def artifact_entry(verdict: TickClassification) -> dict:
    """JSON form of a post_reset_artifact for the report."""
    tick = verdict.tick
    return {
        "env_index": tick.env_index,
        "step_index": tick.step_index,
        "terms": list(tick.terms),
        "episode_length_steps": tick.episode_length_steps,
        "episode_generation": tick.episode_generation,
        "artifact_of_step_index": verdict.artifact_of_step_index,
        "classification": verdict.classification,
    }


def undo_post_reset_contact_read(env, env_ids) -> bool:
    """Undo what a contact-data read does to an environment reset in the same step.

    ``ContactSensor.data`` fills outdated history lazily from the last PhysX
    step, and ``_reset_idx`` marks a reset environment outdated without
    stepping physics, so a read between the reset and the next physics step
    stores the pre-reset contact force in the new episode's history. Resetting
    the sensor again for those environments restores the zeroed history Isaac's
    own reset produced. Returns False when the scene has no contact sensor.
    """
    try:
        sensor = env.scene["contact_forces"]
    except KeyError:
        return False
    sensor.reset(list(env_ids))
    return True


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
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--recompute-report" in argv:
        # No simulator: re-derive a v2 report from a recorded v1 report.
        return recompute_cli(argv)
    args = parse_args(argv)
    print(f"[harvest] args: {args}", flush=True)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    print("[harvest] app launched", flush=True)
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args), simulation_app, label="harvest")


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
    termination_ticks_seen = 0
    time_out_resets_seen = 0
    unattributed_seen = 0
    records: list[TerminationRecord] = []
    # Every termination tick goes through this one classifier; see
    # is_post_reset_artifact for why a tick is not always a failure.
    ledger = TerminationLedger()
    artifacts: list[TickClassification] = []
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
        # Read inside _reset_idx this is the length of the episode that ENDED,
        # before Isaac zeroes it; the overlay carries that value through.
        s["episode_length"] = to_numpy(env.unwrapped.episode_length_buf).copy()
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
            # Generation of the episode each env is in BEFORE this step, i.e. the
            # episode a termination on this step ends.
            generation_before = dict(capture.generation)
            capture.begin_step()
            obs = policy_obs(env.step(actions))
            reset_now = set(capture.reset_this_step)
            state = capture.overlay(snapshot())
            # snapshot() just read contact-sensor data. For an env Isaac reset
            # inside this step, that read wrote the PRE-reset PhysX contact
            # force into its freshly zeroed history and made base_contact fire
            # again on the next step (see is_post_reset_artifact). Re-reset the
            # sensor for exactly those envs so the next step sees what Isaac
            # intended.
            if reset_now:
                undo_post_reset_contact_read(env.unwrapped, sorted(reset_now))
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

            # A reset with no termination tick is a time-out (truncation):
            # `terminated` excludes time-out terms. It still starts a new
            # episode, so it must clear that env's history below.
            time_out_resets_seen += sum(1 for i in reset_now if not terminated[i])

            for i in range(n_env):
                if not terminated[i]:
                    continue
                termination_ticks_seen += 1
                fired = [term for term, mask in reasons.items() if mask[i]]
                unattributed = not fired
                if unattributed:
                    # `terminated` already excludes time-outs, so a termination
                    # with no named term is a genuine failure the manager did
                    # not attribute. Keep it, loudly. Dropping it would be the
                    # same class of defect as dropping a detector miss.
                    fired = ["unattributed_termination"]
                tick = TerminationTick(
                    env_index=i,
                    step_index=step_idx,
                    terms=tuple(fired),
                    episode_length_steps=int(state["episode_length"][i]),
                    episode_generation=generation_before.get(i, 0),
                )
                verdict = ledger.observe(tick)
                if verdict.classification == CLASS_POST_RESET_ARTIFACT:
                    artifacts.append(verdict)
                    logger.warning(
                        "env %d step %d: termination on the first step after its reset, "
                        "counted as a post-reset artifact of the termination at step %d",
                        i,
                        step_idx,
                        verdict.artifact_of_step_index,
                    )
                    continue
                if verdict.classification == CLASS_TIME_OUT:
                    continue
                if unattributed:
                    unattributed_seen += 1
                    logger.warning(
                        "env %d terminated at step %d with no termination term set; "
                        "retained as an unattributed failure",
                        i,
                        step_idx,
                    )
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
                    episode_length_steps=tick.episode_length_steps,
                    episode_generation=tick.episode_generation,
                    # History is cleared on EVERY reset below, so a window can
                    # no longer span one.
                    window_spans_unobserved_reset=False,
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
            # forward into the next episode's window. Clear on EVERY reset, not
            # only on `terminated`: a time-out reset has terminated=False, and
            # clearing only on termination spliced two episodes into one window
            # (1.0 harvest, sim_fall_0001_env058_step001102: fallen at z=0.18 m
            # until the time-out at row 496, fresh spawn at z=0.40 m on row 497).
            for i in range(n_env):
                if terminated[i] or i in reset_now:
                    history[i].clear()
            if harvested >= args.num_failures:
                break

    capture.close()
    env.close()
    report = build_report(
        records,
        dt_ctrl=dt_ctrl,
        min_pre_onset_rows=args.min_pre_onset_rows,
        termination_ticks_seen=termination_ticks_seen,
        time_out_ticks_seen=ledger.count(CLASS_TIME_OUT),
        unattributed_seen=unattributed_seen,
        post_reset_artifacts=artifacts,
        time_out_resets_seen=time_out_resets_seen,
    )
    report_path = out_dir / "harvest_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(
        format_report(report) + f"\n[harvest] report: {report_path}\n[harvest] out: {out_dir}",
        flush=True,
    )
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
    # --- episode identity (schema 2.0; None on records carried over from 1.0)
    # Control steps in the episode that ended, from Isaac's episode_length_buf.
    episode_length_steps: int | None = None
    # PreResetCapture.generation of that episode.
    episode_generation: int | None = None
    # True when the window holds rows from before a reset other than the
    # termination itself, so it is not one episode. None when not determined.
    window_spans_unobserved_reset: bool | None = None
    # --- detector output, measured AGAINST the ground truth above ----------
    detector_id: str = DETECTOR_ID
    # Whether the detector was actually SHOWN this window. A window too short to
    # evaluate is not a detector miss, and counting it as one silently deflates
    # recall: it produced an exactly-0.500 figure that was an artifact of the
    # window guard rather than a property of the detector.
    detector_evaluated: bool = False
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
    episode_length_steps=None,
    episode_generation=None,
    window_spans_unobserved_reset=None,
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
        episode_length_steps=episode_length_steps,
        episode_generation=episode_generation,
        window_spans_unobserved_reset=window_spans_unobserved_reset,
    )
    if len(rows) < 2:
        record.status = "rejected"
        record.rejected_reason = "window_shorter_than_two_rows"
        return record

    evaluation = evaluate_detector(rows, dt_ctrl, detector)
    record.detector_evaluated = True
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
    termination_ticks_seen,
    time_out_ticks_seen,
    unattributed_seen,
    post_reset_artifacts=(),
    time_out_resets_seen=None,
) -> dict:
    """Summarize the run, including detector recall as a measured quantity.

    ``records`` are physical terminations only. The tick accounting must close:
    every termination tick is exactly one of a time-out tick, a post-reset
    artifact, or a physical termination, and a report that does not add up is
    refused rather than written.
    """
    records = list(records)
    artifacts = [a if isinstance(a, dict) else artifact_entry(a) for a in post_reset_artifacts]
    genuine = len(records)
    if termination_ticks_seen != time_out_ticks_seen + len(artifacts) + genuine:
        raise ValueError(
            f"termination accounting does not close: {termination_ticks_seen} ticks != "
            f"{time_out_ticks_seen} time-out ticks + {len(artifacts)} post-reset artifacts + "
            f"{genuine} physical terminations"
        )
    spliced_written = [
        r.path for r in records if r.status == "written" and r.window_spans_unobserved_reset
    ]
    evaluated = [r for r in records if r.detector_evaluated]
    not_evaluable = [r for r in records if not r.detector_evaluated]
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
        "classification_rule": POST_RESET_ARTIFACT_RULE,
        "counts": {
            # Raw control steps on which the termination manager reported done.
            "termination_ticks_seen": termination_ticks_seen,
            "time_out_ticks": time_out_ticks_seen,
            "post_reset_artifacts": len(artifacts),
            # One per physical episode termination. The failure population.
            "physical_terminations": genuine,
            "unattributed_terminations_retained": unattributed_seen,
            # Resets with no termination tick (truncations). Not part of the
            # tick accounting; None when the run did not observe resets.
            "time_out_resets_observed": time_out_resets_seen,
            "written": len(written),
            "written_with_spliced_window": len(spliced_written),
            "rejected": rejected,
        },
        "written_with_spliced_window": spliced_written,
        "post_reset_artifacts": artifacts,
        "detector": {
            "id": DETECTOR_ID,
            "physical_terminations": genuine,
            # Windows the detector was actually shown. Recall is over THESE, not
            # over every termination: a window too short to evaluate is not a
            # detector miss, and dividing by all of them reports the window
            # guard's behaviour as if it were the detector's.
            "evaluated": len(evaluated),
            "not_evaluable": len(not_evaluable),
            "fired": len(fired),
            # Recall over the windows actually evaluated. Reported, never acted on.
            "recall": (len(fired) / len(evaluated)) if evaluated else None,
            "false_negatives": len(evaluated) - len(fired),
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
        f"[harvest] termination ticks seen: {counts['termination_ticks_seen']}",
        f"[harvest]   time-out ticks (not failures): {counts['time_out_ticks']}",
        f"[harvest]   post-reset artifacts (not failures): {counts['post_reset_artifacts']}",
        f"[harvest]   physical terminations: {counts['physical_terminations']}",
        f"[harvest]   unattributed terminations kept: "
        f"{counts['unattributed_terminations_retained']}",
        f"[harvest] time-out resets observed: {counts['time_out_resets_observed']}",
        f"[harvest] trajectories written: {counts['written']} "
        f"(of which spanning an unobserved reset: {counts['written_with_spliced_window']})",
    ]
    for reason, n in sorted(rejected.items()):
        lines.append(f"[harvest]   rejected, {reason}: {n} (structural, not detector-driven)")
    lines.append(
        f"[harvest] detector recall vs simulator ground truth: "
        f"{'n/a' if recall is None else f'{recall:.3f}'} "
        f"({detector['fired']}/{detector['evaluated']} windows evaluated), "
        f"false negatives: {detector['false_negatives']}"
    )
    lines.append(
        f"[harvest]   windows too short to evaluate (NOT counted as misses): "
        f"{detector['not_evaluable']} of {detector['physical_terminations']}"
    )
    lines.append(f"[harvest]   fired by mode: {detector['fired_by_mode'] or 'none'}")
    if detector["mean_lead_s"] is not None:
        lines.append(
            f"[harvest]   mean lead before simulator termination: "
            f"{detector['mean_lead_s']:.3f} s"
        )
    lines.append(
        "[harvest] detector recall is a MEASUREMENT. A missed failure is kept, " "not discarded."
    )
    return "\n".join(lines)


# ------------------------------------------------------------------ recompute
#
# Re-derives a 2.0 report from a recorded 1.0 report through the SAME
# TerminationLedger the live loop uses. The 1.0 artifacts are read, never
# written.

V1_SCHEMA_VERSION = "1.0"

# Fields a 1.0 record carries, copied verbatim onto the recomputed record.
_V1_RECORD_FIELDS = (
    "env_index",
    "step_index",
    "window_rows",
    "sim_termination_terms",
    "sim_termination_index",
    "sim_termination_time_s",
    "detector_id",
    "detector_evaluated",
    "detector_fired",
    "detector_mode",
    "detector_onset_index",
    "detector_onset_time_s",
    "detector_latency_s",
    "detector_false_negative",
    "onset_index",
    "onset_source",
    "status",
    "rejected_reason",
    "path",
    "schema_version",
    "field_notes",
)


def time_out_reset_steps(
    episode_start_step: int, before_step: int, max_episode_length_steps: int
) -> list[int]:
    """Steps strictly before ``before_step`` at which a time-out reset occurs.

    An episode that begins on control step ``e`` reaches ``episode_length_buf ==
    L`` on step ``e + L - 1``, where Isaac truncates and resets it, and the next
    episode begins on ``e + L``. Assumes every episode starts at length zero, which
    holds for this harvest (no randomized initial episode length outside training).
    """
    if max_episode_length_steps < 1:
        raise ValueError(f"max_episode_length_steps must be >= 1, got {max_episode_length_steps}")
    steps: list[int] = []
    t = episode_start_step + max_episode_length_steps - 1
    while t < before_step:
        steps.append(t)
        t += max_episode_length_steps
    return steps


def window_spans_reset(step_index: int, window_rows: int, reset_steps: list[int]) -> bool:
    """True when a window ending at ``step_index`` holds a row from before a reset.

    A reset on step ``t`` leaves row ``t`` as the pre-reset state and row ``t + 1``
    as the new episode, so the window ``[step_index - window_rows + 1, step_index]``
    is spliced iff some ``t`` in ``reset_steps`` satisfies
    ``window_start <= t < step_index``.
    """
    window_start = step_index - window_rows + 1
    return any(window_start <= t < step_index for t in reset_steps)


def _detector_summary(records: list[TerminationRecord]) -> dict:
    evaluated = [r for r in records if r.detector_evaluated]
    fired = [r for r in evaluated if r.detector_fired]
    by_mode: dict[str, int] = {}
    for r in fired:
        by_mode[r.detector_mode] = by_mode.get(r.detector_mode, 0) + 1
    leads = [-r.detector_latency_s for r in fired if r.detector_latency_s is not None]
    return {
        "evaluated": len(evaluated),
        "fired": len(fired),
        "recall": (len(fired) / len(evaluated)) if evaluated else None,
        "false_negatives": len(evaluated) - len(fired),
        "fired_by_mode": by_mode,
        "mean_lead_s": (sum(leads) / len(leads)) if leads else None,
    }


def recompute_report_from_v1(
    source: str | Path, *, max_episode_length_steps: int, repo_root: Path = REPO_ROOT
) -> dict:
    """Re-derive the corrected accounting of a recorded 1.0 harvest report.

    What a 1.0 report does and does not carry, and how each gap is handled:

    * No episode length. ``window_rows`` stands in for it in the artifact rule.
      The harvest cleared a window on every termination tick, so after a
      termination on step ``s`` a tick on ``s + 1`` has ``window_rows == 1`` exactly
      when its episode is one control step old, which is the only question the
      rule asks. Condition (4), the generation, is skipped: it was not recorded.
    * No record of time-out resets, which did NOT clear the window. They are
      reconstructed per environment from ``max_episode_length_steps`` with
      :func:`time_out_reset_steps`, and every record whose window spans one is
      marked ``window_spans_unobserved_reset``. Detector statistics are reported
      both as recorded and over unspliced windows only, because a spliced window
      can show the detector a fall from the PREVIOUS episode.
    """
    source = Path(source)
    raw = source.read_bytes()
    v1 = json.loads(raw)
    if v1.get("schema_version") != V1_SCHEMA_VERSION:
        raise ValueError(
            f"{source} has schema_version {v1.get('schema_version')!r}, expected "
            f"{V1_SCHEMA_VERSION!r}"
        )
    entries = sorted(v1["trajectories"], key=lambda e: (e["step_index"], e["env_index"]))

    ledger = TerminationLedger()
    records: list[TerminationRecord] = []
    artifacts: list[TickClassification] = []
    episode_start: dict[int, int] = {}
    for entry in entries:
        env = int(entry["env_index"])
        step = int(entry["step_index"])
        start = episode_start.get(env, 0)
        time_outs = time_out_reset_steps(start, step, max_episode_length_steps)
        if time_outs:
            start = time_outs[-1] + 1
        tick = TerminationTick(
            env_index=env,
            step_index=step,
            terms=tuple(entry["sim_termination_terms"]),
            episode_length_steps=int(entry["window_rows"]),
            episode_generation=None,
        )
        verdict = ledger.observe(tick)
        # Every tick, artifact or not, ended an episode with a reset.
        episode_start[env] = step + 1
        if verdict.classification == CLASS_POST_RESET_ARTIFACT:
            artifacts.append(verdict)
            continue
        if verdict.classification == CLASS_TIME_OUT:
            raise ValueError(f"1.0 trajectories never hold time-out ticks: {entry}")
        record = TerminationRecord(**{key: entry[key] for key in _V1_RECORD_FIELDS})
        # The reconstructed episode length is a derived value, not a recorded one.
        record.episode_length_steps = None
        record.window_spans_unobserved_reset = window_spans_reset(
            step, int(entry["window_rows"]), time_outs
        )
        records.append(record)

    v1_counts = v1["counts"]
    v1_time_outs = int(v1_counts.get("time_out_terminations", 0))
    report = build_report(
        records,
        dt_ctrl=v1["control_dt_s"],
        min_pre_onset_rows=v1["min_pre_onset_rows"],
        termination_ticks_seen=len(entries) + v1_time_outs,
        time_out_ticks_seen=v1_time_outs,
        unattributed_seen=sum(
            1 for r in records if "unattributed_termination" in r.sim_termination_terms
        ),
        post_reset_artifacts=artifacts,
        time_out_resets_seen=None,
    )
    clean = [r for r in records if not r.window_spans_unobserved_reset]
    try:
        source_label = str(source.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        source_label = str(source)
    report["generated_by"] = "scripts/harvest_sim_failures.py --recompute-report"
    report["recomputed_from"] = {
        "path": source_label,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": V1_SCHEMA_VERSION,
        "counts_as_recorded": v1_counts,
        "detector_as_recorded": v1["detector"],
    }
    report["reconstruction"] = {
        "max_episode_length_steps": max_episode_length_steps,
        "records_with_window_spanning_a_time_out_reset": sum(
            1 for r in records if r.window_spans_unobserved_reset
        ),
        "detector_over_unspliced_windows_only": _detector_summary(clean),
        "notes": [
            "episode_length_steps for the artifact rule is window_rows (exact for the "
            "question 'is the episode one step old'); episode_generation was not "
            "recorded in 1.0, so rule condition (4) is not applied.",
            "Time-out resets are reconstructed, not recorded, assuming every episode "
            "starts at length 0 on the step after its reset.",
            "Records are copied verbatim from 1.0 (their schema_version stays 1.0); "
            "window_spans_unobserved_reset is the only field computed here.",
            "No 1.0 artifact (report, Parquet, sidecar) was modified.",
        ],
    }
    return report


def write_recomputed_report(
    source: str | Path, out: str | Path, *, max_episode_length_steps: int
) -> dict:
    """Write the recomputed report to a NEW path; never over the source or other evidence."""
    source, out = Path(source), Path(out)
    if out.resolve() == source.resolve():
        raise ValueError("refusing to overwrite the historical report it is recomputed from")
    report = recompute_report_from_v1(source, max_episode_length_steps=max_episode_length_steps)
    text = json.dumps(report, indent=2) + "\n"
    if out.exists() and out.read_text() != text:
        raise FileExistsError(f"refusing to overwrite differing evidence: {out}")
    out.write_text(text)
    return report


def recompute_cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Recompute a 1.0 harvest report under schema 2.0.")
    p.add_argument("--recompute-report", type=Path, required=True, help="1.0 harvest_report.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--max-episode-length-steps",
        type=int,
        required=True,
        help="episode_length_s / control_dt_s of the harvested env, used to "
        "reconstruct time-out resets (20.0 s / 0.02 s = 1000 for configs/env/base.yaml).",
    )
    args = p.parse_args(argv)
    report = write_recomputed_report(
        args.recompute_report, args.out, max_episode_length_steps=args.max_episode_length_steps
    )
    print(format_report(report), flush=True)
    rec = report["reconstruction"]
    print(
        f"[recompute] records whose window spans a time-out reset: "
        f"{rec['records_with_window_spanning_a_time_out_reset']}\n"
        f"[recompute] detector over unspliced windows only: "
        f"{rec['detector_over_unspliced_windows_only']}\n"
        f"[recompute] wrote {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
