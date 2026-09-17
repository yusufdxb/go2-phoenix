"""Write one trajectory Parquet per replayed variant environment.

``phoenix.replay.reconstruct`` rolls N perturbed copies of a seeded failure
forward in Isaac Sim. Until 2026-09-17 it wrote a single ``replay_summary.json``
and no trajectories at all, so ``scripts/loop_closure.sh`` aborted at its
replay stage by design: there was nothing to augment the curriculum pool with,
and the "pool of 1 real + N variants" claim in the report would have been false.

This module is the part of that fix which does not need the simulator. It takes
plain numpy arrays of shape ``(num_envs, ...)`` per control step and fans them
out into one :class:`phoenix.real_world.TrajectoryLogger` per env, so the
variants are written in exactly the schema the curriculum reader already
consumes. Keeping it simulator-free is what makes the row-level behaviour
(failure labelling, per-env isolation, termination bookkeeping) testable in CI
rather than only on a GPU.

Every row it writes is labelled ``capture_source="sim"`` and the files declare
``position_frame="env_local"``, which is what the variants actually are: states
measured from each environment's own origin. A consumer therefore never has to
guess, and a hardware capture (boot-relative odometry) can never be confused
for one of these.

Failure labelling uses
:func:`phoenix.real_world.failure_detector.sim_analysis_thresholds`, the
simulator analysis bar, NOT the hardware intervention threshold. The two were
the same dataclass default until 2026-09-17; scoring replayed sim variants at
the hardware bar would make their failure counts incomparable with every
recorded sim result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phoenix.real_world.failure_detector import (
    FailureDetector,
    FailureMode,
    sim_analysis_thresholds,
)
from phoenix.real_world.trajectory_logger import (
    CAPTURE_SOURCE_SIM,
    POSITION_FRAME_ENV_LOCAL,
    TrajectoryLogger,
    TrajectoryStep,
)
from phoenix.sim2real.hw_probe import roll_pitch_from_quat_xyzw
from phoenix.sim2real.observation import BASE_LIN_VEL_SOURCE_SIM

#: Contact-force units when the caller did not read a contact sensor. The
#: column is then a structural zero, not measured Newtons, and says so: the
#: same choice ``phoenix.real_world.synthesize_failure`` makes.
CONTACT_FORCES_UNMEASURED = "unmeasured"

#: ``failure_onset_source`` for a row whose flag came from the rule-based
#: detector, matching the vocabulary in the logger's module docstring.
ONSET_SOURCE_DETECTOR = "detector"
ONSET_SOURCE_SIM_TERMINATION = "simulator_termination"


@dataclass(frozen=True)
class VariantResult:
    """What one variant env produced. Written to ``variants_index.json``."""

    env_index: int
    path: Path
    rows: int
    failed: bool
    failure_mode: str | None
    failure_step: int | None
    terminated_step: int | None

    def to_json(self) -> dict:
        return {
            "env_index": self.env_index,
            "path": str(self.path),
            "rows": self.rows,
            "failed": self.failed,
            "failure_mode": self.failure_mode,
            "failure_step": self.failure_step,
            "terminated_step": self.terminated_step,
        }


def _as_2d(value, name: str, num_envs: int, width: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (num_envs, width):
        raise ValueError(f"{name} must have shape ({num_envs}, {width}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


class VariantTrajectoryWriter:
    """Fan per-step simulator tensors out into one Parquet per env.

    ``base_height_source`` decides whether the collapse mode is available. Sim
    captures have a true env-origin-relative base height, so it is passed and
    collapse can fire; pass ``None`` and the detector reports collapse as
    unavailable rather than silently never firing.
    """

    def __init__(
        self,
        output_dir: str | Path,
        num_envs: int,
        *,
        control_dt: float,
        prefix: str = "variant",
        row_group_size: int = 512,
        contact_forces_units: str = CONTACT_FORCES_UNMEASURED,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_envs = int(num_envs)
        self.control_dt = float(control_dt)
        self.contact_forces_units = str(contact_forces_units)
        self.paths = [self.output_dir / f"{prefix}_{i:03d}.parquet" for i in range(self.num_envs)]
        self._loggers = [
            TrajectoryLogger(
                path,
                row_group_size=row_group_size,
                position_frame=POSITION_FRAME_ENV_LOCAL,
            )
            for path in self.paths
        ]
        self._detectors = [FailureDetector(sim_analysis_thresholds()) for _ in range(num_envs)]
        self._rows = [0] * self.num_envs
        self._failure_mode: list[str | None] = [None] * self.num_envs
        self._failure_step: list[int | None] = [None] * self.num_envs
        self._terminated_step: list[int | None] = [None] * self.num_envs
        # An env stops recording at its first termination. Rows after a reset
        # belong to a different episode and would silently splice two episodes
        # into one file.
        self._closed_env = [False] * self.num_envs
        self._closed = False

    @property
    def active_envs(self) -> int:
        return sum(1 for done in self._closed_env if not done)

    def append_step(
        self,
        step_index: int,
        *,
        base_pos,
        base_quat_xyzw,
        base_lin_vel_body,
        base_ang_vel_body,
        joint_pos,
        joint_vel,
        command_vel,
        action,
        contact_forces,
        base_height=None,
        terminated=None,
    ) -> None:
        """Record one control step for every env still running.

        ``base_height`` is the validated ground-relative trunk height; pass
        ``None`` to mark the collapse mode unavailable. ``terminated`` is the
        simulator's per-env done flag; an env that terminates records this row
        and then stops, so one file is exactly one episode.
        """

        if self._closed:
            raise RuntimeError("cannot append to a closed VariantTrajectoryWriter")
        n = self.num_envs
        pos = _as_2d(base_pos, "base_pos", n, 3)
        quat = _as_2d(base_quat_xyzw, "base_quat_xyzw", n, 4)
        lin = _as_2d(base_lin_vel_body, "base_lin_vel_body", n, 3)
        ang = _as_2d(base_ang_vel_body, "base_ang_vel_body", n, 3)
        jpos = _as_2d(joint_pos, "joint_pos", n, 12)
        jvel = _as_2d(joint_vel, "joint_vel", n, 12)
        cmd = _as_2d(command_vel, "command_vel", n, 3)
        act = _as_2d(action, "action", n, 12)
        contact = _as_2d(contact_forces, "contact_forces", n, 4)
        heights = None if base_height is None else _as_2d(base_height, "base_height", n, 1)[:, 0]
        dones = (
            np.zeros(n, dtype=bool)
            if terminated is None
            else np.asarray(terminated, dtype=bool).reshape(n)
        )

        timestamp = float(step_index) * self.control_dt
        for i in range(n):
            if self._closed_env[i]:
                continue
            q = quat[i]
            norm = float(np.linalg.norm(q))
            if not np.isfinite(norm) or norm < 1e-10:
                raise ValueError(f"env {i} quaternion must be finite and nonzero")
            qx, qy, qz, qw = (q / norm).tolist()
            roll, pitch = roll_pitch_from_quat_xyzw(qx, qy, qz, qw)
            event = self._detectors[i].step(
                timestamp_s=timestamp,
                pitch_rad=float(pitch),
                roll_rad=float(roll),
                base_height_m=None if heights is None else float(heights[i]),
                cmd_lin_vel=cmd[i][:2],
                actual_lin_vel=lin[i][:2],
            )
            failed = event is not None
            if failed and self._failure_step[i] is None:
                self._failure_mode[i] = (
                    event.mode.value if isinstance(event.mode, FailureMode) else str(event.mode)
                )
                self._failure_step[i] = step_index
            onset_source: str | None = ONSET_SOURCE_DETECTOR if failed else None
            if bool(dones[i]):
                self._terminated_step[i] = step_index
                if onset_source is None:
                    onset_source = ONSET_SOURCE_SIM_TERMINATION
            self._loggers[i].append(
                TrajectoryStep(
                    step=step_index,
                    timestamp_s=timestamp,
                    base_pos=pos[i],
                    base_quat=quat[i],
                    base_lin_vel_body=lin[i],
                    base_ang_vel_body=ang[i],
                    joint_pos=jpos[i],
                    joint_vel=jvel[i],
                    command_vel=cmd[i],
                    action=act[i],
                    contact_forces=contact[i],
                    failure_flag=failed,
                    failure_mode=self._failure_mode[i] if failed else None,
                    odom_valid=True,
                    capture_source=CAPTURE_SOURCE_SIM,
                    base_lin_vel_source=BASE_LIN_VEL_SOURCE_SIM,
                    # In sim the policy consumes the env's own base_lin_vel
                    # term, so the logged value and the policy input agree.
                    obs_base_lin_vel_source=BASE_LIN_VEL_SOURCE_SIM,
                    contact_forces_units=self.contact_forces_units,
                    sim_termination_index=step_index if bool(dones[i]) else None,
                    sim_termination_time_s=timestamp if bool(dones[i]) else None,
                    failure_onset_source=onset_source,
                )
            )
            self._rows[i] += 1
            if bool(dones[i]):
                self._closed_env[i] = True

    def close(self) -> list[VariantResult]:
        """Finalize every Parquet and return what each env produced.

        An env that recorded zero rows writes no file: an empty Parquet in the
        curriculum pool would be counted as a variant by
        ``scripts/loop_closure.sh`` and would seed nothing.
        """

        if self._closed:
            return self._results
        self._closed = True
        for log in self._loggers:
            log.close()
        results: list[VariantResult] = []
        for i, path in enumerate(self.paths):
            if self._rows[i] == 0:
                path.unlink(missing_ok=True)
                continue
            results.append(
                VariantResult(
                    env_index=i,
                    path=path,
                    rows=self._rows[i],
                    failed=self._failure_step[i] is not None,
                    failure_mode=self._failure_mode[i],
                    failure_step=self._failure_step[i],
                    terminated_step=self._terminated_step[i],
                )
            )
        self._results = results
        return results

    def write_index(self, path: str | Path, extra: dict | None = None) -> Path:
        """Write the machine-readable index of what the replay produced."""
        results = self.close()
        payload = {
            "num_envs": self.num_envs,
            "control_dt": self.control_dt,
            "variants_written": len(results),
            "variants_with_failure": sum(1 for r in results if r.failed),
            "variants": [r.to_json() for r in results],
        }
        if extra:
            payload.update(extra)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        return out

    def __enter__(self) -> VariantTrajectoryWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
