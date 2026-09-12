"""Build the policy observation vector from ROS 2 messages.

Layout must match what the policy was trained with. On flat terrain the
observation is 48-dim proprioception:

.. code-block:: text

    [ base_lin_vel (3) ,
      base_ang_vel (3) ,
      projected_gravity (3) ,
      velocity_command (3) ,
      joint_pos_rel_default (12) ,
      joint_vel (12) ,
      last_action (12) ]  -> 48 dims

The term order is fixed by Isaac Lab's
``manager_based/locomotion/velocity/velocity_env_cfg.py``,
``ObservationsCfg.PolicyCfg``, and mirrored by ``observation.include`` in
``configs/env/base.yaml``. No term carries a scale factor: every term enters
the vector in its raw SI unit. :data:`OBS_TERM_ORDER` is the machine-readable
copy of that contract, and :mod:`phoenix.sim2real.obs_parity` gates the
deploy-side assembly against it.

On rough terrain Isaac Lab's task also appends a 187-dim height scanner
reading (total 235 dims); a deployed rough-terrain policy therefore
needs an equivalent scanner feed on the real robot. The builder below
returns only the proprioceptive 48-dim prefix, and
:func:`assemble_policy_observation` appends the height-scan padding that
deploy scripts ask for.

``base_lin_vel`` is the one term the GO2 cannot measure directly. It is a
REAL, noised term at training time, so feeding it a silent zero is a
distribution shift the policy never saw. This module therefore refuses to
invent it: :func:`resolve_base_lin_vel` takes an explicit, operator-selected
source and raises :class:`BaseLinVelUnavailableError` rather than
substituting a plausible zero.

This module is pure numpy (no rclpy), so it can be unit-tested in CI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

#: Canonical training-time observation term order. This is the contract the
#: deploy path must reproduce term-by-term and in order. Sourced from Isaac
#: Lab ``velocity_env_cfg.ObservationsCfg.PolicyCfg`` and cross-checked in CI
#: against ``configs/env/base.yaml`` ``observation.include`` by
#: :mod:`phoenix.sim2real.obs_parity`.
OBS_TERM_ORDER: tuple[str, ...] = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_command",
    "joint_pos",
    "joint_vel",
    "last_action",
)

#: Per-term dimensionality. ``None`` means "one per joint".
OBS_TERM_DIMS: dict[str, int | None] = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_command": 3,
    "joint_pos": None,
    "joint_vel": None,
    "last_action": None,
}

#: Per-term scale factor applied between the sensor value and the policy
#: input. Training applies none (see ``docs/native_runtime_audit.md`` section
#: 2), so every entry is 1.0 and the parity gate asserts it stays that way.
OBS_TERM_SCALES: dict[str, float] = dict.fromkeys(OBS_TERM_ORDER, 1.0)

#: ``base_lin_vel`` is taken from validated odometry, transformed into the
#: body frame. This is the only source that feeds the policy a measurement.
BASE_LIN_VEL_SOURCE_ODOM = "odom"
#: ``base_lin_vel`` is deliberately zeroed. This is a DISTRIBUTION SHIFT, not
#: a measurement: training noised a real velocity into these three dims. It
#: exists because every Phoenix checkpoint and every reliability-shield
#: artifact to date was calibrated with this substitution in place, so
#: removing the option silently would invalidate them. It must be selected by
#: hand in the deploy config, is logged loudly at startup, and is recorded in
#: every parquet row it produced.
BASE_LIN_VEL_SOURCE_ZEROS = "zeros"
#: Captures written from a simulator, where the value is ground truth.
BASE_LIN_VEL_SOURCE_SIM = "sim_ground_truth"

BASE_LIN_VEL_SOURCES: tuple[str, ...] = (
    BASE_LIN_VEL_SOURCE_ODOM,
    BASE_LIN_VEL_SOURCE_ZEROS,
)


class BaseLinVelUnavailableError(RuntimeError):
    """The selected ``base_lin_vel`` source produced no valid sample.

    Raised instead of returning zeros, because a zero here is numerically
    indistinguishable from a stationary robot and the policy was trained on a
    real measurement.
    """


@dataclass(frozen=True)
class BaseLinVelSample:
    """A resolved ``base_lin_vel`` plus the provenance of that resolution."""

    value: np.ndarray  # (3,) float32, body frame, m/s
    #: Free-form provenance string recorded per-row in the trajectory parquet,
    #: e.g. ``odom:body_passthrough`` or ``zeros:operator_selected``.
    provenance: str
    #: True only when ``value`` came from a sensor. False means the operator
    #: selected a documented substitution.
    measured: bool


@dataclass(frozen=True)
class JointOrder:
    """The canonical joint order used at training time.

    ROS 2 ``/joint_states`` messages may emit joints in a different
    order. :meth:`remap` builds an index vector that reorders a
    joint-state array into the policy's canonical layout.
    """

    names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.names)

    def remap(self, ros_joint_names: list[str]) -> np.ndarray:
        """Return indices into ``ros_joint_names`` such that
        ``arr[indices]`` yields values in canonical order."""
        lookup = {name: i for i, name in enumerate(ros_joint_names)}
        missing = [n for n in self.names if n not in lookup]
        if missing:
            raise KeyError(f"ROS /joint_states is missing joints: {missing}")
        return np.asarray([lookup[n] for n in self.names], dtype=np.int64)


class ObservationBuilder:
    """Stateless builder for the policy observation vector."""

    def __init__(self, joint_order: JointOrder, default_joint_pos: Mapping[str, float]) -> None:
        self.joint_order = joint_order
        self.default_q = np.asarray(
            [default_joint_pos[n] for n in joint_order.names], dtype=np.float32
        )
        self._zero_action = np.zeros(len(joint_order), dtype=np.float32)

    @property
    def dim(self) -> int:
        return 3 + 3 + 3 + 3 + 3 * len(self.joint_order)

    def term_slices(self) -> dict[str, slice]:
        """Return ``{term name: slice}`` for the proprioceptive vector.

        The single machine-readable answer to "which dims are which term",
        used by the parity gate and by anything that needs to inspect one
        term of a captured observation.
        """
        return term_slices(len(self.joint_order))

    def build(
        self,
        *,
        base_lin_vel: np.ndarray,  # (3,) m/s in body frame
        base_ang_vel: np.ndarray,  # (3,) rad/s in body frame
        projected_gravity: np.ndarray,  # (3,) body-frame gravity unit vector
        velocity_command: np.ndarray,  # (3,) [vx, vy, wz]
        joint_pos: np.ndarray,  # (N,) canonical order, absolute rad
        joint_vel: np.ndarray,  # (N,) canonical order, rad/s
        last_action: np.ndarray | None = None,  # (N,) policy output from prev step
    ) -> np.ndarray:
        if last_action is None:
            last_action = self._zero_action
        parts = [
            base_lin_vel.astype(np.float32, copy=False),
            base_ang_vel.astype(np.float32, copy=False),
            projected_gravity.astype(np.float32, copy=False),
            velocity_command.astype(np.float32, copy=False),
            (joint_pos - self.default_q).astype(np.float32, copy=False),
            joint_vel.astype(np.float32, copy=False),
            last_action.astype(np.float32, copy=False),
        ]
        obs = np.concatenate(parts, axis=-1)
        if obs.shape[-1] != self.dim:
            raise ValueError(f"Observation dim mismatch: got {obs.shape[-1]}, expected {self.dim}")
        return obs


def term_slices(n_joints: int) -> dict[str, slice]:
    """Return ``{term name: slice}`` for a proprioceptive vector with ``n_joints``."""
    out: dict[str, slice] = {}
    start = 0
    for name in OBS_TERM_ORDER:
        dims = OBS_TERM_DIMS[name]
        width = n_joints if dims is None else dims
        out[name] = slice(start, start + width)
        start += width
    return out


def projected_gravity_from_quat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """World-frame gravity ``(0, 0, -1)`` rotated into the body frame.

    Matches Isaac Lab's ``mdp.projected_gravity`` observation term, i.e.
    ``g_body = R(q).T @ g_world``. Canonical implementation: the deploy node
    and the ONNX parity gate both route here so the two cannot drift apart
    again (they did once, with gx/gy sign-flipped, which fed the policy a
    mirror-image gravity vector).
    """
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return np.asarray([gx, gy, gz], dtype=np.float32)


def resolve_base_lin_vel(
    source: str,
    *,
    odom_lin_vel_body: np.ndarray | None = None,
    odom_valid: bool = False,
    odom_provenance: str = "odom",
) -> BaseLinVelSample:
    """Resolve the ``base_lin_vel`` observation term, or fail loudly.

    ``source`` must be one of :data:`BASE_LIN_VEL_SOURCES` and is an explicit
    operator choice in ``configs/sim2real/*.yaml``; there is deliberately no
    default, because the two options put the policy in measurably different
    input distributions.

    * ``odom``: return the validated body-frame odometry twist. If odometry
      is absent, stale, or arrived in a frame we could not identify, raise
      :class:`BaseLinVelUnavailableError`. The caller is expected to fail closed.
    * ``zeros``: return zeros, flagged ``measured=False``. This is the
      historical Phoenix deploy behaviour and the distribution every exported
      checkpoint and shield artifact was calibrated in.

    Never returns a silent zero for the ``odom`` source. That substitution is
    the defect this function exists to make impossible.
    """
    if source == BASE_LIN_VEL_SOURCE_ZEROS:
        return BaseLinVelSample(
            value=np.zeros(3, dtype=np.float32),
            provenance="zeros:operator_selected",
            measured=False,
        )
    if source == BASE_LIN_VEL_SOURCE_ODOM:
        if not odom_valid or odom_lin_vel_body is None:
            raise BaseLinVelUnavailableError(
                "base_lin_vel_source='odom' but no valid body-frame odometry twist this step "
                f"(provenance={odom_provenance!r}). Refusing to substitute zeros for a trained "
                "observation term."
            )
        value = np.asarray(odom_lin_vel_body, dtype=np.float32).reshape(-1)
        if value.shape != (3,):
            raise BaseLinVelUnavailableError(
                f"odometry twist has shape {value.shape}, expected (3,)"
            )
        if not np.isfinite(value).all():
            raise BaseLinVelUnavailableError(f"odometry twist is not finite: {value.tolist()}")
        return BaseLinVelSample(value=value, provenance=f"odom:{odom_provenance}", measured=True)
    raise ValueError(
        f"unknown base_lin_vel source {source!r}; expected one of {list(BASE_LIN_VEL_SOURCES)}"
    )


def assemble_policy_observation(
    builder: ObservationBuilder,
    *,
    base_lin_vel: np.ndarray,
    quat_xyzw: tuple[float, float, float, float],
    base_ang_vel: np.ndarray,
    velocity_command: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray | None = None,
    pad_zeros: int = 0,
) -> np.ndarray:
    """Assemble the full policy input from raw sensor quantities.

    This is the single sensor-to-policy path: the deploy node calls it, and
    :mod:`phoenix.sim2real.obs_parity` gates it against the training term
    contract. Anything that bypasses it (an inline ``np.concatenate`` in a
    node, say) is invisible to that gate, which is exactly how
    ``base_lin_vel`` came to be fed a hardcoded zero.

    ``pad_zeros`` appends the rough-terrain height-scan padding. Those dims
    ARE a knowingly-zeroed trained term for a rough-terrain checkpoint; every
    deployed Phoenix config sets ``obs_pad_zeros: 0`` and runs a flat-task
    policy that has no scanner term at all.
    """
    proprio = builder.build(
        base_lin_vel=np.asarray(base_lin_vel, dtype=np.float32),
        base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32),
        projected_gravity=projected_gravity_from_quat(*quat_xyzw),
        velocity_command=np.asarray(velocity_command, dtype=np.float32),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
        joint_vel=np.asarray(joint_vel, dtype=np.float32),
        last_action=None if last_action is None else np.asarray(last_action, dtype=np.float32),
    )
    if pad_zeros > 0:
        return np.concatenate([proprio, np.zeros(pad_zeros, dtype=np.float32)], axis=-1)
    return proprio
