"""Sensor-to-policy observation parity gate.

WHAT THIS CATCHES THAT THE EXISTING PARITY GATE CANNOT.

``phoenix.sim2real.verify_deploy`` compares Torch against ONNX on the SAME
observation vector. That gate is blind by construction to everything upstream
of the vector: if the deploy node hands both backends a term full of zeros,
both agree perfectly and the gate passes. That is exactly what happened.
``ros2_policy_node`` fed the policy ``base_lin_vel = np.zeros(3)`` for the
whole life of the deploy path while training fed it a real, noised body
velocity, and every parity run was green.

So this gate checks the other half: that the vector assembled FROM SENSOR
INPUTS matches the training contract term by term, in order, at scale 1.0,
and that no term has been quietly replaced by a constant.

Method: drive :func:`phoenix.sim2real.observation.assemble_policy_observation`
with a probe in which every term carries a distinct, non-degenerate
signature, then read each term back out of the assembled vector by its
declared slice and compare. A term that is dropped, reordered, rescaled,
zeroed, or aliased onto another term fails.

Pure numpy. No Isaac Lab, no ROS, no onnxruntime, no torch, so it runs in the
standard CI suite. Run standalone with::

    python -m phoenix.sim2real.obs_parity
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .observation import (
    BASE_LIN_VEL_SOURCE_ODOM,
    BASE_LIN_VEL_SOURCE_ZEROS,
    OBS_TERM_ORDER,
    OBS_TERM_SCALES,
    JointOrder,
    ObservationBuilder,
    assemble_policy_observation,
    projected_gravity_from_quat,
    resolve_base_lin_vel,
    term_slices,
)

logger = logging.getLogger("phoenix.sim2real.obs_parity")

#: Repo root, resolved from this file rather than from the working directory,
#: so the gate finds its inputs no matter where it is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The training-side declaration this gate cross-checks the deploy layout
#: against. ``observation.include`` in that file is the repo's own statement
#: of which terms the policy is trained on, in order.
TRAINING_ENV_CONFIG = REPO_ROOT / "configs" / "env" / "base.yaml"

#: The deploy config the standalone CLI checks. The pytest gate checks every
#: config in that directory.
DEFAULT_DEPLOY_CONFIG = REPO_ROOT / "configs" / "sim2real" / "deploy.yaml"


@dataclass(frozen=True)
class TermCheck:
    name: str
    dims: int
    expected: np.ndarray
    actual: np.ndarray
    max_abs_diff: float
    passed: bool
    note: str = ""


@dataclass(frozen=True)
class ObsParityReport:
    terms: tuple[TermCheck, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        lines = [
            f"{'TERM':<20} {'DIMS':>5} {'MAXDIFF':>12}  RESULT",
        ]
        for t in self.terms:
            lines.append(
                f"{t.name:<20} {t.dims:>5} {t.max_abs_diff:>12.3e}  "
                f"{'PASS' if t.passed else 'FAIL'}{(' ' + t.note) if t.note else ''}"
            )
        for f in self.failures:
            lines.append(f"FAIL: {f}")
        lines.append("OBSERVATION PARITY: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def load_training_term_order(config_path: Path = TRAINING_ENV_CONFIG) -> tuple[str, ...] | None:
    """Return ``observation.include`` from the training env config.

    Returns ``None`` when the file is not present (for example when the
    package is installed outside a checkout), so the caller can skip the
    cross-check instead of failing for the wrong reason.
    """
    if not config_path.is_file():
        return None
    import yaml

    data = yaml.safe_load(config_path.read_text()) or {}
    include = ((data.get("observation") or {}).get("include")) or None
    return None if include is None else tuple(str(x) for x in include)


def _probe_inputs(n_joints: int) -> dict[str, np.ndarray]:
    """Distinct, non-degenerate values for every sensor input.

    Every entry is nonzero and no two terms share a value, so a zeroed term,
    a duplicated term, and a swapped pair are all distinguishable. Values are
    physically plausible but not realistic; realism is irrelevant here and
    round numbers make a failure easy to read.
    """
    return {
        "base_lin_vel": np.asarray([0.31, -0.62, 0.13], dtype=np.float32),
        "base_ang_vel": np.asarray([1.11, -2.22, 3.33], dtype=np.float32),
        "velocity_command": np.asarray([0.77, -0.44, 1.55], dtype=np.float32),
        "joint_pos": np.arange(1, n_joints + 1, dtype=np.float32) * 0.017,
        "joint_vel": np.arange(1, n_joints + 1, dtype=np.float32) * -0.29,
        "last_action": np.arange(1, n_joints + 1, dtype=np.float32) * 0.041,
    }


def check_observation_parity(
    builder: ObservationBuilder,
    *,
    config_path: Path = TRAINING_ENV_CONFIG,
) -> ObsParityReport:
    """Check the deploy assembly against the training observation contract."""
    n_joints = len(builder.joint_order)
    probe = _probe_inputs(n_joints)
    # A quaternion with roll, pitch and yaw all nonzero, so projected gravity
    # is not accidentally equal to the canonical (0, 0, -1).
    quat = _normalized_quat(0.11, 0.23, 0.37, 0.89)

    failures: list[str] = []

    training_order = load_training_term_order(config_path)
    if training_order is None:
        failures.append(
            f"training env config {config_path} not found; cannot cross-check term order"
        )
    elif training_order != OBS_TERM_ORDER:
        failures.append(
            f"training observation.include {list(training_order)} does not match the deploy "
            f"term order {list(OBS_TERM_ORDER)}"
        )

    sample = resolve_base_lin_vel(
        BASE_LIN_VEL_SOURCE_ODOM,
        odom_lin_vel_body=probe["base_lin_vel"],
        odom_valid=True,
        odom_provenance="parity_probe",
    )
    obs = assemble_policy_observation(
        builder,
        base_lin_vel=sample.value,
        quat_xyzw=quat,
        base_ang_vel=probe["base_ang_vel"],
        velocity_command=probe["velocity_command"],
        joint_pos=probe["joint_pos"],
        joint_vel=probe["joint_vel"],
        last_action=probe["last_action"],
        pad_zeros=0,
    )

    if obs.shape != (builder.dim,):
        failures.append(f"assembled observation has shape {obs.shape}, expected ({builder.dim},)")
        return ObsParityReport(terms=(), failures=tuple(failures))
    if obs.dtype != np.float32:
        failures.append(f"assembled observation dtype is {obs.dtype}, expected float32")

    expected_by_term = {
        "base_lin_vel": probe["base_lin_vel"],
        "base_ang_vel": probe["base_ang_vel"],
        "projected_gravity": projected_gravity_from_quat(*quat),
        "velocity_command": probe["velocity_command"],
        # The one term with a documented transform: joint_pos enters the
        # policy relative to the training default pose (Isaac Lab
        # joint_pos_rel). Everything else is raw.
        "joint_pos": probe["joint_pos"] - builder.default_q,
        "joint_vel": probe["joint_vel"],
        "last_action": probe["last_action"],
    }

    slices = term_slices(n_joints)
    checks: list[TermCheck] = []
    for name in OBS_TERM_ORDER:
        expected = np.asarray(expected_by_term[name], dtype=np.float32)
        actual = obs[slices[name]]
        diff = float(np.max(np.abs(actual - expected))) if actual.size else float("inf")
        note = ""
        passed = diff <= 1e-6
        if not passed:
            failures.append(
                f"term {name!r} at dims {slices[name].start}..{slices[name].stop - 1} does not "
                f"match its sensor input (max abs diff {diff:.3e})"
            )
        # A trained term silently replaced by a constant is the specific
        # regression this gate exists for, so name it explicitly rather than
        # letting it hide inside a generic mismatch.
        if np.allclose(actual, 0.0) and not np.allclose(expected, 0.0):
            note = "(term is all zeros while its sensor input is not)"
            failures.append(f"term {name!r} is a constant zero in the assembled observation")
        if OBS_TERM_SCALES[name] != 1.0:
            failures.append(
                f"term {name!r} declares scale {OBS_TERM_SCALES[name]}; training applies none"
            )
        checks.append(
            TermCheck(
                name=name,
                dims=int(expected.size),
                expected=expected,
                actual=np.asarray(actual),
                max_abs_diff=diff,
                passed=passed,
                note=note,
            )
        )

    covered = sum(c.dims for c in checks)
    if covered != builder.dim:
        failures.append(f"terms cover {covered} dims but the vector is {builder.dim}")

    return ObsParityReport(terms=tuple(checks), failures=tuple(failures))


def check_zeros_fallback_is_explicit() -> tuple[str, ...]:
    """Confirm the zeros fallback can only be reached deliberately.

    The gate above proves the assembler is honest when it is given a
    measurement. This proves the other half: that the historical behaviour,
    a zeroed ``base_lin_vel``, is reachable only by naming it, and that an
    unavailable measurement raises instead of degrading into the same zeros.
    """
    from .observation import BaseLinVelUnavailableError

    failures: list[str] = []
    zeros = resolve_base_lin_vel(BASE_LIN_VEL_SOURCE_ZEROS)
    if zeros.measured or not np.allclose(zeros.value, 0.0):
        failures.append("the zeros source must return unmeasured zeros")
    if "operator_selected" not in zeros.provenance:
        failures.append("the zeros source must record that an operator selected it")
    try:
        resolve_base_lin_vel(BASE_LIN_VEL_SOURCE_ODOM, odom_lin_vel_body=None, odom_valid=False)
    except BaseLinVelUnavailableError:
        pass
    else:
        failures.append("an unavailable odom source must raise, not return zeros")
    return tuple(failures)


def _normalized_quat(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _default_builder() -> ObservationBuilder:
    """Build the gate's reference builder from the shipped deploy config."""
    import yaml

    cfg = yaml.safe_load(DEFAULT_DEPLOY_CONFIG.read_text())
    return ObservationBuilder(
        JointOrder(tuple(cfg["joint_order"])), cfg["control"]["default_joint_pos"]
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI plumbing
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    report = check_observation_parity(_default_builder())
    print(report.summary())
    extra = check_zeros_fallback_is_explicit()
    for f in extra:
        print(f"FAIL: {f}")
    return 0 if (report.passed and not extra) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
