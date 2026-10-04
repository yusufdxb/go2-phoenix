"""Deploy-side builder for the manifest-declared PhoenixVelocity actor observation.

The checkpoint manifest names the observation schema the policy was trained on
(``observation.schema_version`` + ``schema_sha256``). For ``phoenix-velocity-obs/v2``
that is a 45-D vector, ``base_ang_vel, projected_gravity, velocity_command,
joint_pos_rel, joint_vel, last_action``, with per-term scales and NO
``base_lin_vel``. The pre-v2 deploy path (``observation.ObservationBuilder``,
48-D) cannot produce it.

This module is a thin, refusing adapter over the ONE shared definition,
:func:`phoenix.velocity.observation.build_actor_observation` (the function the
MuJoCo sim-to-sim runner also calls, built on :mod:`phoenix.velocity.contract`,
the table training's env config reads its scales from). It adds what deploy
needs and training does not: the manifest cross-check, so the node refuses to
start on a manifest whose schema this code does not reproduce exactly.

Pure numpy, no torch, no Isaac Lab, no rclpy: CI-reachable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from phoenix.velocity.contract import (
    ACTOR_OBS_DIM,
    ACTOR_OBS_TERMS,
    DEFAULT_JOINT_POS,
    JOINT_ORDER,
    NUM_JOINTS,
    OBS_SCHEMA_V2,
    UnknownObsSchemaError,
    actor_obs_scales,
    manifest_obs_schema,
    obs_slices,
)
from phoenix.velocity.observation import (
    build_actor_observation,
    quat_xyzw_to_wxyz,
)

#: The schema hash this builder reproduces for v2. Pinned here as a literal so a
#: silent edit of the contract tables (terms, scales, joint order, default pose)
#: cannot re-bless a manifest: both this literal and the contract must agree.
V2_SCHEMA_SHA256 = "53a41f30142f6afdec92428314ae5092886d81e5ac9a59936f12c634fad7805a"


class ManifestObservationError(ValueError):
    """The manifest's observation declaration is not one this builder reproduces."""


def manifest_actor_schema(manifest: Mapping[str, object]) -> str | None:
    """The velocity-contract schema a manifest declares, validated; None if it declares none.

    None means an H25-era manifest with no ``phoenix-velocity-obs`` block: the caller
    keeps the pre-v2 48-D path. Anything that declares a schema is checked in full
    and refused on any disagreement, never skipped.
    """
    obs = manifest.get("observation") if isinstance(manifest, Mapping) else None
    if not isinstance(obs, Mapping) or not str(obs.get("schema_version", "")).startswith(
        "phoenix-velocity-obs/"
    ):
        return None
    try:
        version = manifest_obs_schema(manifest)  # known version + sha == this code's contract
    except UnknownObsSchemaError as exc:
        raise ManifestObservationError(str(exc)) from exc
    if version == OBS_SCHEMA_V2 and obs.get("schema_sha256") != V2_SCHEMA_SHA256:
        raise ManifestObservationError(
            f"manifest schema_sha256 {obs.get('schema_sha256')!r} is not the v2 hash this "
            f"builder reproduces ({V2_SCHEMA_SHA256})"
        )
    terms = [name for name, *_ in ACTOR_OBS_TERMS]
    if list(obs.get("actor_terms") or []) != terms:
        raise ManifestObservationError(
            f"manifest actor_terms {obs.get('actor_terms')!r} != builder terms {terms}"
        )
    if obs.get("actor_dim") != ACTOR_OBS_DIM:
        raise ManifestObservationError(
            f"manifest actor_dim {obs.get('actor_dim')!r} != builder dim {ACTOR_OBS_DIM}"
        )
    if "actor_scales" in obs and dict(obs["actor_scales"]) != actor_obs_scales(version):
        raise ManifestObservationError(
            f"manifest actor_scales {obs['actor_scales']!r} != builder scales "
            f"{actor_obs_scales(version)!r}"
        )
    return version


class ManifestObservationBuilder:
    """The actor observation for the schema a checkpoint manifest declares."""

    def __init__(
        self,
        schema_version: str,
        joint_order: Sequence[str],
        default_joint_pos: Mapping[str, float],
    ) -> None:
        actor_obs_scales(schema_version)  # raises on an unknown version
        if tuple(joint_order) != JOINT_ORDER:
            raise ManifestObservationError(
                f"deploy joint_order {list(joint_order)} != contract {list(JOINT_ORDER)}"
            )
        bad = {
            n: (default_joint_pos.get(n), DEFAULT_JOINT_POS[n])
            for n in JOINT_ORDER
            if default_joint_pos.get(n) is None
            or abs(float(default_joint_pos[n]) - DEFAULT_JOINT_POS[n]) > 1e-9
        }
        if bad:
            raise ManifestObservationError(f"deploy default_joint_pos != contract: {bad}")
        self.schema_version = schema_version

    @property
    def dim(self) -> int:
        return ACTOR_OBS_DIM

    def term_slices(self) -> dict[str, slice]:
        return obs_slices()

    @classmethod
    def from_manifest_file(
        cls,
        manifest_path: Path | str,
        joint_order: Sequence[str],
        default_joint_pos: Mapping[str, float],
    ) -> ManifestObservationBuilder:
        data = json.loads(Path(manifest_path).read_text())
        version = manifest_actor_schema(data)
        if version is None:
            raise ManifestObservationError(
                f"manifest {manifest_path} declares no phoenix-velocity-obs schema"
            )
        return cls(version, joint_order, default_joint_pos)

    def build(
        self,
        *,
        base_ang_vel: np.ndarray,
        quat_xyzw: Sequence[float],
        velocity_command: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        """Raw robot state to the float32 actor observation (policy joint order, ROS xyzw quat)."""
        return build_actor_observation(
            gyro_body=base_ang_vel,
            quat_wxyz=quat_xyzw_to_wxyz(quat_xyzw),
            command=velocity_command,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_action=last_action,
            obs_schema_version=self.schema_version,
        )


def canonical_stand_observation(builder: ManifestObservationBuilder) -> np.ndarray:
    """Body at rest and upright, zero command, default pose, no previous action."""
    return builder.build(
        base_ang_vel=np.zeros(3),
        quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        velocity_command=np.zeros(3),
        joint_pos=np.asarray([DEFAULT_JOINT_POS[j] for j in JOINT_ORDER]),
        joint_vel=np.zeros(NUM_JOINTS),
        last_action=np.zeros(NUM_JOINTS),
    )


def _axis_angle_quat_xyzw(axis: Sequence[float], angle: float) -> tuple[float, ...]:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return (*(a * np.sin(angle / 2.0)), float(np.cos(angle / 2.0)))


def parity_failures(builder: ManifestObservationBuilder) -> list[str]:
    """Drive the builder with a distinct value per term and read every term back.

    The expectation is computed here from the contract's term table and an
    independent Rodrigues rotation, not by calling the shared builder again, so a
    reordered, rescaled, zeroed or aliased term fails.
    """
    scales = actor_obs_scales(builder.schema_version)
    rng = np.random.default_rng(7)
    gyro = np.asarray([0.31, -0.62, 0.93])
    cmd = np.asarray([0.4, -0.2, 0.7])
    q = np.asarray([DEFAULT_JOINT_POS[j] for j in JOINT_ORDER]) + rng.uniform(-0.3, 0.3, NUM_JOINTS)
    qd = rng.uniform(-4.0, 4.0, NUM_JOINTS)
    last = rng.uniform(-1.0, 1.0, NUM_JOINTS)
    axis, angle = (1.0, 2.0, -0.5), 0.6
    quat = _axis_angle_quat_xyzw(axis, angle)
    k = np.asarray(axis) / np.linalg.norm(axis)
    kx = np.asarray([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)
    expected = {
        "base_ang_vel": gyro,
        "projected_gravity": rot.T @ np.asarray([0.0, 0.0, -1.0]),
        "velocity_command": cmd,
        "joint_pos_rel": q - np.asarray([DEFAULT_JOINT_POS[j] for j in JOINT_ORDER]),
        "joint_vel": qd,
        "last_action": last,
    }
    obs = builder.build(
        base_ang_vel=gyro,
        quat_xyzw=quat,
        velocity_command=cmd,
        joint_pos=q,
        joint_vel=qd,
        last_action=last,
    )
    failures: list[str] = []
    if obs.shape != (ACTOR_OBS_DIM,) or obs.dtype != np.float32:
        return [f"observation is {obs.dtype}{obs.shape}, want float32 ({ACTOR_OBS_DIM},)"]
    for name, sl in builder.term_slices().items():
        want = expected[name] * scales[name]
        err = float(np.max(np.abs(obs[sl] - want)))
        if err > 1e-5:
            failures.append(f"{name}: max abs error {err:.3g} (slice {sl.start}:{sl.stop})")
        if not np.any(np.abs(obs[sl]) > 1e-6):
            failures.append(f"{name}: term is all zeros")
    return failures


def reference_observations(
    builder: ManifestObservationBuilder, n: int = 2000, seed: int = 0
) -> np.ndarray:
    """Deterministic SYNTHETIC observations spread around the stand pose, (n, 45) float32.

    Not a rollout: a torch-vs-ONNX parity input needs broad, reproducible coverage of
    the input space, which a fixed seed gives. Commands cover the manifest's trained
    ranges; tilts up to ~0.5 rad; joint offsets +-0.3 rad; joint velocities and
    previous actions at the scale a trotting policy produces.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        axis = rng.normal(size=3)
        quat = _axis_angle_quat_xyzw(axis, float(rng.uniform(0.0, 0.5)))
        rows.append(
            builder.build(
                base_ang_vel=rng.normal(0.0, 0.8, 3),
                quat_xyzw=quat,
                velocity_command=np.asarray(
                    [rng.uniform(-0.8, 1.0), rng.uniform(-0.5, 0.5), rng.uniform(-1.0, 1.0)]
                ),
                joint_pos=np.asarray([DEFAULT_JOINT_POS[j] for j in JOINT_ORDER])
                + rng.uniform(-0.3, 0.3, NUM_JOINTS),
                joint_vel=rng.normal(0.0, 3.0, NUM_JOINTS),
                last_action=np.clip(rng.normal(0.0, 0.6, NUM_JOINTS), -3.0, 3.0),
            )
        )
    return np.stack(rows).astype(np.float32)
