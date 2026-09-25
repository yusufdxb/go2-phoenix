"""The PhoenixVelocity contract: 45-D actor observation, action space, checkpoint manifest.

This module is the ONE definition that training (Isaac Lab), evaluation, the
MuJoCo sim-to-sim path and the hardware deploy path all import. It is pure
Python (stdlib plus the numpy-only go2_model constants), so it runs in CI, on the
workstation and on the Jetson.

Why it exists
-------------
H25 was trained with every velocity command at zero and a 48-D observation whose
first three dims are ``base_lin_vel``, a signal the GO2 cannot measure. It was
then deployed with those three dims zeroed. Nothing machine-checkable recorded
either fact next to the checkpoint, so a stand-only checkpoint could be pointed
at a walking config and nothing but prose said no. The contract closes that:

* the actor observation is exactly :data:`ACTOR_OBS_TERMS` (45 dims, no base
  linear velocity, no exteroception);
* every trained checkpoint carries a manifest (:data:`MANIFEST_NAME`) that
  records what it was trained on, including the command ranges;
* whether a checkpoint may run in velocity mode is DERIVED from the recorded
  command ranges (:func:`derive_locomotion_capable`), never read from a flag
  someone typed, and a stored flag that disagrees with the derivation is refused;
* :func:`validate_manifest_for_mode` fails closed on any mismatch.

Term conventions (all SI, no scaling, matching Isaac Lab ``mdp`` functions)
--------------------------------------------------------------------------
``base_ang_vel``       body-frame angular velocity, rad/s (IMU gyro on the GO2)
``projected_gravity``  unit gravity vector in the body frame; upright = (0, 0, -1)
``velocity_command``   (vx m/s, vy m/s, wz rad/s), body-frame planar twist command
``joint_pos_rel``      q - q_default, rad, :data:`JOINT_ORDER`
``joint_vel``          qd, rad/s, :data:`JOINT_ORDER` (Isaac ``joint_vel_rel``;
                       the default joint velocity is zero)
``last_action``        the previous RAW policy action, dimensionless, pre-scale
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER, TRAINING_DEFAULT_JOINT_POS

OBS_SCHEMA_VERSION = "phoenix-velocity-obs/v1"
MANIFEST_SCHEMA = "phoenix-checkpoint-manifest/v1"
#: File written beside every PhoenixVelocity checkpoint directory.
MANIFEST_NAME = "phoenix_manifest.json"

JOINT_ORDER: tuple[str, ...] = POLICY_JOINT_ORDER
DEFAULT_JOINT_POS: dict[str, float] = dict(TRAINING_DEFAULT_JOINT_POS)
NUM_JOINTS = len(JOINT_ORDER)

#: Actor observation, in order. ``(name, dim, unit, frame)``.
ACTOR_OBS_TERMS: tuple[tuple[str, int, str, str], ...] = (
    ("base_ang_vel", 3, "rad/s", "body"),
    ("projected_gravity", 3, "unit vector", "body"),
    ("velocity_command", 3, "m/s,m/s,rad/s", "body"),
    ("joint_pos_rel", NUM_JOINTS, "rad", "joint, relative to DEFAULT_JOINT_POS"),
    ("joint_vel", NUM_JOINTS, "rad/s", "joint"),
    ("last_action", NUM_JOINTS, "dimensionless", "raw policy action, pre-scale"),
)
ACTOR_OBS_DIM = sum(t[1] for t in ACTOR_OBS_TERMS)
assert ACTOR_OBS_DIM == 45, ACTOR_OBS_DIM

#: Signals the actor must never depend on: not reliably measurable on the GO2, or
#: exteroceptive / external localization. The critic may see them.
FORBIDDEN_ACTOR_TERMS: frozenset[str] = frozenset(
    {
        "base_lin_vel",
        "height_scan",
        "base_height",
        "base_pos",
        "lidar",
        "odometry",
        "foot_contact",
        "feet_contact",
    }
)

ACTION_DIM = NUM_JOINTS
#: Radians of joint target per unit of policy action (Isaac GO2 velocity default).
ACTION_SCALE = 0.25
CONTROL_HZ = 50
PHYSICS_HZ = 200
DECIMATION = PHYSICS_HZ // CONTROL_HZ

#: A checkpoint whose trained command envelope does not reach these magnitudes
#: was not trained to move. The thresholds are deliberately low: they separate
#: "trained with zero commands" (H25) from "trained with any real command", and
#: the separate deploy-envelope check stops a small-envelope policy from being
#: asked for more than it saw.
MIN_TRAINED_LIN_VEL_X = 0.2  # m/s
MIN_TRAINED_ANG_VEL_Z = 0.2  # rad/s
#: A policy that saw nonzero commands on less than this fraction of envs never
#: really learned to track them.
MIN_MOVING_ENV_FRACTION = 0.5

MODE_STAND = "stand"
MODE_VELOCITY = "velocity"
MODES = (MODE_STAND, MODE_VELOCITY)


def obs_slices() -> dict[str, slice]:
    """Name -> slice into the 45-D actor observation."""
    out: dict[str, slice] = {}
    start = 0
    for name, dim, _unit, _frame in ACTOR_OBS_TERMS:
        out[name] = slice(start, start + dim)
        start += dim
    return out


def obs_schema_dict() -> dict[str, Any]:
    """Everything the observation's MEANING depends on, as canonical data."""
    return {
        "version": OBS_SCHEMA_VERSION,
        "terms": [{"name": n, "dim": d, "unit": u, "frame": f} for n, d, u, f in ACTOR_OBS_TERMS],
        "dim": ACTOR_OBS_DIM,
        "joint_order": list(JOINT_ORDER),
        "default_joint_pos": [DEFAULT_JOINT_POS[j] for j in JOINT_ORDER],
        "scales": "none (all terms raw SI)",
    }


def _canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def obs_schema_sha256() -> str:
    return _canonical_sha256(obs_schema_dict())


@dataclass(frozen=True)
class CommandRanges:
    """The velocity-command distribution a checkpoint was trained on.

    ``lin_vel_x`` etc. are the widest ranges ever sampled during training (the
    final curriculum stage when a curriculum is used). ``rel_standing_envs`` is
    the fraction of envs forced to a zero command.
    """

    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    rel_standing_envs: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CommandRanges:
        def pair(key: str) -> tuple[float, float]:
            value = data[key]
            if not (isinstance(value, Sequence) and len(value) == 2):
                raise ValueError(f"command range {key} must be [lo, hi], got {value!r}")
            lo, hi = float(value[0]), float(value[1])
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
                raise ValueError(f"command range {key} is invalid: {value!r}")
            return lo, hi

        rel = float(data["rel_standing_envs"])
        if not 0.0 <= rel <= 1.0:
            raise ValueError(f"rel_standing_envs must be in [0, 1], got {rel}")
        return cls(pair("lin_vel_x"), pair("lin_vel_y"), pair("ang_vel_z"), rel)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lin_vel_x": list(self.lin_vel_x),
            "lin_vel_y": list(self.lin_vel_y),
            "ang_vel_z": list(self.ang_vel_z),
            "rel_standing_envs": self.rel_standing_envs,
        }


def _max_abs(r: tuple[float, float]) -> float:
    return max(abs(r[0]), abs(r[1]))


def derive_locomotion_capable(commands: CommandRanges) -> tuple[bool, list[str]]:
    """Whether training commands could have taught locomotion. ``(capable, reasons)``.

    ``reasons`` lists why NOT when ``capable`` is False.
    """
    reasons: list[str] = []
    if _max_abs(commands.lin_vel_x) < MIN_TRAINED_LIN_VEL_X:
        reasons.append(
            f"trained |vx| max {_max_abs(commands.lin_vel_x):.3f} m/s < {MIN_TRAINED_LIN_VEL_X}"
        )
    if _max_abs(commands.ang_vel_z) < MIN_TRAINED_ANG_VEL_Z:
        reasons.append(
            f"trained |wz| max {_max_abs(commands.ang_vel_z):.3f} rad/s < {MIN_TRAINED_ANG_VEL_Z}"
        )
    moving = 1.0 - commands.rel_standing_envs
    if moving < MIN_MOVING_ENV_FRACTION:
        reasons.append(
            f"only {moving:.2f} of envs received nonzero commands (< {MIN_MOVING_ENV_FRACTION})"
        )
    return (not reasons), reasons


def build_manifest(
    *,
    checkpoint_sha256: str | None,
    commands: CommandRanges,
    git_sha: str,
    git_dirty: bool,
    seed: int,
    task: str,
    simulator: str,
    reward_scales: Mapping[str, float],
    domain_randomization: Mapping[str, Any],
    curriculum: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest for a PhoenixVelocity checkpoint (45-D contract)."""
    capable, reasons = derive_locomotion_capable(commands)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "classification": "velocity_candidate" if capable else "stand_only",
        "locomotion_capable": capable,
        "locomotion_incapable_reasons": reasons,
        "observation": {
            "schema_version": OBS_SCHEMA_VERSION,
            "schema_sha256": obs_schema_sha256(),
            "actor_dim": ACTOR_OBS_DIM,
            "actor_terms": [t[0] for t in ACTOR_OBS_TERMS],
        },
        "action": {"dim": ACTION_DIM, "scale": ACTION_SCALE, "type": "joint_position_offset"},
        "control_hz": CONTROL_HZ,
        "physics_hz": PHYSICS_HZ,
        "joint_order": list(JOINT_ORDER),
        "default_joint_pos": {j: DEFAULT_JOINT_POS[j] for j in JOINT_ORDER},
        "commands": commands.to_dict(),
        "curriculum": dict(curriculum),
        "reward_scales": dict(reward_scales),
        "domain_randomization": dict(domain_randomization),
        "provenance": {
            "git_sha": git_sha,
            "git_dirty": bool(git_dirty),
            "seed": int(seed),
            "task": task,
            "simulator": simulator,
        },
        "checkpoint_sha256": checkpoint_sha256,
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def validate_manifest_for_mode(
    manifest: Mapping[str, Any] | None,
    mode: str,
    *,
    deploy_action_scale: float | None = None,
    deploy_control_hz: int | None = None,
    deploy_joint_order: Sequence[str] | None = None,
    max_deploy_command: Mapping[str, float] | None = None,
    checkpoint_sha256: str | None = None,
    require_clean_tree: bool = True,
) -> list[str]:
    """Every reason this checkpoint must NOT run in ``mode``. Empty list = allowed.

    Fails closed: a missing manifest, a missing key, an unknown mode or any
    disagreement with this module's contract is a problem.

    ``max_deploy_command`` is the largest command the deploy config can issue,
    as ``{"lin_vel_x": m/s, "lin_vel_y": m/s, "ang_vel_z": rad/s}`` magnitudes; it
    must fit inside the trained ranges.
    """
    if mode not in MODES:
        return [f"unknown mode {mode!r}; expected one of {MODES}"]
    if not isinstance(manifest, Mapping):
        return ["no checkpoint manifest: provenance unknown, refusing"]
    problems: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(
            f"manifest schema {manifest.get('schema')!r} != {MANIFEST_SCHEMA!r} "
            "(legacy or foreign checkpoint)"
        )

    obs = manifest.get("observation") or {}
    if obs.get("schema_version") != OBS_SCHEMA_VERSION:
        problems.append(
            f"observation schema {obs.get('schema_version')!r} != {OBS_SCHEMA_VERSION!r}"
        )
    if obs.get("schema_sha256") != obs_schema_sha256():
        problems.append("observation schema hash does not match this code's 45-D contract")
    if obs.get("actor_dim") != ACTOR_OBS_DIM:
        problems.append(f"actor observation dim {obs.get('actor_dim')!r} != {ACTOR_OBS_DIM}")
    terms = list(obs.get("actor_terms") or [])
    if terms != [t[0] for t in ACTOR_OBS_TERMS]:
        problems.append(f"actor terms {terms} != contract order")
    forbidden = sorted(FORBIDDEN_ACTOR_TERMS.intersection(terms))
    if forbidden:
        problems.append(f"actor depends on signals unavailable on hardware: {forbidden}")

    action = manifest.get("action") or {}
    scale = action.get("scale")
    if not isinstance(scale, (int, float)) or abs(float(scale) - ACTION_SCALE) > 1e-12:
        problems.append(f"action scale {scale!r} != contract {ACTION_SCALE}")
    if deploy_action_scale is not None and (
        not isinstance(scale, (int, float)) or abs(float(scale) - deploy_action_scale) > 1e-12
    ):
        problems.append(f"deploy action_scale {deploy_action_scale} != trained {scale!r}")
    if action.get("dim") != ACTION_DIM:
        problems.append(f"action dim {action.get('dim')!r} != {ACTION_DIM}")

    hz = manifest.get("control_hz")
    if hz != CONTROL_HZ:
        problems.append(f"control_hz {hz!r} != contract {CONTROL_HZ}")
    if deploy_control_hz is not None and deploy_control_hz != hz:
        problems.append(f"deploy control rate {deploy_control_hz} != trained {hz!r}")

    order = list(manifest.get("joint_order") or [])
    if order != list(JOINT_ORDER):
        problems.append("joint_order differs from the contract order")
    if deploy_joint_order is not None and list(deploy_joint_order) != order:
        problems.append("deploy joint_order differs from the trained joint_order")
    pose = manifest.get("default_joint_pos") or {}
    for j in JOINT_ORDER:
        v = pose.get(j)
        if not isinstance(v, (int, float)) or abs(float(v) - DEFAULT_JOINT_POS[j]) > 1e-9:
            problems.append(f"default_joint_pos[{j}]={v!r} != contract {DEFAULT_JOINT_POS[j]}")

    prov = manifest.get("provenance") or {}
    if not _is_hex(prov.get("git_sha"), 40):
        problems.append(f"provenance.git_sha {prov.get('git_sha')!r} is not a full commit sha")
    if require_clean_tree and prov.get("git_dirty") is not False:
        problems.append("checkpoint was trained from a dirty or unknown working tree")
    if not isinstance(prov.get("seed"), int):
        problems.append("provenance.seed missing")

    recorded_sha = manifest.get("checkpoint_sha256")
    if checkpoint_sha256 is not None and recorded_sha != checkpoint_sha256:
        problems.append("checkpoint file hash does not match the manifest")

    try:
        commands = CommandRanges.from_mapping(manifest.get("commands") or {})
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"commands record unusable: {exc!r}")
        return problems

    capable, reasons = derive_locomotion_capable(commands)
    stored = manifest.get("locomotion_capable")
    if stored is not capable:
        problems.append(
            f"stored locomotion_capable={stored!r} disagrees with the value derived from "
            f"the recorded command ranges ({capable})"
        )
    if mode == MODE_VELOCITY:
        if not capable:
            problems.append("checkpoint is not locomotion-capable: " + "; ".join(reasons))
        if max_deploy_command is None:
            problems.append("velocity mode needs the deploy command envelope to check it")
        else:
            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                want = max_deploy_command.get(key)
                trained = _max_abs(getattr(commands, key))
                if not isinstance(want, (int, float)) or want < 0:
                    problems.append(f"deploy command envelope {key}={want!r} invalid")
                elif want > trained + 1e-9:
                    problems.append(
                        f"deploy {key} up to {want} exceeds the trained envelope {trained}"
                    )
    return problems


__all__ = [
    "ACTION_DIM",
    "ACTION_SCALE",
    "ACTOR_OBS_DIM",
    "ACTOR_OBS_TERMS",
    "CONTROL_HZ",
    "CommandRanges",
    "DECIMATION",
    "DEFAULT_JOINT_POS",
    "FORBIDDEN_ACTOR_TERMS",
    "JOINT_ORDER",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "MODES",
    "MODE_STAND",
    "MODE_VELOCITY",
    "OBS_SCHEMA_VERSION",
    "PHYSICS_HZ",
    "build_manifest",
    "derive_locomotion_capable",
    "obs_schema_dict",
    "obs_schema_sha256",
    "obs_slices",
    "validate_manifest_for_mode",
]
