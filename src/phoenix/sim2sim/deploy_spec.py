"""Deploy spec: everything the sim2sim gate needs to drive a policy exactly as deploy does.

Pure Python / numpy (no MuJoCo, no torch), so CI imports it.

A :class:`DeploySpec` is built from one JSON file. Three kinds are accepted:

``phoenix-checkpoint-manifest/v1``
    The PhoenixVelocity manifest written beside every checkpoint
    (:mod:`phoenix.velocity.contract`). The observation is built by
    :func:`phoenix.velocity.observation.build_actor_observation` and the targets by
    :func:`phoenix.velocity.observation.actions_to_joint_targets`: the SAME
    functions the hardware node calls. The action is clamped to
    ``[-action_clip, action_clip]`` first (default 1.0, the training wrapper's
    ``clip_actions``). The manifest is validated for velocity mode against the
    contract; any problem is carried as ``manifest_problems`` and fails the gate.
    PD gains come from an optional ``deploy`` block
    (``{"kp": float|[12], "kd": float|[12], "action_clip": float|null}``);
    without it the Phoenix train == deploy gains Kp 25 / Kd 0.5 are used and
    recorded as an assumption.

``phoenix-legacy-checkpoint-manifest/v0``
    The after-the-fact H25 record (48-D: ``base_lin_vel`` fed zeros, then the
    45-D Phoenix terms). Same builder for the 45 dims, zeros prepended.

``phoenix-sim2sim-deploy-spec/v1``
    Explicit spec for a policy from another stack (the positive control). The
    observation is a list of rl_sar-style terms with scales, see
    :data:`TERM_NAMES`.

Joint arrays inside a spec are in the POLICY's joint order (``joint_order``).
The Unitree SDK index of each policy joint is derived by NAME from
:data:`phoenix.sim2real.go2_model.UNITREE_MOTOR_ORDER`; a spec that also states
``sdk_joint_ids_map`` must agree with the derivation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER
from phoenix.velocity.contract import (
    ACTION_SCALE,
    CONTROL_HZ,
    JOINT_ORDER,
    MANIFEST_SCHEMA,
    MODE_VELOCITY,
    validate_manifest_for_mode,
)
from phoenix.velocity.observation import (
    actions_to_joint_targets,
    build_actor_observation,
    projected_gravity_wxyz,
)

SPEC_SCHEMA = "phoenix-sim2sim-deploy-spec/v1"
LEGACY_SCHEMA = "phoenix-legacy-checkpoint-manifest/v0"

OBS_PHOENIX_45 = "phoenix_velocity_45"
OBS_LEGACY_48 = "phoenix_legacy_48_zero_lin_vel"
OBS_TERMS = "terms"

#: Phoenix train == deploy PD gains (configs/sim2real deploy yamls, Isaac
#: UNITREE_GO2_CFG stiffness/damping). Used when a manifest has no deploy block.
PHOENIX_DEFAULT_KP = 25.0
PHOENIX_DEFAULT_KD = 0.5
PHOENIX_DEFAULT_ACTION_CLIP = 1.0

#: rl_sar ``ComputeObservation`` term names this module reproduces.
#: ``lin_vel`` is refused: the GO2 cannot measure it.
TERM_NAMES = ("ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions")
TERM_DIMS = {"ang_vel": 3, "gravity_vec": 3, "commands": 3, "dof_pos": 12, "dof_vel": 12, "actions": 12}

COMMAND_KEYS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
JOINT_GROUPS = ("hip", "thigh", "calf")


class DeploySpecError(ValueError):
    """The spec file cannot describe a runnable deploy path."""


def joint_group(name: str) -> str:
    """``FL_thigh_joint`` -> ``thigh``."""
    parts = name.split("_")
    if len(parts) != 3 or parts[1] not in JOINT_GROUPS or parts[2] != "joint":
        raise DeploySpecError(f"not a GO2 leg joint name: {name!r}")
    return parts[1]


def sdk_ids_for(order: Sequence[str]) -> tuple[int, ...]:
    """Unitree ``motor_cmd`` index of each joint in ``order`` (policy i -> SDK index)."""
    try:
        return tuple(UNITREE_MOTOR_ORDER.index(n) for n in order)
    except ValueError as exc:
        raise DeploySpecError(f"joint order has a name not in the Unitree motor table: {exc}") from exc


def _per_joint(value: Any, n: int, what: str, order: Sequence[str] | None = None) -> np.ndarray:
    """Scalar, ``n`` values in ``order``, or a mapping by joint group or joint name."""
    if isinstance(value, Mapping):
        if order is None:
            raise DeploySpecError(f"{what}: a mapping needs a joint order")
        out = []
        for name in order:
            if name in value:
                out.append(float(value[name]))
            elif joint_group(name) in value:
                out.append(float(value[joint_group(name)]))
            else:
                raise DeploySpecError(f"{what}: no value for {name} (by name or group)")
        value = out
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise DeploySpecError(f"{what} must be a finite scalar or {n} values, got {value!r}")
    return arr


@dataclass(frozen=True)
class DeploySpec:
    name: str
    kind: str
    obs_builder: str
    joint_order: tuple[str, ...]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    action_clip: float | None
    kp: np.ndarray
    kd: np.ndarray
    control_hz: int
    trained_commands: dict[str, tuple[float, float]] | None
    obs_terms: tuple[dict[str, Any], ...] = ()
    clip_obs: float | None = None
    obs_dim: int = 0
    #: Frames fed to the policy, newest first, zero-filled at start (rl_sar
    #: ObservationBuffer "time" priority). 1 = no history. ``obs_dim`` is the
    #: full policy input, ``obs_dim // history_length`` one frame.
    history_length: int = 1
    sdk_joint_ids_map: tuple[int, ...] = ()
    manifest_problems: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- derived
    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(joint_group(n) for n in self.joint_order)

    def with_action_clip(self, clip: float | None) -> DeploySpec:
        """Copy with a different deploy clip (diagnostic arms only)."""
        from dataclasses import replace

        return replace(self, action_clip=clip)

    # ---------------------------------------------------------------- obs
    def build_obs(
        self,
        *,
        gyro_body: np.ndarray,
        quat_wxyz: np.ndarray,
        command: Sequence[float],
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        """Observation for the policy. Joint arrays are in THIS spec's joint order."""
        if self.obs_builder in (OBS_PHOENIX_45, OBS_LEGACY_48):
            obs = build_actor_observation(
                gyro_body=gyro_body,
                quat_wxyz=quat_wxyz,
                command=command,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                last_action=last_action,
            )
            if self.obs_builder == OBS_LEGACY_48:
                obs = np.concatenate([np.zeros(3, dtype=np.float32), obs])
            return obs
        parts: list[np.ndarray] = []
        for term in self.obs_terms:
            name, scale = term["name"], term.get("scale", 1.0)
            if name == "ang_vel":
                v = np.asarray(gyro_body, dtype=np.float64)
            elif name == "gravity_vec":
                v = projected_gravity_wxyz(quat_wxyz)
            elif name == "commands":
                v = np.asarray(command, dtype=np.float64)
            elif name == "dof_pos":
                v = np.asarray(joint_pos, dtype=np.float64) - self.default_joint_pos
            elif name == "dof_vel":
                v = np.asarray(joint_vel, dtype=np.float64)
            else:  # actions
                v = np.asarray(last_action, dtype=np.float64)
            parts.append(v * np.asarray(scale, dtype=np.float64))
        obs = np.concatenate(parts)
        if self.clip_obs is not None:
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        return obs.astype(np.float32)

    # ---------------------------------------------------------------- action
    def postprocess(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``raw -> (action_used, joint_targets, pre_clip_saturated_mask)``, policy order.

        Deploy order of operations: clamp, then scale about the default pose.
        """
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if raw.shape != (len(self.joint_order),):
            raise DeploySpecError(f"policy returned {raw.shape}, expected ({len(self.joint_order)},)")
        if self.action_clip is None:
            sat = np.zeros(raw.shape, dtype=bool)
            action = raw
        else:
            sat = np.abs(raw) > self.action_clip
            action = np.clip(raw, -self.action_clip, self.action_clip)
        if self.obs_builder in (OBS_PHOENIX_45, OBS_LEGACY_48):
            targets = actions_to_joint_targets(action, float(self.action_scale[0]))
        else:
            targets = self.default_joint_pos + self.action_scale * action
        return action, targets, sat

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "obs_builder": self.obs_builder,
            "obs_dim": self.obs_dim,
            "history_length": self.history_length,
            "joint_order": list(self.joint_order),
            "sdk_joint_ids_map": list(self.sdk_joint_ids_map),
            "default_joint_pos": self.default_joint_pos.tolist(),
            "action_scale": self.action_scale.tolist(),
            "action_clip": self.action_clip,
            "kp": self.kp.tolist(),
            "kd": self.kd.tolist(),
            "control_hz": self.control_hz,
            "trained_commands": (
                None
                if self.trained_commands is None
                else {k: list(v) for k, v in self.trained_commands.items()}
            ),
            "obs_terms": list(self.obs_terms),
            "clip_obs": self.clip_obs,
            "manifest_problems": list(self.manifest_problems),
            "assumptions": list(self.assumptions),
            "provenance": dict(self.provenance),
        }


class ObsHistory:
    """Newest-first stack of ``length`` frames, zero-filled at reset (rl_sar semantics)."""

    def __init__(self, length: int, frame_dim: int) -> None:
        self.length = int(length)
        self.frame_dim = int(frame_dim)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros((self.length, self.frame_dim), dtype=np.float32)

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.shape != (self.frame_dim,):
            raise DeploySpecError(f"frame has {frame.shape}, expected ({self.frame_dim},)")
        self._buf[1:] = self._buf[:-1].copy()
        self._buf[0] = frame
        return self._buf.reshape(-1).copy()


# --------------------------------------------------------------------- loaders


def _commands(block: Mapping[str, Any] | None) -> dict[str, tuple[float, float]] | None:
    if not isinstance(block, Mapping):
        return None
    out = {}
    for k in COMMAND_KEYS:
        lo, hi = block[k]
        out[k] = (float(lo), float(hi))
    return out


def _pose(order: Sequence[str], pose: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    if isinstance(pose, Mapping):
        return np.asarray([float(pose[n]) for n in order], dtype=np.float64)
    return _per_joint(pose, len(order), "default_joint_pos")


def _deploy_block(manifest: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float | None, list[str]]:
    deploy = manifest.get("deploy") or {}
    assumptions = []
    order = tuple(manifest.get("joint_order") or JOINT_ORDER)
    if "kp" in deploy and "kd" in deploy:
        kp = _per_joint(deploy["kp"], 12, "deploy.kp", order)
        kd = _per_joint(deploy["kd"], 12, "deploy.kd", order)
    elif "kp" in deploy or "kd" in deploy:
        raise DeploySpecError("deploy block must give both kp and kd")
    else:
        kp = np.full(12, PHOENIX_DEFAULT_KP)
        kd = np.full(12, PHOENIX_DEFAULT_KD)
        assumptions.append(
            f"manifest has no deploy.kp/kd; used Phoenix train==deploy Kp {PHOENIX_DEFAULT_KP} "
            f"Kd {PHOENIX_DEFAULT_KD}"
        )
    if "action_clip" in deploy:
        clip = deploy["action_clip"]
    else:
        clip = PHOENIX_DEFAULT_ACTION_CLIP
        assumptions.append(f"manifest has no deploy.action_clip; used {PHOENIX_DEFAULT_ACTION_CLIP}")
    clip = None if clip is None else float(clip)
    if clip is not None and not clip > 0:
        raise DeploySpecError(f"deploy.action_clip must be > 0 or null, got {clip}")
    return kp, kd, clip, assumptions


def spec_from_phoenix_manifest(
    manifest: Mapping[str, Any],
    *,
    name: str,
    required_envelope: Mapping[str, float] | None,
) -> DeploySpec:
    problems = validate_manifest_for_mode(
        manifest,
        MODE_VELOCITY,
        deploy_action_scale=ACTION_SCALE,
        deploy_control_hz=CONTROL_HZ,
        deploy_joint_order=JOINT_ORDER,
        max_deploy_command=dict(required_envelope) if required_envelope else None,
    )
    kp, kd, clip, assumptions = _deploy_block(manifest)
    order = tuple(manifest.get("joint_order") or JOINT_ORDER)
    if order != JOINT_ORDER:
        raise DeploySpecError("Phoenix manifest joint_order is not the contract order; refusing")
    try:
        trained = _commands(manifest.get("commands"))
    except (KeyError, TypeError, ValueError):
        trained = None
    return DeploySpec(
        name=name,
        kind=MANIFEST_SCHEMA,
        obs_builder=OBS_PHOENIX_45,
        joint_order=order,
        default_joint_pos=_pose(order, manifest.get("default_joint_pos") or {}),
        action_scale=np.full(12, ACTION_SCALE),
        action_clip=clip,
        kp=kp,
        kd=kd,
        control_hz=CONTROL_HZ,
        trained_commands=trained,
        obs_dim=45,
        sdk_joint_ids_map=sdk_ids_for(order),
        manifest_problems=tuple(problems),
        assumptions=tuple(assumptions),
        provenance={"provenance": manifest.get("provenance"), "checkpoint_sha256": manifest.get("checkpoint_sha256")},
    )


def spec_from_legacy_manifest(manifest: Mapping[str, Any], *, name: str) -> DeploySpec:
    order = tuple(manifest["joint_order"])
    if order != JOINT_ORDER:
        raise DeploySpecError("legacy manifest joint_order is not the Phoenix order")
    obs = manifest.get("observation") or {}
    if obs.get("actor_dim") != 48 or list(obs.get("actor_terms") or [])[:1] != ["base_lin_vel"]:
        raise DeploySpecError("legacy manifest is not the 48-D base_lin_vel-first layout")
    kp, kd, clip, assumptions = _deploy_block(manifest)
    assumptions.append(
        "legacy 48-D policy: base_lin_vel dims fed zeros, as deployed "
        f"(deploy_base_lin_vel_source={obs.get('deploy_base_lin_vel_source')!r})"
    )
    action = manifest.get("action") or {}
    if "deploy" not in manifest:
        assumptions.append(
            f"gate applies the patched deploy clip {clip}; the recorded deploy_clip was "
            f"{action.get('deploy_clip')!r}"
        )
    return DeploySpec(
        name=name,
        kind=LEGACY_SCHEMA,
        obs_builder=OBS_LEGACY_48,
        joint_order=order,
        default_joint_pos=_pose(order, manifest["default_joint_pos"]),
        action_scale=np.full(12, float(action.get("scale", ACTION_SCALE))),
        action_clip=clip,
        kp=kp,
        kd=kd,
        control_hz=int(manifest.get("control_hz", CONTROL_HZ)),
        trained_commands=_commands(manifest.get("commands")),
        obs_dim=48,
        sdk_joint_ids_map=sdk_ids_for(order),
        manifest_problems=(
            "legacy manifest: not a PhoenixVelocity checkpoint, never valid for velocity mode",
        ),
        assumptions=tuple(assumptions),
        provenance={"checkpoint_sha256": manifest.get("checkpoint_sha256"), "summary": manifest.get("summary")},
    )


def spec_from_explicit(data: Mapping[str, Any]) -> DeploySpec:
    order = tuple(data["joint_order"])
    if sorted(order) != sorted(JOINT_ORDER):
        raise DeploySpecError("joint_order must name the 12 GO2 leg joints exactly once")
    derived = sdk_ids_for(order)
    stated = data.get("sdk_joint_ids_map")
    problems: list[str] = []
    if stated is not None and tuple(int(i) for i in stated) != derived:
        problems.append(f"stated sdk_joint_ids_map {list(stated)} != derived-by-name {list(derived)}")
    obs = data["observation"]
    terms = tuple({"name": t["name"], "scale": t.get("scale", 1.0)} for t in obs["terms"])
    dim = 0
    for t in terms:
        if t["name"] not in TERM_NAMES:
            raise DeploySpecError(f"unsupported observation term {t['name']!r} (allowed {TERM_NAMES})")
        dim += TERM_DIMS[t["name"]]
    hist = obs.get("history") or {"length": 1}
    hlen = int(hist.get("length", 1))
    if hlen < 1:
        raise DeploySpecError("observation.history.length must be >= 1")
    if hlen > 1 and (hist.get("order") != "newest_first" or hist.get("init") != "zeros"):
        raise DeploySpecError("only history order 'newest_first' with init 'zeros' is supported")
    action = data["action"]
    clip = action.get("clip")
    return DeploySpec(
        name=str(data.get("name", "explicit")),
        kind=SPEC_SCHEMA,
        obs_builder=OBS_TERMS,
        joint_order=order,
        default_joint_pos=_pose(order, data["default_joint_pos"]),
        action_scale=_per_joint(action["scale"], 12, "action.scale"),
        action_clip=None if clip is None else float(clip),
        kp=_per_joint(data["kp"], 12, "kp", order),
        kd=_per_joint(data["kd"], 12, "kd", order),
        control_hz=int(data["control_hz"]),
        trained_commands=_commands(data.get("trained_commands")),
        obs_terms=terms,
        clip_obs=None if obs.get("clip") is None else float(obs["clip"]),
        obs_dim=dim * hlen,
        history_length=hlen,
        sdk_joint_ids_map=derived,
        manifest_problems=tuple(problems),
        assumptions=tuple(data.get("assumptions") or ()),
        provenance=dict(data.get("provenance") or {}),
    )


def load_deploy_spec(
    path: str | Path, *, required_envelope: Mapping[str, float] | None = None
) -> DeploySpec:
    p = Path(path)
    data = json.loads(p.read_text())
    schema = data.get("schema")
    if schema == MANIFEST_SCHEMA:
        return spec_from_phoenix_manifest(data, name=p.parent.name or p.stem, required_envelope=required_envelope)
    if schema == LEGACY_SCHEMA:
        return spec_from_legacy_manifest(data, name=str(data.get("name", p.stem)))
    if schema == SPEC_SCHEMA:
        return spec_from_explicit(data)
    raise DeploySpecError(
        f"{p}: unknown schema {schema!r}; expected {MANIFEST_SCHEMA}, {LEGACY_SCHEMA} or {SPEC_SCHEMA}"
    )


__all__ = [
    "COMMAND_KEYS",
    "DeploySpec",
    "DeploySpecError",
    "JOINT_GROUPS",
    "LEGACY_SCHEMA",
    "ObsHistory",
    "OBS_LEGACY_48",
    "OBS_PHOENIX_45",
    "OBS_TERMS",
    "SPEC_SCHEMA",
    "TERM_NAMES",
    "joint_group",
    "load_deploy_spec",
    "sdk_ids_for",
    "spec_from_explicit",
    "spec_from_legacy_manifest",
    "spec_from_phoenix_manifest",
]
