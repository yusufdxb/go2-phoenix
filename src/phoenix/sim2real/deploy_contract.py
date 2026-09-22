"""Deploy contract, artifact lock, and the stand-only / walking-blocked guard.

Three questions every hardware run has to answer before a motor moves, and this
module answers them without ROS, torch or onnxruntime so they are covered by the
no-hardware test suite and can run first in the preflight:

1. **Is the selected deploy config allowed to run at all?**
   :func:`validate_deploy_contract`. The one rule that matters most: with
   ``observation.base_lin_vel_source: zeros`` the policy is blind to its own body
   velocity, which is acceptable ONLY for a commanded-zero stand. Such a config
   must say ``safety.stand_only: true`` explicitly, and a config that is not
   stand-only (walking) is refused outright until the gates in
   :data:`WALKING_PREREQUISITES` have hardware evidence.

2. **Are the files the node will open exactly the ones that were gated?**
   :func:`verify_lock`. A lock file names the deploy config (by semantic hash, so
   the bundle's path-pinned copy still matches) and the SHA-256 of every model
   artifact the config reaches, including the ONNX external-data sidecar.

3. **What exactly was run?** :func:`observed_artifact_hashes` returns the hashes
   that the policy node, the bridge and the preflight write into their records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from phoenix.real_world.failure_detector import resolve_attitude_intervention_rad

from .activation import PATH_KEYS, file_sha256
from .go2_model import POLICY_JOINT_ORDER, verify_default_pose

LOCK_SCHEMA = "phoenix-deploy-lock/v1"

#: The only base_lin_vel sources a deploy config may name (mirrors
#: ``phoenix.sim2real.observation.BASE_LIN_VEL_SOURCES``; kept literal here so this
#: module has no numpy dependency).
_ZEROS = "zeros"
_ODOM = "odom"

#: Walking stays impossible in code, not by convention. Flip this only in the same
#: change that adds the evidence records named below and a walking deploy config
#: that references them.
WALKING_ENABLED = False

#: What has to exist, on hardware, before a walking config is even considered.
WALKING_PREREQUISITES: tuple[str, ...] = (
    "H25 cmd=0 stand passes the corrected hardware gate (final LowCmd bridge "
    "clip telemetry), stages F, G and H",
    "/utlidar/robot_odom validated on the GO2 from a bag with known motion: contents, "
    "body-frame twist confirmed from data (not only child_frame_id), latency, noise, dropout",
    "a walking deploy config with base_lin_vel_source: odom",
    "observation parity and the no-motor dryrun re-run after that change",
)

_REQUIRED_POSITIVE_SAFETY_KEYS = (
    "max_runtime_s",
    "estop_timeout_s",
    "sensor_timeout_s",
    "first_message_timeout_s",
)


def validate_deploy_contract(cfg: Mapping[str, Any]) -> list[str]:
    """Return every reason ``cfg`` must not be run on hardware. Empty means allowed.

    Deliberately strict and deliberately literal: a missing key is a problem, not
    a default.
    """
    problems: list[str] = []
    observation = cfg.get("observation") or {}
    safety = cfg.get("safety") or {}
    policy = cfg.get("policy") or {}
    control = cfg.get("control") or {}

    source = observation.get("base_lin_vel_source")
    if source not in (_ZEROS, _ODOM):
        problems.append(
            f"observation.base_lin_vel_source must be 'zeros' or 'odom', got {source!r}"
        )

    stand_only = safety.get("stand_only")
    if not isinstance(stand_only, bool):
        problems.append(
            f"safety.stand_only must be set explicitly to true or false, got {stand_only!r}"
        )
    elif stand_only is False:
        if source == _ZEROS:
            problems.append(
                "walking is blocked while observation.base_lin_vel_source=zeros: the policy "
                "would get velocity commands with no body-velocity observation. Set "
                "safety.stand_only: true (commanded-zero stand) or validate odom first."
            )
        if not WALKING_ENABLED:
            problems.append(
                "walking deploy configs are blocked. Prerequisites, none of which have "
                "hardware evidence yet: " + "; ".join(WALKING_PREREQUISITES)
            )

    mode_switch = policy.get("mode_switch") or {}
    if mode_switch.get("enabled") and stand_only is not False:
        problems.append(
            "policy.mode_switch.enabled is a walking feature and cannot run in a stand-only config"
        )

    joint_order = tuple(cfg.get("joint_order") or ())
    if joint_order != POLICY_JOINT_ORDER:
        problems.append(
            f"joint_order {list(joint_order)} is not the training order {list(POLICY_JOINT_ORDER)}"
        )
    problems.extend(verify_default_pose(control.get("default_joint_pos") or {}))

    if control.get("rate_hz") != 50:
        problems.append(
            f"control.rate_hz must be 50 (training control rate), got {control.get('rate_hz')!r}"
        )
    if not isinstance(control.get("action_scale"), (int, float)):
        problems.append("control.action_scale is missing")

    for key in _REQUIRED_POSITIVE_SAFETY_KEYS:
        value = safety.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            problems.append(f"safety.{key} must be a positive number, got {value!r}")
    try:
        resolve_attitude_intervention_rad(dict(safety))
    except ValueError as exc:
        problems.append(str(exc))
    if not policy.get("onnx_path"):
        problems.append("policy.onnx_path is missing")
    problems.extend(_limiter_problems(cfg))
    return problems


def _limiter_problems(cfg: Mapping[str, Any]) -> list[str]:
    """Phoenix v2: the soft limiter and the trained action clamp are declared or absent.

    Absent means the incumbent (measured-q clip, no action clamp). Declared means every
    constant is explicit: a command-rate limiter without the tracking abort would drop
    the only effort protection the incumbent clip provided, so that is refused.
    """
    from .safety import LIMITER_MODES, TRAINED_ACTION_CLIP

    problems: list[str] = []
    limiter = cfg.get("limiter")
    control = cfg.get("control") or {}
    clip = control.get("action_clip")
    if clip is not None and (
        not isinstance(clip, (int, float)) or isinstance(clip, bool) or clip != TRAINED_ACTION_CLIP
    ):
        problems.append(
            f"control.action_clip must equal the training clamp {TRAINED_ACTION_CLIP}, got {clip!r}"
        )
    if limiter is None:
        return problems
    mode = limiter.get("mode")
    if mode not in LIMITER_MODES:
        problems.append(f"limiter.mode must be one of {list(LIMITER_MODES)}, got {mode!r}")
    delta = limiter.get("max_delta_per_step")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not 0 < delta <= 0.175:
        problems.append(f"limiter.max_delta_per_step must be in (0, 0.175], got {delta!r}")
    if mode == "prev_command":
        abort = limiter.get("tracking_abort_rad")
        if not isinstance(abort, (int, float)) or isinstance(abort, bool) or abort <= 0:
            problems.append(
                "limiter.tracking_abort_rad must be a positive number with a command-rate "
                f"limiter (it replaces the measured-q clip's effort bound), got {abort!r}"
            )
        hold = limiter.get("tracking_abort_s")
        if not isinstance(hold, (int, float)) or isinstance(hold, bool) or hold <= 0:
            problems.append(f"limiter.tracking_abort_s must be positive, got {hold!r}")
        if clip != TRAINED_ACTION_CLIP:
            problems.append(
                "a command-rate limiter config must declare control.action_clip: "
                f"{TRAINED_ACTION_CLIP} (the trained plant's action clamp)"
            )
    return problems


def is_stand_only(cfg: Mapping[str, Any]) -> bool:
    """True only when the config explicitly declares ``safety.stand_only: true``."""
    return (cfg.get("safety") or {}).get("stand_only") is True


def semantic_config_sha256(cfg: Mapping[str, Any]) -> str:
    """SHA-256 of the config's MEANING, invariant to comments and to path pinning.

    ``activation.py pin`` rewrites every path key to an absolute in-bundle path, so
    the bundle's copy never byte-matches the repo's. Replacing each path value by
    its file name and hashing canonical JSON gives one hash both copies share, while
    any change to a gain, a timeout, the pose, the joint order or a file name still
    changes it.
    """
    return hashlib.sha256(
        json.dumps(_normalize_paths(cfg), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _normalize_paths(node: Any, key: str | None = None) -> Any:
    if isinstance(node, Mapping):
        return {str(k): _normalize_paths(v, str(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_paths(v) for v in node]
    if key in PATH_KEYS and isinstance(node, str):
        return Path(node).name
    return node


def artifact_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    """Every model file the deploy path opens for this config, by lock role.

    Resolved exactly as the node resolves them (a bare ``Path`` of the config
    value), so a hash taken here is a hash of the file the node will load.
    """
    policy = cfg.get("policy") or {}
    out: dict[str, Path] = {}
    onnx = policy.get("onnx_path")
    if onnx:
        onnx_path = Path(str(onnx))
        out["policy.onnx"] = onnx_path
        out["policy.onnx.data"] = onnx_path.with_name(onnx_path.name + ".data")
        out["checkpoint"] = onnx_path.with_name("latest.pt")
    torchscript = policy.get("torchscript_path")
    if torchscript:
        out["policy.pt"] = Path(str(torchscript))
    return out


def observed_artifact_hashes(
    cfg: Mapping[str, Any], config_path: str | Path, *, hasher=file_sha256
) -> dict[str, Any]:
    """Hash the config and every artifact that exists. Missing files hash to ``None``."""
    config_path = Path(config_path)
    record: dict[str, Any] = {
        "deploy_config": {
            "path": str(config_path),
            "sha256": hasher(config_path) if config_path.is_file() else None,
            "semantic_sha256": semantic_config_sha256(cfg),
        },
        "artifacts": {},
    }
    for role, path in sorted(artifact_paths(cfg).items()):
        present = path.is_file()
        record["artifacts"][role] = {
            "path": str(path),
            "resolved": str(path.resolve()) if present else None,
            "sha256": hasher(path) if present else None,
            "bytes": path.stat().st_size if present else None,
        }
    return record


def load_lock(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a lock file. Raises ``ValueError`` if malformed."""
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    if data.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"{path}: schema is {data.get('schema')!r}, expected {LOCK_SCHEMA!r}")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError(f"{path}: artifacts must be a non-empty mapping")
    for role, entry in artifacts.items():
        if not isinstance(entry, Mapping) or not _is_sha256(entry.get("sha256")):
            raise ValueError(f"{path}: artifacts.{role}.sha256 is not a sha256 hex digest")
    deploy_config = data.get("deploy_config") or {}
    if not _is_sha256(deploy_config.get("semantic_sha256")):
        raise ValueError(f"{path}: deploy_config.semantic_sha256 is not a sha256 hex digest")
    return data


def verify_lock(
    lock: Mapping[str, Any],
    cfg: Mapping[str, Any],
    config_path: str | Path,
    *,
    required_roles: tuple[str, ...] = ("policy.onnx", "policy.onnx.data"),
    hasher=file_sha256,
) -> tuple[list[str], dict[str, Any]]:
    """Compare what is on disk against the lock. Returns ``(problems, observed)``.

    ``required_roles`` lists the artifacts that must exist and match. The policy
    node needs only the ONNX pair; the preflight also requires the TorchScript
    export and the source checkpoint. Any role the lock names that IS present on
    disk must match, whether or not it is required, so a present-but-wrong file is
    never silently tolerated.
    """
    observed = observed_artifact_hashes(cfg, config_path, hasher=hasher)
    problems: list[str] = []

    want_semantic = (lock.get("deploy_config") or {}).get("semantic_sha256")
    got_semantic = observed["deploy_config"]["semantic_sha256"]
    if got_semantic != want_semantic:
        problems.append(
            f"deploy config {config_path} semantic sha256 {got_semantic[:12]} != lock "
            f"{str(want_semantic)[:12]}: the config is not the one that was gated"
        )

    locked = lock.get("artifacts") or {}
    for role in required_roles:
        if role not in locked:
            problems.append(f"lock does not name required artifact {role}")
    for role, entry in sorted(locked.items()):
        seen = observed["artifacts"].get(role)
        if seen is None:
            problems.append(f"lock names {role} but the config reaches no such file")
            continue
        if seen["sha256"] is None:
            if role in required_roles:
                problems.append(f"required artifact {role} is missing at {seen['path']}")
            continue
        if seen["sha256"] != entry["sha256"]:
            problems.append(
                f"{role} at {seen['path']} hashes {seen['sha256'][:12]} but the lock records "
                f"{entry['sha256'][:12]}"
            )
    return problems, observed


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


__all__ = [
    "LOCK_SCHEMA",
    "WALKING_ENABLED",
    "WALKING_PREREQUISITES",
    "artifact_paths",
    "is_stand_only",
    "load_lock",
    "observed_artifact_hashes",
    "semantic_config_sha256",
    "validate_deploy_contract",
    "verify_lock",
]
