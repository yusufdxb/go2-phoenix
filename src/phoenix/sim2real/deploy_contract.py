"""Deploy contract, artifact lock, and the stand-only / walking-blocked guard.

Three questions every hardware run has to answer before a motor moves, and this
module answers them without ROS, torch or onnxruntime so they are covered by the
no-hardware test suite and can run first in the preflight:

1. **Is the selected deploy config allowed to run at all?**
   :func:`validate_deploy_contract`. The one rule that matters most: with
   ``observation.base_lin_vel_source: zeros`` the policy is blind to its own body
   velocity, which is acceptable ONLY for a commanded-zero stand. Such a config
   must say ``safety.stand_only: true`` explicitly. That rule is for pre-v2 (48-D) policies.
   A policy whose manifest declares ``phoenix-velocity-obs/v2`` (45-D, no
   base_lin_vel term) is held to a different rule: the config must say v2 too and
   must not name an odometry source for a term the policy does not have. A config
   that is not stand-only (walking) is refused until
   :func:`walking_evidence_problems` finds stages F and G PASSED on hardware, for
   the same onnx / onnx.data / manifest hashes, under the lock of the config's own
   stand-phase twin (see :data:`WALKING_PREREQUISITES`). There is no switch to flip.

2. **Are the files the node will open exactly the ones that were gated?**
   :func:`verify_lock`. A lock file names the deploy config (by semantic hash, so
   the bundle's path-pinned copy still matches) and the SHA-256 of every model
   artifact the config reaches, including the ONNX external-data sidecar.

3. **What exactly was run?** :func:`observed_artifact_hashes` returns the hashes
   that the policy node, the bridge and the preflight write into their records.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from phoenix.real_world.failure_detector import resolve_attitude_intervention_rad

from .activation import PATH_KEYS, file_sha256
from .go2_model import POLICY_JOINT_ORDER, verify_default_pose

LOCK_SCHEMA = "phoenix-deploy-lock/v1"

#: The only base_lin_vel sources a deploy config may name (mirrors
#: ``phoenix.sim2real.observation.BASE_LIN_VEL_SOURCES``; kept literal here so this
#: module has no numpy dependency).
_ZEROS = "zeros"
_ODOM = "odom"

OBS_SCHEMA_V2 = "phoenix-velocity-obs/v2"
_STAGE_SCHEMA = (
    "phoenix-preflight-stage/v1"  # preflight_eval.STAGE_SCHEMA, literal: no heavy imports
)
#: Stages that must have PASSED on hardware before a walking config is allowed.
WALKING_EVIDENCE_STAGES = ("F", "G")
_EVIDENCE_ROLES = ("policy.onnx", "policy.onnx.data", "policy.manifest")
_V2_ACTOR_DIM = 45

#: What has to exist, on hardware, before a walking config is allowed. Walking is a
#: function of this evidence (:func:`walking_evidence_problems`), not of a constant.
WALKING_PREREQUISITES: tuple[str, ...] = (
    "stages F and G (cmd=0 stand of the SAME walking policy, safety.stand_only: true) "
    "recorded GO on hardware, live and not rehearsal, under a lock for the stand-phase "
    "twin of the walking config, with the same policy.onnx, policy.onnx.data and "
    "policy.manifest sha256 as the walking config",
    "the manifest declares phoenix-velocity-obs/v2 (45-D, no base_lin_vel term) and the "
    "config's observation block says so (observation.schema_version)",
    "the deploy runtime builds the v2 45-D observation (ros2_policy_node assumes a 48-D "
    "proprio vector today)",
    "stage H (3 x 10 s stand) for the same artifacts before the first walk: expected, "
    "not enforced here",
    "observation parity and the no-motor dryrun re-run for the v2 artifacts",
    "pre-v2 (48-D) walking configs additionally need /utlidar/robot_odom validated on the "
    "GO2 from a bag with known motion (contents, body-frame twist, latency, noise, dropout) "
    "and base_lin_vel_source: odom",
)

_REQUIRED_POSITIVE_SAFETY_KEYS = (
    "max_runtime_s",
    "estop_timeout_s",
    "sensor_timeout_s",
    "first_message_timeout_s",
)


def _load_manifest(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """The checkpoint manifest ``policy.manifest_path`` names, or None if absent/unreadable."""
    path = (cfg.get("policy") or {}).get("manifest_path")
    if not path:
        return None
    try:
        data = json.loads(Path(str(path)).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _v2_observation_problems(
    observation: Mapping[str, Any], policy: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[str]:
    """Config/manifest consistency for a ``phoenix-velocity-obs/v2`` (45-D) policy."""
    problems: list[str] = []
    mobs = manifest.get("observation") or {}
    if observation.get("schema_version") != OBS_SCHEMA_V2:
        problems.append(
            f"manifest declares {OBS_SCHEMA_V2} but observation.schema_version is "
            f"{observation.get('schema_version')!r}: the config must say v2 too"
        )
    if mobs.get("actor_dim") != _V2_ACTOR_DIM or "base_lin_vel" in (mobs.get("actor_terms") or ()):
        problems.append(
            f"manifest observation is not the v2 layout: actor_dim {mobs.get('actor_dim')!r}, "
            f"terms {mobs.get('actor_terms')!r} (want {_V2_ACTOR_DIM} and no base_lin_vel)"
        )
    source = observation.get("base_lin_vel_source")
    if source != _ZEROS:
        problems.append(
            f"observation.base_lin_vel_source is {source!r} but a v2 policy has no base_lin_vel "
            "term: it must stay 'zeros' (an inert placeholder the node requires), never an "
            "odometry source for an observation the policy does not have"
        )
    if int(policy.get("obs_pad_zeros", 0) or 0) != 0:
        problems.append("policy.obs_pad_zeros must be 0 for the 45-D v2 observation")
    return problems


def validate_deploy_contract(
    cfg: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    evidence_sessions: Sequence[str | Path] | None = None,
) -> list[str]:
    """Return every reason ``cfg`` must not be run on hardware. Empty means allowed.

    Deliberately strict and deliberately literal: a missing key is a problem, not
    a default. ``manifest`` defaults to the file ``policy.manifest_path`` names;
    ``evidence_sessions`` (hardware session directories) defaults to
    :func:`default_session_dirs`.
    """
    problems: list[str] = []
    observation = cfg.get("observation") or {}
    safety = cfg.get("safety") or {}
    policy = cfg.get("policy") or {}
    control = cfg.get("control") or {}

    if manifest is None:
        manifest = _load_manifest(cfg)
    manifest_schema = ((manifest or {}).get("observation") or {}).get("schema_version")
    is_v2 = manifest_schema == OBS_SCHEMA_V2
    if observation.get("schema_version") == OBS_SCHEMA_V2 and manifest is None:
        problems.append(
            f"observation.schema_version is {OBS_SCHEMA_V2} but policy.manifest_path "
            f"{policy.get('manifest_path')!r} cannot be read, so v2 cannot be confirmed"
        )
    elif manifest is not None and observation.get("schema_version") not in (None, manifest_schema):
        problems.append(
            f"observation.schema_version {observation.get('schema_version')!r} disagrees with the "
            f"manifest's {manifest_schema!r}"
        )

    source = observation.get("base_lin_vel_source")
    if is_v2:
        assert manifest is not None
        problems.extend(_v2_observation_problems(observation, policy, manifest))
    elif source not in (_ZEROS, _ODOM):
        problems.append(
            f"observation.base_lin_vel_source must be 'zeros' or 'odom', got {source!r}"
        )

    stand_only = safety.get("stand_only")
    if not isinstance(stand_only, bool):
        problems.append(
            f"safety.stand_only must be set explicitly to true or false, got {stand_only!r}"
        )
    elif stand_only is False:
        if source == _ZEROS and not is_v2:
            problems.append(
                "walking is blocked while observation.base_lin_vel_source=zeros: the policy "
                "would get velocity commands with no body-velocity observation. Set "
                "safety.stand_only: true (commanded-zero stand) or validate odom first."
            )
        evidence = walking_evidence_problems(cfg, sessions=evidence_sessions)
        if evidence:
            problems.append(
                "walking deploy configs are blocked until hardware evidence exists: "
                + "; ".join(evidence)
                + ". Prerequisites: "
                + "; ".join(WALKING_PREREQUISITES)
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
    return problems


def stand_twin(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """``cfg`` with ``safety.stand_only: true``: the commanded-zero stand of the same policy."""
    twin = copy.deepcopy(dict(cfg))
    twin["safety"] = {**(twin.get("safety") or {}), "stand_only": True}
    return twin


def default_session_dirs() -> list[Path]:
    """Hardware session directories: ``$PHOENIX_HW_SESSIONS`` or ``<repo>/logs/hw_sessions``."""
    root = Path(
        os.environ.get("PHOENIX_HW_SESSIONS")
        or Path(__file__).resolve().parents[3] / "logs" / "hw_sessions"
    )
    return sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []


def _stage_record_rejections(
    rec: Mapping[str, Any], stage: str, twin_hash: str, want: Mapping[str, str]
) -> list[str]:
    why: list[str] = []
    if rec.get("schema") != _STAGE_SCHEMA:
        why.append(f"schema {rec.get('schema')!r}")
    if rec.get("stage") != stage:
        why.append(f"record is stage {rec.get('stage')!r}")
    if rec.get("verdict") != "GO":
        why.append(f"verdict {rec.get('verdict')!r}")
    if rec.get("rehearsal"):
        why.append("rehearsal (no robot)")
    if rec.get("motors") != "live":
        why.append(f"motors {rec.get('motors')!r}")
    if rec.get("code_identity_problems"):
        why.append("code identity problems")
    if rec.get("lock_semantic_sha256") != twin_hash:
        why.append("lock is not for the stand-phase twin of this config")
    got = rec.get("lock_artifact_sha256") or {}
    for role, sha in want.items():
        if got.get(role) != sha:
            why.append(f"{role} hash differs from the walking config's")
    return why


def walking_evidence_problems(
    cfg: Mapping[str, Any], *, sessions: Sequence[str | Path] | None = None
) -> list[str]:
    """Why ``cfg`` may not walk yet. Empty means stages F and G have hardware evidence.

    A stage counts when a ``stage_<S>.json`` record in one of ``sessions`` is a GO,
    live, non-rehearsal record whose lock was for the stand-phase twin of ``cfg``
    (same config except ``safety.stand_only: true``) and names the same sha256 for
    policy.onnx, policy.onnx.data and policy.manifest as the files ``cfg`` reaches
    on disk now. Records written before ``lock_artifact_sha256`` existed never count.
    """
    observed = observed_artifact_hashes(cfg, "<walking-config>")["artifacts"]
    want: dict[str, str] = {}
    problems: list[str] = []
    for role in _EVIDENCE_ROLES:
        sha = (observed.get(role) or {}).get("sha256")
        if sha is None:
            problems.append(
                f"cannot hash {role} for the walking config (file missing or not named)"
            )
        else:
            want[role] = sha
    if problems:
        return problems
    twin_hash = semantic_config_sha256(stand_twin(cfg))
    dirs = [Path(d) for d in (default_session_dirs() if sessions is None else sessions)]
    for stage in WALKING_EVIDENCE_STAGES:
        seen: list[str] = []
        ok = False
        for d in dirs:
            path = d / f"stage_{stage}.json"
            if not path.is_file():
                continue
            try:
                rec = json.loads(path.read_text())
            except ValueError:
                seen.append(f"{path}: unreadable")
                continue
            why = _stage_record_rejections(rec, stage, twin_hash, want)
            if not why:
                ok = True
                break
            seen.append(f"{path.parent.name}: " + ", ".join(why))
        if not ok:
            problems.append(
                f"stage {stage} has no GO hardware record for this policy's stand-phase twin "
                f"({len(dirs)} session dir(s) searched; "
                + ("rejected: " + " | ".join(seen[:3]) if seen else "no stage record found")
                + ")"
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
        # Default: the run's latest.pt beside the export. policy.checkpoint_path names
        # the source checkpoint explicitly when no latest.pt exists (an archived export).
        checkpoint = policy.get("checkpoint_path")
        out["checkpoint"] = Path(str(checkpoint)) if checkpoint else onnx_path.with_name("latest.pt")
    manifest_path = policy.get("manifest_path")
    if manifest_path:
        out["policy.manifest"] = Path(str(manifest_path))
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


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARITY_DEFAULTS = {"max_abs_tol": 1.0e-5, "cos_tol": 0.9999, "max_steps_per_input": 2000}


def canonical_bench_record(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The locked canonical-stand bench value, computed the way stage A recomputes it."""
    import numpy as np
    import onnxruntime as ort

    from .bench_export import canonical_stand_obs_for_cfg

    onnx = artifact_paths(cfg)["policy.onnx"]
    obs = canonical_stand_obs_for_cfg(cfg)
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    action = np.asarray(
        session.run(None, {session.get_inputs()[0].name: obs[None].astype(np.float32)})[0]
    ).reshape(-1)
    return {
        "abs_action_inf": float(np.max(np.abs(action))),
        "max_abs_tol": 1.0e-5,
        "legacy_threshold": 0.3,
        "legacy_threshold_is_gate": False,
    }


def parity_input_record(spec: str) -> dict[str, Any]:
    """``PATH:parquet`` or ``PATH:npz:KEY`` (PATH relative to the repo root) to a pinned entry."""
    parts = spec.split(":")
    if len(parts) not in (2, 3) or parts[1] not in ("parquet", "npz"):
        raise ValueError(f"--parity-input {spec!r}: want PATH:parquet or PATH:npz:KEY")
    if (parts[1] == "npz") != (len(parts) == 3):
        raise ValueError(f"--parity-input {spec!r}: npz needs a KEY, parquet takes none")
    path = _REPO_ROOT / parts[0]
    if not path.is_file():
        raise FileNotFoundError(f"parity reference input {path} does not exist")
    entry: dict[str, Any] = {"path": parts[0], "kind": parts[1]}
    if parts[1] == "npz":
        entry["key"] = parts[2]
    entry["sha256"] = file_sha256(path)
    return entry


def build_lock(
    cfg: Mapping[str, Any],
    config_path: str | Path,
    *,
    name: str,
    repo_config_path: str,
    checkpoint: str | Path | None = None,
    parity_inputs: Sequence[str] = (),
    canonical_bench: bool = False,
) -> dict[str, Any]:
    """A lock for ``cfg`` from the files on disk now (the generator; never hand-write hashes).

    ``checkpoint`` must be the file ``artifact_paths(cfg)['checkpoint']`` resolves to
    (``policy.checkpoint_path``, else ``latest.pt`` beside the export): the lock
    hashes what stage A will hash, never a file the config does not reach.
    """
    observed = observed_artifact_hashes(cfg, config_path)
    artifacts = {
        role: {"sha256": e["sha256"], "bytes": e["bytes"]}
        for role, e in observed["artifacts"].items()
        if e["sha256"] is not None
    }
    lock: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "name": name,
        "stand_only": is_stand_only(cfg),
        "deploy_config": {
            "path": repo_config_path,
            "semantic_sha256": observed["deploy_config"]["semantic_sha256"],
        },
    }
    if checkpoint is not None:
        want = artifact_paths(cfg).get("checkpoint")
        if want is None or Path(checkpoint).resolve() != want.resolve():
            raise ValueError(
                f"--checkpoint {checkpoint} is not the checkpoint the config reaches ({want}); "
                "set policy.checkpoint_path in the deploy config to it"
            )
        if "checkpoint" not in artifacts:
            raise FileNotFoundError(f"checkpoint {want} does not exist")
        lock["checkpoint_source"] = "/".join(want.resolve().parts[-3:])
    lock["artifacts"] = artifacts
    if parity_inputs:
        lock["parity_reference_inputs"] = [parity_input_record(s) for s in parity_inputs]
        lock["parity"] = dict(_PARITY_DEFAULTS)
    if canonical_bench:
        lock["canonical_bench"] = canonical_bench_record(cfg)
    return lock


def main(argv: list[str] | None = None) -> int:
    """``python -m phoenix.sim2real.deploy_contract lock|reference-obs ...``."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("lock", help="write a lock from the artifacts on disk")
    lk.add_argument("--config", required=True)
    lk.add_argument("--out", required=True)
    lk.add_argument("--name", required=True)
    lk.add_argument("--checkpoint", help="source checkpoint; must be the one the config reaches")
    lk.add_argument(
        "--parity-input", action="append", default=[], help="PATH:parquet | PATH:npz:KEY (repo-relative)"
    )
    lk.add_argument("--canonical-bench", action="store_true", help="record the canonical-stand |a|_inf")
    ro = sub.add_parser("reference-obs", help="write the deterministic synthetic v2 parity batch")
    ro.add_argument("--config", required=True)
    ro.add_argument("--out", required=True)
    ro.add_argument("--n", type=int, default=2000)
    ro.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.cmd == "reference-obs":
        import numpy as np

        from .observation_v2 import ManifestObservationBuilder, reference_observations

        builder = ManifestObservationBuilder.from_manifest_file(
            cfg["policy"]["manifest_path"], cfg["joint_order"], cfg["control"]["default_joint_pos"]
        )
        obs = reference_observations(builder, n=args.n, seed=args.seed)
        np.savez_compressed(args.out, obs=obs)
        print(f"{args.out} obs{obs.shape} sha256 {file_sha256(args.out)}")
        return 0
    lock = build_lock(
        cfg,
        args.config,
        name=args.name,
        repo_config_path=str(args.config),
        checkpoint=args.checkpoint,
        parity_inputs=args.parity_input,
        canonical_bench=args.canonical_bench,
    )
    extras = "".join(
        [f" --checkpoint {args.checkpoint}" if args.checkpoint else ""]
        + [f" --parity-input {s}" for s in args.parity_input]
        + [" --canonical-bench" if args.canonical_bench else ""]
    )
    header = (
        f"# Generated by `python -m phoenix.sim2real.deploy_contract lock --config {args.config} "
        f"--out {args.out} --name {args.name}{extras}`. Never hand-edit a hash.\n"
    )
    Path(args.out).write_text(header + yaml.safe_dump(lock, sort_keys=False))
    problems, _ = verify_lock(load_lock(args.out), cfg, args.config)
    print("\n".join(problems) or f"lock {args.out} verifies")
    return 1 if problems else 0


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


__all__ = [
    "LOCK_SCHEMA",
    "OBS_SCHEMA_V2",
    "WALKING_EVIDENCE_STAGES",
    "WALKING_PREREQUISITES",
    "artifact_paths",
    "build_lock",
    "default_session_dirs",
    "is_stand_only",
    "load_lock",
    "observed_artifact_hashes",
    "semantic_config_sha256",
    "stand_twin",
    "validate_deploy_contract",
    "verify_lock",
    "walking_evidence_problems",
]

if __name__ == "__main__":
    raise SystemExit(main())
