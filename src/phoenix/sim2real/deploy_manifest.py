"""Read the ``deploy`` block of a policy checkpoint manifest: the training-matched
action clip, and optional PD gain / action-scale overrides.

Contract (2026-09-24): training moved from ``clip_actions=1.0`` to ``100.0``
(the legged_gym / rl_sar / robot_lab convention). At ``action_scale=0.25`` a
clip of 1.0 pinned the policy at the clamp on close to every step, since 0.25
rad is a tiny fraction of what a learned gait needs; a clip of 100 lets the
network's raw output through unmodified in normal operation and leaves the
clamp as a true outlier guard rather than the policy's everyday operating
point.

The rule going forward: **deploy clip must equal the training clip**, and it
is read from the manifest, never hardcoded. The previous deploy-side default
(``phoenix.sim2real.safety.ACTION_CLIP = 1.0``) was correct for the H25
checkpoint and would have been silently wrong the moment a different
checkpoint trained with a different clip; a manifest with no
``deploy.action_clip`` is refused outright, not defaulted, by
:func:`load_deploy_manifest`.

This is layered UNDERNEATH the real safety net, not instead of it: the torque
limit (:func:`phoenix.sim2real.safety.torque_limited_target_array`,
:mod:`phoenix.sim2real.go2_model` for the N m table) and the hard
position-limit abort band bound the motor command regardless of what the
network was trained to output, even when ``action_clip`` is large enough
that the action clamp itself rarely engages.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """The manifest is missing, unreadable, or its ``deploy`` block is incomplete."""


@dataclass(frozen=True)
class DeployManifest:
    #: The exact training clip_actions value. Deploy must clamp raw actions
    #: to [-action_clip, action_clip], never a hardcoded constant.
    action_clip: float
    #: PD gains the checkpoint was trained/tuned with, if the manifest states
    #: them. When absent the caller falls back to its own deploy-config gains.
    kp: float | None
    kd: float | None
    #: Either a single scalar or one value per joint (in the manifest's own
    #: joint_order, if it names one), overriding the deploy config's
    #: control.action_scale when present.
    action_scale: float | tuple[float, ...] | None
    path: Path
    #: PhoenixVelocity observation schema (contract.ACTOR_OBS_SCALES) when the
    #: manifest has a velocity-contract ``observation`` block, else None (H25-era
    #: stand manifests). The velocity FSM's default builder requires it.
    obs_schema_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_schema_version": self.obs_schema_version,
            "action_clip": self.action_clip,
            "kp": self.kp,
            "kd": self.kd,
            "action_scale": self.action_scale,
            "path": str(self.path),
        }


def load_deploy_manifest(path: Path | str) -> DeployManifest:
    """Read a checkpoint manifest JSON file and return its validated ``deploy`` block.

    Refuses (raises :class:`ManifestError`) if:

    * the file cannot be read or is not valid JSON,
    * there is no ``deploy`` object,
    * ``deploy.action_clip`` is missing, not a number, or not positive and finite.

    ``kp``, ``kd`` and ``action_scale`` are optional; their absence is not an
    error, only ``action_clip`` is mandatory. There is deliberately no
    fallback value for ``action_clip`` anywhere in this function: a missing
    field must stop the node from starting, not run with a guess.
    """
    p = Path(path)
    try:
        text = p.read_text()
    except OSError as exc:
        raise ManifestError(f"manifest {p} could not be read: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {p} is not valid JSON: {exc}") from exc

    deploy = data.get("deploy")
    if not isinstance(deploy, dict):
        raise ManifestError(
            f"manifest {p} has no 'deploy' object; refusing to guess an action clip. "
            "Deploy clip must equal the training clip (see this module's docstring)."
        )
    if "action_clip" not in deploy or deploy["action_clip"] is None:
        raise ManifestError(
            f"manifest {p}: deploy.action_clip is missing. There is no safe default; "
            "deploy clip must equal the training clip."
        )
    try:
        clip = float(deploy["action_clip"])
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            f"manifest {p}: deploy.action_clip={deploy['action_clip']!r} is not a number"
        ) from exc
    if not math.isfinite(clip) or clip <= 0:
        raise ManifestError(
            f"manifest {p}: deploy.action_clip must be positive and finite, got {clip}"
        )

    kp = deploy.get("kp")
    kd = deploy.get("kd")
    action_scale_raw = deploy.get("action_scale")
    action_scale: float | tuple[float, ...] | None
    if action_scale_raw is None:
        action_scale = None
    elif isinstance(action_scale_raw, list):
        action_scale = tuple(float(v) for v in action_scale_raw)
    else:
        action_scale = float(action_scale_raw)

    obs_schema_version: str | None = None
    obs = data.get("observation")
    if isinstance(obs, dict) and str(obs.get("schema_version", "")).startswith(
        "phoenix-velocity-obs/"
    ):
        from phoenix.velocity.contract import UnknownObsSchemaError, manifest_obs_schema

        try:
            obs_schema_version = manifest_obs_schema(data)
        except UnknownObsSchemaError as exc:
            raise ManifestError(f"manifest {p}: {exc}") from exc

    return DeployManifest(
        obs_schema_version=obs_schema_version,
        action_clip=clip,
        kp=None if kp is None else float(kp),
        kd=None if kd is None else float(kd),
        action_scale=action_scale,
        path=p,
    )


def cfg_manifest_mismatches(cfg: Any, manifest_path: Path | str, *, tol: float = 1e-6) -> list[str]:
    """Compare the deploy YAML's control contract with the checkpoint manifest.

    The node takes ``action_clip``/``kp``/``kd`` from the manifest, but joint order,
    default joint positions, action scale and control rate come from the deploy
    YAML (``joint_order``, ``control.*``). Each is a training-contract value, so a
    YAML that disagrees with the manifest is a silent sim-to-real bug (wrong action
    offset AND wrong joint_pos observation reference). Only keys the manifest
    states are compared (H25-era stand manifests state none, so they pass).
    Returns human-readable problems; empty means consistent.
    """
    try:
        data = json.loads(Path(manifest_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"manifest {manifest_path} could not be read: {exc}"]
    control = (cfg.get("control") or {}) if isinstance(cfg, dict) else {}
    problems: list[str] = []
    m_order = data.get("joint_order")
    if m_order is not None and list(cfg.get("joint_order") or []) != list(m_order):
        problems.append(f"joint_order differs from manifest {m_order}")
    m_def = data.get("default_joint_pos")
    if isinstance(m_def, dict):
        c_def = control.get("default_joint_pos") or {}
        for name, val in m_def.items():
            got = c_def.get(name)
            if got is None or abs(float(got) - float(val)) > tol:
                problems.append(f"control.default_joint_pos[{name}]={got} != manifest {val}")
    m_scale = (data.get("action") or {}).get("scale")
    if m_scale is not None and not isinstance(m_scale, list):
        got = control.get("action_scale")
        if got is None or abs(float(got) - float(m_scale)) > tol:
            problems.append(f"control.action_scale={got} != manifest action.scale {m_scale}")
    m_hz = data.get("control_hz")
    if m_hz is not None:
        got = control.get("rate_hz")
        if got is None or abs(float(got) - float(m_hz)) > tol:
            problems.append(f"control.rate_hz={got} != manifest control_hz {m_hz}")
    return problems


__all__ = ["DeployManifest", "ManifestError", "cfg_manifest_mismatches", "load_deploy_manifest"]
