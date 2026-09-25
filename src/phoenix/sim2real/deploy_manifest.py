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

    def to_dict(self) -> dict[str, Any]:
        return {
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

    return DeployManifest(
        action_clip=clip,
        kp=None if kp is None else float(kp),
        kd=None if kd is None else float(kd),
        action_scale=action_scale,
        path=p,
    )


__all__ = ["DeployManifest", "ManifestError", "load_deploy_manifest"]
