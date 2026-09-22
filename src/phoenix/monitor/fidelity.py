"""Deployment-fidelity gate: did the robot execute the policy it was given?

Phoenix makes no adaptation claim about a hardware run in which its own execution
stack rewrote the policy. This module measures, from one bridge telemetry file,
how far the target actually sent was from the target the policy requested, and
applies preregistered limits. A run that fails is still useful data; it is simply
not admissible as scientific evidence about the policy or the actuators.

Metrics (over policy ticks that processed a new command, Unitree motor order)
----------------------------------------------------------------------------
* ``altered_fraction``: share of joint-samples where ``|sent - requested|`` exceeds
  ``tol_rad`` (1 mrad). ``altered_fraction_any_change`` counts every change, which
  is the existing ``end_to_end_clip_pct`` of
  :func:`phoenix.sim2real.bridge_telemetry.summarize` as a fraction.
* ``rms_distortion_rad`` / ``max_distortion_rad``: size of those changes. A clip
  that moves a target by 0.1 mrad is not the same finding as one that moves it by
  0.2 rad; the fraction alone hid that.
* ``authority_s``: seconds of continuous policy authority. A run that aborted after
  half a second cannot be used to evaluate a stand, never mind locomotion.
* attribution: the share altered by the policy node's slew clip, by the bridge's
  slew clip and by the hard-limit clip.

Thresholds are frozen in :data:`PREREGISTERED` and recorded in every report. They
are set before any scientific hardware run and must not be tuned on one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from phoenix.sim2real.go2_model import UNITREE_MOTOR_ORDER

from .layers import CommandLayers

SCHEMA = "phoenix-deploy-fidelity/v1"


@dataclass(frozen=True)
class FidelityThresholds:
    max_altered_fraction: float = 0.05
    max_rms_distortion_rad: float = 0.01
    min_authority_s: float = 10.0
    #: A change smaller than 1 mrad is below what the gate calls material. The
    #: exact (any change) fraction is reported too: on the 2026-09-21 motors-off
    #: runs 18-24 % of joint-samples were "clipped" by about 1e-5 rad, which is
    #: why a clip fraction without a magnitude misled earlier reports.
    tol_rad: float = 1e-3


#: Frozen 2026-09-22, before any Phoenix v2 scientific hardware run. Change only
#: with a dated entry in docs/research/EXPERIMENT.md explaining why.
PREREGISTERED = FidelityThresholds()


def _longest_run_s(t_s: np.ndarray, mask: np.ndarray) -> float:
    best, start = 0.0, None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            best = max(best, float(t_s[end] - t_s[start]))
            start = None
    return best


def fidelity_report(
    layers: CommandLayers, thresholds: FidelityThresholds = PREREGISTERED
) -> dict[str, Any]:
    pol = layers.policy_mask
    new = pol & layers.cmd_is_new
    authority = _longest_run_s(layers.t_s, pol)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "thresholds": asdict(thresholds),
        "policy_ticks": int(pol.sum()),
        "policy_new_command_ticks": int(new.sum()),
        "authority_s": authority,
    }
    reasons: list[str] = []
    if not new.any():
        report.update(
            altered_fraction=None,
            rms_distortion_rad=None,
            max_distortion_rad=None,
            per_joint=None,
            attribution=None,
            verdict="FAIL",
            reasons=["no policy ticks: nothing was executed"],
        )
        return report
    tol = thresholds.tol_rad
    d = layers.sent[new] - layers.requested[new]
    altered = np.abs(d) > tol
    pn = np.abs(layers.policy_node_target[new] - layers.requested[new]) > tol
    br = layers.bridge_slew_clip[new]
    lim = layers.limit_clip[new]
    report["altered_fraction"] = float(altered.mean())
    report["altered_fraction_any_change"] = float((np.abs(d) > 0.0).mean())
    report["ticks_with_any_altered_fraction"] = float(altered.any(axis=1).mean())
    report["rms_distortion_rad"] = float(np.sqrt(np.mean(np.square(d))))
    report["max_distortion_rad"] = float(np.max(np.abs(d)))
    report["per_joint"] = {
        name: {
            "altered_fraction": float(altered[:, j].mean()),
            "rms_distortion_rad": float(np.sqrt(np.mean(np.square(d[:, j])))),
        }
        for j, name in enumerate(UNITREE_MOTOR_ORDER)
    }
    report["attribution"] = {
        "policy_node_slew_clip_fraction": (
            float(np.nanmean(pn)) if np.isfinite(layers.policy_node_target[new]).any() else None
        ),
        "bridge_slew_clip_fraction": float(br.mean()),
        "limit_clip_fraction": float(lim.mean()),
    }
    if report["altered_fraction"] > thresholds.max_altered_fraction:
        reasons.append(
            f"altered_fraction {report['altered_fraction']:.3f} > "
            f"{thresholds.max_altered_fraction}"
        )
    if report["rms_distortion_rad"] > thresholds.max_rms_distortion_rad:
        reasons.append(
            f"rms_distortion_rad {report['rms_distortion_rad']:.4f} > "
            f"{thresholds.max_rms_distortion_rad}"
        )
    if authority < thresholds.min_authority_s:
        reasons.append(f"authority_s {authority:.2f} < {thresholds.min_authority_s}")
    report["verdict"] = "PASS" if not reasons else "FAIL"
    report["reasons"] = reasons
    return report


__all__ = ["PREREGISTERED", "SCHEMA", "FidelityThresholds", "fidelity_report"]
