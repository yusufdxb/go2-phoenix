"""REDEPLOY: only a PROMOTE decision may replace the incumbent policy.

A candidate becomes the robot's new incumbent (the policy Phoenix monitors next)
only when a :func:`phoenix.validate.candidate_gate.candidate_gate` decision says
PROMOTE for the exact checkpoint the deploy lock pins. Experiment arms that are
deployed for comparison (for example the broad-randomisation policy) are NOT
promoted and do not need this; they need the deployment-fidelity and parity
preconditions of the experiment protocol instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .candidate_gate import SCHEMA


def promotion_problems(decision: Mapping[str, Any], lock: Mapping[str, Any]) -> list[str]:
    """Why ``lock`` may not be deployed as the new incumbent under ``decision``."""
    problems: list[str] = []
    if decision.get("schema") != SCHEMA:
        problems.append(f"decision schema {decision.get('schema')!r}, expected {SCHEMA!r}")
    if decision.get("decision") != "PROMOTE":
        problems.append(f"decision is {decision.get('decision')!r}, not PROMOTE")
    artifacts = lock.get("artifacts") or {}
    ckpt = artifacts.get("checkpoint")
    lock_sha = ckpt.get("sha256") if isinstance(ckpt, Mapping) else ckpt
    if not lock_sha:
        problems.append("lock pins no checkpoint hash")
    elif lock_sha != decision.get("candidate_sha256"):
        problems.append("lock checkpoint is not the checkpoint the gate promoted")
    return problems


__all__ = ["promotion_problems"]
