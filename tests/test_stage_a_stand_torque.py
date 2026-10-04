"""Stage A canonical-stand torque check (owner decision 2026-10-04) and the branch source."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from phoenix.sim2real import preflight as pf
from phoenix.sim2real.go2_model import JOINT_TORQUE_LIMITS_NM, POLICY_JOINT_ORDER
from phoenix.sim2real.provenance import (
    DEFAULT_EXPECTED_BRANCH,
    EXPECT_BRANCH_ENV,
    CodeIdentity,
    identity_problems,
    resolve_expected_branch,
)

ORDER = list(POLICY_JOINT_ORDER)
DEFAULT_Q = {n: 0.0 for n in ORDER}
SCALE = 0.25
GAINS = [("handover start", 60.0, 5.0), ("policy", 25.0, 0.5)]


def _check(action, gains=GAINS):
    return pf.canonical_stand_torque_check(
        action, joint_order=ORDER, default_joint_pos=DEFAULT_Q, action_scale=SCALE, gain_sets=gains
    )


def _action(joint: str, value: float):
    a = np.zeros(12)
    a[ORDER.index(joint)] = value
    return a


def _boundary_action(joint: str, kp: float) -> float:
    """The action at which |kp*scale*a| equals exactly 50% of that joint's limit."""
    return 0.5 * JOINT_TORQUE_LIMITS_NM[joint] / (kp * SCALE)


@pytest.mark.parametrize("joint", ["FL_hip_joint", "RR_thigh_joint", "FR_calf_joint"])
def test_torque_check_passes_just_below_and_fails_just_above_half_the_limit(joint) -> None:
    edge = _boundary_action(joint, 60.0)  # the stand-up gains are the binding set
    assert _check(_action(joint, edge * 0.98)).ok
    assert not _check(_action(joint, edge * 1.02)).ok


def test_torque_check_takes_the_max_over_both_gain_sets() -> None:
    joint = "FL_hip_joint"
    a = _action(joint, _boundary_action(joint, 60.0) * 1.1)
    assert _check(a, GAINS[1:]).ok  # policy gains alone would pass
    assert not _check(a, GAINS[:1]).ok
    assert not _check(a).ok  # the handover-start gains bind


def test_torque_check_reports_per_joint_torque_and_margin() -> None:
    chk = _check(_action("FL_calf_joint", 1.0))
    assert "FL_calf_joint 15.00 N m of 22.71 (margin +7.71)" in chk.detail
    assert "worst margin" in chk.detail and "FL_hip_joint 0.00 N m of 11.75" in chk.detail
    assert chk.gating


def test_torque_check_rejects_non_finite_action() -> None:
    assert not _check(_action("FL_hip_joint", float("nan"))).ok


def test_a_bench_prints_the_retired_slew_number_as_info_only(monkeypatch) -> None:
    from phoenix.sim2real import bench_export, deploy_contract

    cfg = yaml.safe_load(
        (pf.REPO_ROOT / "configs/sim2real/deploy_walk_iter2750_stand.yaml").read_text()
    )
    action = np.full(12, 0.745, dtype=np.float32)

    class _Session:
        def get_inputs(self):
            return [SimpleNamespace(name="obs")]

        def run(self, *_):
            return [action[None]]

    a_inf = float(np.max(np.abs(action)))
    monkeypatch.setattr(
        bench_export, "canonical_stand_obs_for_cfg", lambda c: np.zeros(45, np.float32)
    )
    monkeypatch.setattr(pf, "_ort_session", lambda path: (_Session(), "x"))
    monkeypatch.setattr(
        deploy_contract, "_load_manifest", lambda c: {"deploy": {"kp": 25.0, "kd": 0.5}}
    )
    ctx = SimpleNamespace(
        cfg=cfg, lock={"canonical_bench": {"abs_action_inf": a_inf, "max_abs_tol": 1e-5}}
    )
    checks = pf.a_bench(ctx, None)
    names = [c.name for c in checks]
    assert not any("<= 0.175 rad" in n for n in names)
    torque = next(c for c in checks if "torque <= 50%" in c.name)
    info = next(c for c in checks if "retired 0.175 rad" in c.name)
    assert torque.gating and not info.gating and "0.1863" in info.detail
    assert torque.ok == (0.745 * 60.0 * SCALE <= 0.5 * 23.5)


# ------------------------------------------------------------ branch source
def _ident(branch, dirty=False):
    return CodeIdentity(sha="a" * 40, source="git", branch=branch, dirty=dirty)


def test_expected_branch_default_lock_and_env(monkeypatch) -> None:
    monkeypatch.delenv(EXPECT_BRANCH_ENV, raising=False)
    assert resolve_expected_branch() == DEFAULT_EXPECTED_BRANCH == "feat/phoenix-walk-v1-hw"
    assert resolve_expected_branch({"expected_branch": "feat/x"}) == "feat/x"
    monkeypatch.setenv(EXPECT_BRANCH_ENV, "feat/y")
    assert resolve_expected_branch({"expected_branch": "feat/x"}) == "feat/y"


def test_identity_still_fails_on_wrong_branch_and_dirty_tree(monkeypatch) -> None:
    monkeypatch.delenv(EXPECT_BRANCH_ENV, raising=False)
    want = resolve_expected_branch()
    assert identity_problems(_ident(want), expected_branch=want) == []
    assert any(
        "branch is" in p
        for p in identity_problems(
            _ident("feat/causal-viability-replication"), expected_branch=want
        )
    )
    assert any(
        "dirty" in p for p in identity_problems(_ident(want, dirty=True), expected_branch=want)
    )
