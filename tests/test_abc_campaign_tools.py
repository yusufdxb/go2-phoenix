"""Leg odometry, answer provenance, the A/B/C comparison and the runner's answer prompts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phoenix.sim2real import leg_odometry as lo  # noqa: E402
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ga = _load("gantry_ab_analyze")
cmp = _load("abc_compare")
RUNNER = REPO / "scripts/hw_campaign.sh"


# ------------------------------------------------------------------ leg odometry


def test_foot_positions_zero_pose_match_the_mjcf_chain():
    p = lo.foot_positions_body(np.zeros((1, 12)))[0]
    assert p[lo.LEGS.index("FL")] == pytest.approx([0.1934 - 0.002, 0.0465 + 0.0955, -0.426])
    assert p[lo.LEGS.index("RR")] == pytest.approx([-0.1934 - 0.002, -0.0465 - 0.0955, -0.426])


def test_foot_positions_standing_pose_is_about_0_3_m_below_the_base():
    stand = {"hip": 0.0, "thigh": 0.8, "calf": -1.5}
    q = np.array([[stand[n.split("_")[1]] for n in POLICY_JOINT_ORDER]])
    z = lo.foot_positions_body(q)[0, :, 2]
    assert np.all((z < -0.28) & (z > -0.34))


def test_planted_feet_give_the_base_displacement():
    # four feet planted, base slides forward by 0.1 m: the estimate is the base motion
    l1 = l2 = 0.213
    xs = np.linspace(0.0, 0.1, 51)
    q = np.zeros((51, 12))
    idx = {n: i for i, n in enumerate(POLICY_JOINT_ORDER)}
    for t, bx in enumerate(xs):
        x, z = -bx, -0.30
        c = (x * x + z * z - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        calf = -np.arccos(c)
        thigh = np.arctan2(-x, -z) - np.arctan2(l2 * np.sin(calf), l1 + l2 * np.cos(calf))
        for leg in lo.LEGS:
            q[t, idx[f"{leg}_thigh_joint"]] = thigh
            q[t, idx[f"{leg}_calf_joint"]] = calf
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (51, 1))
    tr = lo.integrate(np.arange(51) * 0.02, q, quat)
    assert tr["base"][-1, 0] == pytest.approx(0.1, abs=0.004)
    assert abs(tr["base"][-1, 1]) < 1e-6


def test_heading_frame_follows_the_initial_yaw():
    yaw = np.radians(90.0)
    rot0 = lo._rotmat_xyzw(np.array([[0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]]))[0]
    fwd, left = lo._heading_frame(rot0)
    assert fwd == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert left == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)


# ------------------------------------------------------------------ provenance


def test_corrections_set_fields_and_keep_what_was_there():
    recs = [{"kind": "run", "harness_log": "x.log", "touched": False}]
    ga.apply_corrections(recs, [{"harness_log": "x.log", "set": {"answers_entered_by": "agent_unconfirmed"}, "reason": "r"}])
    assert recs[0]["answers_entered_by"] == "agent_unconfirmed"
    assert recs[0]["corrections"] == [{"set": {"answers_entered_by": "agent_unconfirmed"}, "was": {"answers_entered_by": None}, "reason": "r"}]
    with pytest.raises(ValueError):
        ga.apply_corrections(recs, [{"harness_log": "missing.log", "set": {}, "reason": "r"}])


def test_agent_unconfirmed_tape_is_not_a_measurement(tmp_path):
    rec = {"block": 1, "arm": "A", "slot": "vx03_4s", "attempt": 1, "run_dir": str(tmp_path),
           "tape_fwd_m": 0.7, "answers_entered_by": "agent_unconfirmed"}
    out = ga.analyze_run(rec)  # no bridge.jsonl: unreadable, but provenance is resolved first
    assert out["answers_entered_by"] == "agent_unconfirmed"
    assert out["dropped_unobserved_measurements"] == ["tape_fwd_m"]
    assert out["valid"] is False


def test_remote_judgement_reads_the_operator_button(tmp_path):
    (tmp_path / "remote_W_vx03_4s_judgement.json").write_text(json.dumps({"mode": "judgement", "result": "yes"}))
    assert ga.remote_judgement(tmp_path, "vx03_4s") == "yes"
    assert ga.remote_judgement(tmp_path / "nope", "vx03_4s") is None


def test_battery_from_the_harness_log(tmp_path):
    log = tmp_path / "h.log"
    log.write_text("[battery] ok   battery 57 % (floor 30 %, 1001 samples) -> x\n")
    assert ga.battery_record({"harness_log": str(log)}, tmp_path)["soc_pre_pct"] == 57.0


# ------------------------------------------------------------------ comparison


def test_permutation_p_is_exact():
    assert cmp.permutation_p([0.5, 0.6, 0.55, 0.58], [0.9, 0.95, 0.92, 0.97]) == pytest.approx(2 / 70)
    assert cmp.permutation_p([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert cmp.permutation_p([1.0], [2.0, 3.0]) is None


def _run(block, arm, f, valid=True, attempt=1, odom=None):
    return {"block": block, "arm": arm, "measured_arm": arm, "slot": "vx03_4s", "attempt": attempt,
            "valid": valid, "answers_entered_by": "operator", "battery": {"soc_pre_pct": 90 - block},
            "walk": {"commanded_fwd_m": 1.2, "F": f, "tape_lat_m": 0.01, "yaw_drift_deg": 1.0,
                     "odom_leg_fwd_m": odom, "F_odom": None if odom is None else odom / 1.2}}


def test_trial_rows_use_the_first_valid_attempt_and_never_the_estimate_as_tape():
    runs = [_run(1, "A", 0.5, valid=False), _run(1, "A", 0.6, attempt=2, odom=0.6), _run(2, "B", None, odom=0.9)]
    rows = cmp.trial_rows(runs, "vx03_4s")
    assert [r["F"] for r in rows] == [pytest.approx(0.6), None]
    assert rows[0]["tape_fwd_m"] == pytest.approx(0.72) and rows[0]["odom_over_tape"] == pytest.approx(0.6 / 0.72)
    assert rows[1]["tape_fwd_m"] is None and rows[1]["odom_leg_fwd_m"] == 0.9  # estimate kept apart


def test_compare_and_plots(tmp_path):
    order = "ABCACBCBABCA"
    f = {"A": 0.6, "B": 0.8, "C": 0.95}
    runs = [_run(i, a, f[a] + 0.01 * (i % 3), odom=0.85 * f[a] * 1.2) for i, a in enumerate(order, 1)]
    rows = cmp.trial_rows(runs, "vx03_4s")
    res = cmp.compare(rows)
    assert res["per_arm"]["A"]["F"]["n"] == 4
    assert res["vs_reference"]["C"]["F"]["mean_diff"] > 0.3
    files = cmp.make_plots(rows, tmp_path, "vx03_4s")
    assert {Path(f).name for f in files} == {"forward.png", "drift.png", "trials.png", "odom_vs_tape.png"}


def test_twelve_trial_arms_file_is_balanced():
    doc = json.loads((REPO / "configs/sim2real/campaigns/abc12_2026-10-18.arms.json").read_text())
    order = doc["order"]
    assert len(order) == 12 and doc["slots"] == ["G", "vx03_4s"]
    assert all(order[i] != order[i + 1] for i in range(11))
    sums = {a: sum(i for i, x in enumerate(order) if x == a) for a in "ABC"}
    assert len(set(sums.values())) == 1 and all(order.count(a) == 4 for a in "ABC")
    old = json.loads((REPO / "configs/sim2real/campaigns/abc_2026-10-17.arms.json").read_text())
    for a in "ABC":
        for k in ("onnx_sha256", "data_sha256", "ckpt_sha256", "stand_lock", "walk_lock"):
            assert doc["arms"][a][k] == old["arms"][a][k]
    assert (REPO / doc["prereg"]).is_file()


# ------------------------------------------------------------------ runner prompts


def _runner_fn(*names: str) -> str:
    src = RUNNER.read_text()
    out = []
    for n in names:
        start = src.index(f"{n}() {{")
        out.append(src[start : src.index("\n}\n", start) + 3])
    return "\n".join(out)


def _bash(fn_src: str, call: str, stdin: str) -> subprocess.CompletedProcess:
    # ask() reads /dev/tty in the runner; here a stub reads stdin
    script = 'ask() { local a; read -r a; echo "$a"; }\n' + fn_src + "\n" + call
    return subprocess.run(["bash", "-c", script], input=stdin, capture_output=True, text=True, timeout=10)


def test_ask_num_reasks_until_a_number_or_empty():
    r = _bash(_runner_fn("ask_num"), 'ask_num "x: "', "n\n0.71\n")
    assert r.stdout.strip() == "0.71" and "not a number: 'n'" in r.stderr
    assert _bash(_runner_fn("ask_num"), 'ask_num "x: "', "\n").stdout.strip() == ""
    assert _bash(_runner_fn("ask_num"), 'ask_num "x: "', "nan\n-0.04\n").stdout.strip() == "-0.04"


def test_entered_by_maps_the_three_answers():
    for k, want in (("o", "operator"), ("r", "agent_relaying_operator"), ("a", "agent_unconfirmed")):
        assert _bash(_runner_fn("entered_by"), "entered_by", f"x\n{k}\n").stdout.strip() == want


def test_runner_records_who_typed_and_never_drops_a_record_silently():
    src = RUNNER.read_text()
    assert '"answers_entered_by": a[15]' in src
    assert "the run record was NOT written" in src
    assert 'tape_f="$(ask_num ' in src and 'meas_yaw="$(ask_num ' in src
