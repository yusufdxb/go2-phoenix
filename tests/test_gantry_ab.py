"""Exploratory gantry A/B package: decision rule, arm identity, and runner safety properties."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/gantry_ab_campaign.sh"
PREREG = REPO / "docs/preregistration/2026-10-07-gantry-ab-exploratory.md"


def _analyze():
    spec = importlib.util.spec_from_file_location(
        "gantry_ab_analyze", REPO / "scripts/gantry_ab_analyze.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gantry_ab_analyze"] = mod
    spec.loader.exec_module(mod)
    return mod


ga = _analyze()


def _chosen(y_a, f_a, y_b, f_b, tilt=(2.0, 2.0), dq=(0.03, 0.03)):
    runs = []
    for arm, ys, fs, ti, d in (("A", y_a, f_a, tilt[0], dq[0]), ("B", y_b, f_b, tilt[1], dq[1])):
        for k, (y, f) in enumerate(zip(ys, fs, strict=True)):
            runs.append(
                {
                    "arm": arm,
                    "measured_arm": arm,
                    "block": k,
                    "slot": "yaw05_3s",
                    "walk": {"Y": y, "yaw_rate_overshoot": 1.0},
                    "safety_event": False,
                }
            )
            runs.append(
                {
                    "arm": arm,
                    "measured_arm": arm,
                    "block": k,
                    "slot": "vx03_4s",
                    "walk": {"F": f},
                    "safety_event": False,
                }
            )
            runs.append(
                {
                    "arm": arm,
                    "measured_arm": arm,
                    "block": k,
                    "slot": "G",
                    "stand": {"max_tilt_deg": ti, "dq_rms_max_calf": d, "calf_sag_median": 0.13},
                    "safety_event": False,
                }
            )
    return runs


def test_decision_better_worse_inconclusive_insufficient():
    r = _chosen([0.2, 0.22, 0.21], [0.6, 0.58, 0.61], [0.5, 0.55, 0.6], [0.8, 0.82, 0.79])
    assert ga.decide(r, r)["verdict"] == "B OBVIOUSLY BETTER"
    r = _chosen([0.5, 0.55, 0.6], [0.8, 0.82, 0.79], [0.2, 0.22, 0.21], [0.6, 0.58, 0.61])
    assert ga.decide(r, r)["verdict"] == "B OBVIOUSLY WORSE"
    r = _chosen([0.2, 0.5, 0.21], [0.6, 0.58, 0.61], [0.45, 0.55, 0.6], [0.8, 0.82, 0.79])
    assert ga.decide(r, r)["verdict"] == "INCONCLUSIVE"  # yaw not separated
    r = _chosen([0.2], [0.6], [0.5], [0.8])
    assert ga.decide(r, r)["verdict"] == "INSUFFICIENT"


def test_better_requires_no_new_safety_event_and_no_worse_stand():
    r = _chosen([0.2, 0.22, 0.21], [0.6, 0.58, 0.61], [0.5, 0.55, 0.6], [0.8, 0.82, 0.79])
    r[-1]["safety_event"] = True  # a B-only safety event
    assert ga.decide(r, r)["verdict"] == "B OBVIOUSLY WORSE"
    r = _chosen(
        [0.2, 0.22, 0.21], [0.6, 0.58, 0.61], [0.5, 0.55, 0.6], [0.8, 0.82, 0.79], tilt=(2.0, 5.0)
    )
    assert ga.decide(r, r)["verdict"] == "INCONCLUSIVE"


def test_first_valid_uses_first_valid_attempt_and_never_a_third():
    runs = [
        {"block": 1, "slot": "G", "attempt": 1, "valid": False},
        {"block": 1, "slot": "G", "attempt": 2, "valid": True},
        {"block": 2, "slot": "G", "attempt": 3, "valid": True},
    ]
    chosen = ga.first_valid(runs)
    assert [(c["block"], c["attempt"]) for c in chosen] == [(1, 2)]


def test_arm_identity_from_artifact_hashes_only():
    for arm, want in ga.ARMS.items():
        man = {
            "artifacts": {
                k: {"sha256": want[k]} for k in ("checkpoint", "policy.onnx", "policy.onnx.data")
            }
        }
        assert ga.arm_of(man) == arm
    mixed = {
        "artifacts": {
            "checkpoint": {"sha256": ga.ARMS["A"]["checkpoint"]},
            "policy.onnx": {"sha256": ga.ARMS["B"]["policy.onnx"]},
            "policy.onnx.data": {"sha256": ga.ARMS["B"]["policy.onnx.data"]},
        }
    }
    assert ga.arm_of(mixed) is None


def test_quat_yaw_xyzw():
    yaw = np.radians(30.0)
    q = np.array([[0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]])
    assert ga.quat_yaw(q)[0] == pytest.approx(yaw)


def test_runner_hashes_match_analysis_and_prereg():
    src, pre = RUNNER.read_text(), PREREG.read_text()
    for arm, want in ga.ARMS.items():
        for key, table in (
            ("policy.onnx", "ONNX_SHA"),
            ("policy.onnx.data", "DATA_SHA"),
            ("checkpoint", "CKPT_SHA"),
        ):
            block = src[src.index(f"declare -A {table}") :].split(")", 1)[0]
            assert f"[{arm}]={want[key]}" in block
            assert want[key] in pre


def test_runner_walks_only_the_two_preregistered_presets_inside_blocks():
    src = RUNNER.read_text()
    assert "ORDER=(A B B A B A)" in src
    assert re.search(r"for slot in G vx03_4s yaw05_3s; do", src)
    walk_calls = re.findall(r'run_harness "\$arm" walk [^\n]*', src)
    assert len(walk_calls) == 1 and 'W "$slot"' in walk_calls[0]
    assert "vx05" not in src and "vxm03" not in src
    assert '[[ "$stage" =~ ^[ABCDEFG]$ ]]' in src  # the ladder can never run W
    # never pre-answers a LIVE phrase and never switches rehearsal on (it only records it)
    assert "PHOENIX_CONFIRM" not in src
    assert not re.search(r"PHOENIX_REHEARSAL=|export PHOENIX_REHEARSAL", src)


def test_runner_refuses_without_a_campaign(tmp_path):
    res = subprocess.run(["bash", str(RUNNER), "status"], cwd=REPO, capture_output=True, text=True)
    if (REPO / "logs/gantry_ab/CURRENT").exists():
        pytest.skip("a campaign exists in this checkout")
    assert res.returncode != 0
    assert "no campaign" in res.stderr or "clean checkout" in res.stderr


def _bridge(tmp_path, source, verified, sha="c" * 40):
    arm = ga.ARMS["B"]
    man = {
        "record": "manifest",
        "live": True,
        "code_identity_problems": [],
        "code_identity": {"source": source, "manifest_verified": verified, "sha": sha},
        "artifacts": {
            k: {"sha256": arm[k]} for k in ("checkpoint", "policy.onnx", "policy.onnx.data")
        },
    }
    tick = {"record": "tick", "t_mono_ns": 0, "mode": "silent", "faults": []}
    d = tmp_path / source
    d.mkdir()
    (d / "bridge.jsonl").write_text("\n".join(json.dumps(r) for r in (man, tick)) + "\n")
    return d


def test_rehearsal_and_foreign_commit_runs_are_invalid(tmp_path):
    rehearsal = ga.analyze_run(
        {
            "run_dir": str(_bridge(tmp_path, "git", None)),
            "arm": "B",
            "slot": "stop_drill",
            "block": 1,
        }
    )
    assert not rehearsal["valid"]
    assert any("not a verified payload run" in w for w in rehearsal["why"])
    d = _bridge(tmp_path, "payload_sync", True)
    other = ga.analyze_run(
        {
            "run_dir": str(d),
            "arm": "B",
            "slot": "stop_drill",
            "block": 1,
            "campaign_commit": "d" * 40,
        }
    )
    assert any("not the campaign commit" in w for w in other["why"])
    flagged = ga.analyze_run(
        {"run_dir": str(d), "arm": "B", "slot": "stop_drill", "block": 1, "rehearsal": True}
    )
    assert any("PHOENIX_REHEARSAL" in w for w in flagged["why"])


def test_physical_attempts_count_across_reruns_and_cap_at_two():
    recs = [
        {"block": 1, "slot": "G", "attempt": a} for a in (1, 2, 1)
    ]  # block re-run resets attempt
    ga.number_physical_attempts(recs)
    assert [r["physical_attempt"] for r in recs] == [1, 2, 3]
    runs = [dict(r, valid=(i == 2)) for i, r in enumerate(recs)]
    assert ga.first_valid(runs) == []  # the only valid one is the 3rd physical attempt


def test_stop_drill_faults_are_not_exempt(tmp_path):
    d = _bridge(tmp_path, "payload_sync", True)
    with (d / "bridge.jsonl").open("a") as f:
        f.write(json.dumps({"record": "end", "faults": ["fall_detected"]}) + "\n")
    r = ga.analyze_run(
        {"run_dir": str(d), "arm": "B", "slot": "stop_drill", "block": 5, "aborted": True}
    )
    assert r["safety_event"] and any("faults" in w for w in r["why"])
    assert not any("operator" in w for w in r["why"])  # the drill's deliberate release is expected


def test_ladder_safety_event_makes_b_worse():
    r = _chosen([0.2, 0.22, 0.21], [0.6, 0.58, 0.61], [0.5, 0.55, 0.6], [0.8, 0.82, 0.79])
    assert ga.decide(r, r, [{"arm": "B", "stage": "E"}])["verdict"] == "B OBVIOUSLY WORSE"


def test_runner_never_reruns_a_started_block():
    src = RUNNER.read_text()
    assert "k not in started" in src and "abandon)" in src


# ---------------------------------------------------------------- Amendment 4
# Policy-exit (deadman release) detection. Every run below is built from VERBATIM real
# bridge tick records (tests/fixtures/gantry_ab/real_policy_exits.json): the 10 ticks
# around the real deadman release of 20261007_fa58780dd4f8/FP_20261007T005825Z and the
# 10 ticks around the real planned end of 20261006_a14ab5939c29/G1_20261006T185511Z. The
# policy episode is lengthened by repeating the first real policy tick backwards at the
# real tick spacing; nothing else is synthesized unless a test says so.

FIXTURE = json.loads((REPO / "tests/fixtures/gantry_ab/real_policy_exits.json").read_text())
REL = FIXTURE["release"]["ticks"]  # 4 policy ticks, then 6 hold ticks (estop true)
END = FIXTURE["planned_end"]["ticks"]  # 4 policy ticks, then 6 hold ticks (estop false)


def _ticks(real, episode_s):
    real = json.loads(json.dumps(real))
    first, dt = real[0], real[1]["t_mono_ns"] - real[0]["t_mono_ns"]
    k_last = max(i for i, t in enumerate(real) if t["mode"] == "policy")
    span = real[k_last]["t_mono_ns"] - first["t_mono_ns"]
    n = max(0, round((episode_s * 1e9 - span) / dt))
    pre = []
    for j in range(n, 0, -1):
        t = json.loads(json.dumps(first))
        t["t_mono_ns"] = first["t_mono_ns"] - j * dt
        t["tick"] = first["tick"] - j
        pre.append(t)
    return pre + real


def _run_dir(tmp_path, name, ticks):
    arm = ga.ARMS["B"]
    man = {
        "record": "manifest",
        "live": True,
        "code_identity_problems": [],
        "code_identity": {"source": "payload_sync", "manifest_verified": True, "sha": "c" * 40},
        "artifacts": {
            k: {"sha256": arm[k]} for k in ("checkpoint", "policy.onnx", "policy.onnx.data")
        },
    }
    faults = sorted({f for t in ticks for f in (t.get("faults") or [])} | {"bridge_shutdown"})
    d = tmp_path / name
    d.mkdir()
    (d / "bridge.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [man, *ticks, {"record": "end", "faults": faults}]) + "\n"
    )
    return d


def _an(tmp_path, name, ticks, slot, **rec):
    return ga.analyze_run(
        {"run_dir": str(_run_dir(tmp_path, name, ticks)), "arm": "B", "slot": slot, "block": 5}
        | rec
    )


def _not_exit_related(why):
    keys = ("released", "policy exit", "STOP drill", "ended by", "operator")
    return [w for w in why if any(k in w for k in keys)]


def test_fixture_is_the_real_sequence_that_exposed_the_bug():
    assert [t["mode"] for t in REL] == ["policy"] * 4 + ["hold"] * 6
    # the bridge never logs estop true on a policy tick: the old analyzer, which searched
    # only policy ticks for the release, could never fire on this sequence
    assert all(t["estop_value"] is False for t in REL if t["mode"] == "policy")
    assert REL[4]["estop_value"] is True
    assert REL[4]["hold_cause"] == "policy_abort:external_estop"
    assert [t["tick"] for t in REL] == list(range(4766, 4776))
    assert [t["mode"] for t in END] == ["policy"] * 4 + ["hold"] * 6
    assert END[4]["hold_cause"] == "policy_abort:authority_window_complete"
    assert END[4]["estop_value"] is False


def test_case_b_release_visible_only_on_the_first_non_policy_tick_is_detected():
    ex = ga.policy_exit(json.loads(json.dumps(REL)), None)
    assert ex["exit"] == "deadman_release" and ex["problems"] == []
    assert ex["k_policy_last"] == 3
    assert ex["t_policy_last_ns"] == 4760006217728  # raw t_mono_ns of the last policy tick
    assert ex["t_release_obs_ns"] == 4760009421344  # raw t_mono_ns of the first hold tick
    assert ex["t_release_rx_ns"] == 4760009421344 - 363104  # minus raw estop_age_s
    assert ex["t_zero_ns"] == 4760009421344  # the hold entry publishes the measured q, held


def test_case_b_real_release_as_a_stop_drill_is_valid_and_not_a_safety_event(tmp_path):
    r = _an(tmp_path, "drill", _ticks(REL, 2.0), "stop_drill", aborted=True)
    assert r["valid"] and r["why"] == [], r["why"]
    assert not r["safety_event"]
    ex = r["policy_exit"]
    assert ex["exit"] == "deadman_release"
    assert abs(ex["policy_episode_s"] - 2.0) < 0.004


def test_case_k_stop_latency_from_raw_timestamps(tmp_path):
    r = _an(tmp_path, "k0", _ticks(REL, 2.0), "stop_drill")
    ex = r["policy_exit"]
    assert ex["stop_latency_s"] == 0.0  # t_zero - t_release_obs: the bridge holds in-tick
    assert ex["release_window_s"] == (4760009421344 - 4760006217728) / 1e9
    assert ex["release_rx_to_zero_s"] == 363104 / 1e9
    # a first non-policy tick that still carries a policy target is NOT zero command:
    # t_zero moves to the next tick that holds the posture measured at hold entry
    t = _ticks(REL, 2.0)
    k = max(i for i, x in enumerate(t) if x["mode"] == "policy") + 1
    t[k]["requested_target_unitree"] = t[k - 1]["requested_target_unitree"]
    t[k]["final_target_unitree"] = t[k - 1]["final_target_unitree"]
    for j in range(k + 1, len(t)):
        t[j]["final_target_unitree"] = t[k]["q_unitree"]
    ex = _an(tmp_path, "k1", t, "stop_drill")["policy_exit"]
    assert ex["t_zero_ns"] == t[k + 1]["t_mono_ns"]
    assert ex["stop_latency_s"] == (t[k + 1]["t_mono_ns"] - t[k]["t_mono_ns"]) / 1e9


def test_case_a_estop_on_a_policy_tick_is_a_release_and_fails_closed(tmp_path):
    # Not seen on real hardware (the bridge holds on the tick it reads the release); if a
    # log ever shows it, the release is still detected and the run fails closed.
    t = _ticks(REL, 2.0)
    k = max(i for i, x in enumerate(t) if x["mode"] == "policy")
    t[k]["estop_value"] = True
    g = _an(tmp_path, "a_g", t, "G")
    assert g["aborted_from_logs"] and g["safety_event"] and not g["valid"]
    d = _an(tmp_path, "a_d", t, "stop_drill", aborted=True)
    assert not d["valid"] and d["safety_event"]
    assert any("deadman not shown held" in w for w in d["why"])
    assert d["policy_exit"]["stop_latency_s"] is None


def test_case_c_normal_completion_is_not_an_abort(tmp_path):
    r = _an(tmp_path, "c", _ticks(END, 5.0145), "G", aborted=False)
    assert r["policy_exit"]["exit"] == "planned_end" and r["policy_exit"]["problems"] == []
    assert not r["aborted_from_logs"] and r["operator_log_discrepancy"] is None
    assert _not_exit_related(r["why"]) == [], r["why"]
    assert not r["safety_event"]
    # the same planned end in a STOP drill: drill invalid (no release), not a safety event
    d = _an(tmp_path, "c_d", _ticks(END, 5.0145), "stop_drill")
    assert any("not released during policy authority" in w for w in d["why"])
    assert not d["safety_event"]
    # a planned end claimed far from the authority window is not a planned end
    e = _an(tmp_path, "c_e", _ticks(END, 3.0), "G")
    assert not e["valid"] and e["safety_event"]
    assert any("authority window 5.000 s" in w for w in e["why"])


def test_case_d_operator_abort_confirmed_by_logs(tmp_path):
    r = _an(tmp_path, "d", _ticks(REL, 2.0), "G", aborted=True)
    assert r["operator_aborted"] and r["aborted_from_logs"]
    assert r["operator_log_discrepancy"] is None
    assert r["safety_event"] and not r["valid"]


def test_case_e_unreported_abort_is_detected_from_logs(tmp_path):
    r = _an(tmp_path, "e", _ticks(REL, 2.0), "G")  # no operator answer at all
    assert r["aborted_from_logs"] and r["safety_event"] and not r["valid"]
    assert any("aborted run, from the logs" in w for w in r["why"])


def test_case_f_operator_says_no_abort_but_logs_show_one(tmp_path):
    r = _an(tmp_path, "f", _ticks(REL, 2.0), "G", aborted=False)
    assert r["aborted_from_logs"] and r["safety_event"] and not r["valid"]
    assert "logs show exit deadman_release" in r["operator_log_discrepancy"]
    # and the reverse: operator abort kept although the logs show a planned end
    o = _an(tmp_path, "f2", _ticks(END, 5.0145), "G", aborted=True)
    assert o["safety_event"] and not o["valid"] and o["operator_log_discrepancy"]


def test_case_g_release_signal_missing_is_invalid(tmp_path):
    for slot in ("stop_drill", "G"):
        t = _ticks(REL, 2.0)
        k = max(i for i, x in enumerate(t) if x["mode"] == "policy") + 1
        del t[k]["estop_value"]
        r = _an(tmp_path, f"g_{slot}", t, slot)
        assert not r["valid"] and r["safety_event"]
        assert any("release signal missing" in w for w in r["why"])
        assert r["policy_exit"]["stop_latency_s"] is None


def test_case_h_first_post_policy_tick_missing_is_invalid(tmp_path):
    t = _ticks(REL, 2.0)
    k = max(i for i, x in enumerate(t) if x["mode"] == "policy")
    cut = _an(tmp_path, "h_cut", t[: k + 1], "stop_drill")  # log ends under policy
    assert not cut["valid"] and cut["safety_event"]
    assert any("first post-policy tick missing" in w for w in cut["why"])
    dropped = _an(tmp_path, "h_drop", t[: k + 1] + t[k + 2 :], "stop_drill")  # record lost
    assert not dropped["valid"] and dropped["safety_event"]
    assert any("tick counter" in w for w in dropped["why"])
    assert dropped["policy_exit"]["stop_latency_s"] is None


def test_case_i_duplicate_timestamp_or_contradictory_transition_fails_closed(tmp_path):
    t = _ticks(REL, 2.0)
    k = max(i for i, x in enumerate(t) if x["mode"] == "policy")
    t[k + 1]["t_mono_ns"] = t[k]["t_mono_ns"]
    r = _an(tmp_path, "i_dup", t, "stop_drill")
    assert not r["valid"] and r["safety_event"] and r["policy_exit"]["stop_latency_s"] is None
    assert any("strictly increasing" in w for w in r["why"])
    t = _ticks(END, 5.0145)
    k = max(i for i, x in enumerate(t) if x["mode"] == "policy")
    t[k + 1]["estop_value"] = True  # release on the planned-end tick: cannot be told apart
    r = _an(tmp_path, "i_amb", t, "G")
    assert not r["valid"] and r["safety_event"]
    assert any("ambiguous transition" in w for w in r["why"])
    t = _ticks(REL, 2.0)
    t[max(i for i, x in enumerate(t) if x["mode"] == "policy") + 1]["estop_age_s"] = 0.5
    r = _an(tmp_path, "i_rx", t, "stop_drill")  # receipt before the last policy tick
    assert not r["valid"] and any("receipt time" in w for w in r["why"])


def test_case_j_stop_that_never_reaches_zero_command_fails_the_drill(tmp_path):
    t = _ticks(REL, 2.0)
    k = max(i for i, x in enumerate(t) if x["mode"] == "policy") + 1
    for j in range(k, len(t)):  # target keeps moving after the release
        t[j]["final_target_unitree"] = [v + 0.001 * (j - k + 1) for v in t[j]["q_unitree"]]
    r = _an(tmp_path, "j", t, "stop_drill", aborted=True)
    assert not r["valid"] and r["safety_event"]
    assert any("STOP drill FAILED" in w for w in r["why"])
    assert r["policy_exit"]["stop_latency_s"] is None
    short = _an(tmp_path, "j2", _ticks(REL, 2.0)[:-5], "stop_drill")  # one hold tick only
    assert any("STOP drill FAILED" in w for w in short["why"])


def test_case_l_release_during_an_ordinary_trial_is_a_safety_event(tmp_path):
    w = _an(tmp_path, "l_w", _ticks(REL, 6.0), "vx03_4s", aborted=False, tape_fwd_m=1.0)
    assert w["policy_exit"]["authority_s"] == 13.0
    assert w["aborted_from_logs"] and w["safety_event"] and w["operator_log_discrepancy"]
    assert ga.stop_conditions([w, w]) == ["arm B vx03_4s: 2 aborted runs"]
    # released before the policy ever took authority (no policy episode): still an abort
    pre = [x for x in _ticks(REL, 2.0) if x["mode"] != "policy"]
    for j, x in enumerate(pre):
        x["tick"] = 1 + j
    g = _an(tmp_path, "l_pre", pre, "G")
    assert g["policy_exit"]["exit"] == "release_without_policy_episode"
    assert g["aborted_from_logs"] and g["safety_event"]


def test_case_l_release_in_a_live_ladder_stand_is_a_ladder_safety_event(tmp_path):
    rel = _run_dir(tmp_path, "lad_rel", _ticks(REL, 2.0))
    ok = _run_dir(tmp_path, "lad_ok", _ticks(END, 5.0145))
    ev = ga.ladder_events(
        [{"stage": "G", "run_dir": str(rel)}, {"stage": "G", "run_dir": str(ok)}], tmp_path
    )
    assert [e["policy_exit"] for e in ev] == ["deadman_release"] and ev[0]["arm"] == "B"


def test_classify_run_cli_exit_codes(tmp_path):
    ok = _run_dir(tmp_path, "cli_ok", _ticks(END, 5.0145))
    rel = _run_dir(tmp_path, "cli_rel", _ticks(REL, 2.0))
    run = lambda d, s: ga.main(["--classify-run", str(d), "--slot", s])  # noqa: E731
    assert run(ok, "G") == 0 and run(rel, "G") == 1
    assert run(rel, "stop_drill") == 0 and run(ok, "stop_drill") == 1


def test_runner_checks_the_logs_after_every_run():
    src = RUNNER.read_text()
    assert "--classify-run" in src and "log_check" in src
