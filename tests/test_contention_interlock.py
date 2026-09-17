"""Nothing else may share the robot during a Phoenix session.

``come-here.service`` autostarts on Jetson boot. On 2026-09-17 it was measured
starving the payload (132 topics against 109, ``/lowstate`` 349 Hz against the
field notes' 500) and it can command the robot, so it is simultaneously a
degraded measurement and a second uncommanded authority over the motors. The
audit that day recorded the interlock as NOT BUILT; these tests pin it.

The theme throughout: absent evidence FAILS. A gate that passes because the
probe did not run is worse than no gate, because it produces a signed-looking
ledger entry.
"""

from __future__ import annotations

import pytest

from phoenix.sim2real.hw_probe import active_units, build_parser
from phoenix.sim2real.preflight_eval import (
    COMPETING_NODE_SUBSTRINGS,
    COMPETING_SERVICES,
    LOWSTATE_STARVATION_FLOOR_HZ,
    contention_checks,
    verdict,
)

CLEAN = {
    "active_units": [],
    "ros_nodes": ["/phoenix_policy_node", "/lowcmd_bridge"],
    "lowstate_rate_hz": 500.0,
}


def failed(probe) -> list[str]:
    return [c.name for c in contention_checks(probe) if not c.ok]


def test_come_here_is_named_as_a_competitor():
    assert "come-here.service" in COMPETING_SERVICES


def test_a_clean_payload_passes_every_check():
    checks = contention_checks(CLEAN)
    assert checks, "the interlock produced no checks at all"
    assert all(c.ok for c in checks)
    assert all(c.gating for c in checks)


def test_the_verdict_helper_agrees_it_is_clean():
    assert verdict(contention_checks(CLEAN)) == "GO"


def test_an_empty_probe_is_a_no_go_verdict():
    assert verdict(contention_checks({})) == "NO-GO"


# ------------------------------------------------------- the actual refusal
def test_an_active_come_here_service_blocks():
    probe = dict(CLEAN, active_units=["come-here.service"])
    assert failed(probe) == ["none of ['come-here.service'] is active"]


def test_the_refusal_says_how_to_clear_it():
    probe = dict(CLEAN, active_units=["come-here.service"])
    detail = next(c.detail for c in contention_checks(probe) if not c.ok)
    assert "systemctl stop" in detail


def test_unrelated_active_units_do_not_block():
    probe = dict(CLEAN, active_units=["ssh.service", "nvargus-daemon.service"])
    assert failed(probe) == []


@pytest.mark.parametrize("name", ["/come_here_doa", "/come-here-node", "/ComeHere_driver"])
def test_a_competing_node_blocks_even_without_the_service(name):
    # Started by hand, or under a different unit, but the same authority over
    # the robot. The service check alone would miss it.
    probe = dict(CLEAN, active_units=[], ros_nodes=["/phoenix_policy_node", name])
    assert failed(probe) == ["no competing node in the ROS graph"]


def test_node_matching_is_case_insensitive():
    assert all(s == s.lower() for s in COMPETING_NODE_SUBSTRINGS)


def test_a_starved_lowstate_rate_blocks():
    probe = dict(CLEAN, lowstate_rate_hz=349.0)
    assert failed(probe) == [f"/lowstate at or above {LOWSTATE_STARVATION_FLOOR_HZ:g} Hz"]


def test_the_measured_starved_rate_is_below_the_floor():
    # 349 Hz is what was actually recorded with come-here running; if the floor
    # ever drifts below it this gate stops catching the thing it was built for.
    assert 349.0 < LOWSTATE_STARVATION_FLOOR_HZ <= 500.0


def test_a_healthy_rate_passes():
    assert failed(dict(CLEAN, lowstate_rate_hz=500.0)) == []


# ------------------------------------------------------- absent evidence fails
def test_an_empty_probe_fails_closed():
    names = failed({})
    assert "competing services were checked" in names
    assert "ROS graph was checked for competing nodes" in names


def test_missing_units_evidence_fails_even_with_clean_nodes():
    probe = {"ros_nodes": ["/phoenix_policy_node"]}
    assert failed(probe) == ["competing services were checked"]


def test_missing_node_evidence_fails_even_with_clean_units():
    probe = {"active_units": []}
    assert failed(probe) == ["ROS graph was checked for competing nodes"]


def test_an_omitted_rate_is_not_invented():
    # The rate is optional evidence: absent means "not measured", and the
    # interlock must not fabricate a passing or failing number for it.
    probe = {"active_units": [], "ros_nodes": ["/phoenix_policy_node"]}
    names = [c.name for c in contention_checks(probe)]
    assert not any("lowstate" in n for n in names)


# ---------------------------------------------------------------- the probe
def test_the_probe_has_a_contention_subcommand():
    args = build_parser().parse_args(["contention", "--out", "/tmp/c.json"])
    assert args.command == "contention"
    assert args.lowstate_rate_hz is None


def test_the_probe_requires_an_output_path():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["contention"])


def test_active_units_reports_nothing_for_an_unknown_unit():
    assert active_units(["phoenix-definitely-not-a-real-unit.service"]) == []


def test_active_units_raises_when_systemctl_is_missing(monkeypatch):
    # Reporting "nothing competing" because the check could not run would turn
    # a broken probe into a green gate.
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="systemctl not found"):
        active_units(["come-here.service"])
