"""Directional walking diagnostics: binning, segmentation, and a synthetic asymmetric policy."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.monitor.walk_directional import bin_of, command_segments, directional_report


@pytest.mark.parametrize(
    ("vx", "name"),
    [(-0.9, "strong_backward"), (-0.4, "mild_backward"), (-0.01, "mild_backward"),
     (0.0, None), (0.01, "mild_forward"), (0.4, "mild_forward"), (0.41, "strong_forward")],
)
def test_bin_edges(vx, name):
    assert bin_of(vx) == name


def test_segments_split_at_resample_and_drop_settle():
    cmd = np.zeros((100, 1, 3), np.float32)
    cmd[:50, 0, 0] = 0.5
    cmd[50:, 0, 0] = -0.5
    segs = list(command_segments(cmd, np.ones((100, 1), bool), settle=10))
    assert segs == [(0, 10, 50), (0, 60, 100)]


def test_report_separates_a_backward_only_policy():
    t, n = 200, 4
    cmd = np.zeros((t, n, 3), np.float32)
    cmd[:, 0, 0], cmd[:, 1, 0], cmd[:, 2, 0], cmd[:, 3, 0] = 0.75, 0.2, -0.2, -0.75
    linv = np.zeros((t, n, 3), np.float32)
    linv[:, 2:, 0] = cmd[:, 2:, 0]  # tracks backward only
    z = {"cmd": cmd, "linv": linv, "angv": np.zeros_like(linv), "valid": np.ones((t, n), bool),
         "grav": np.tile(np.array([0, 0, -1.0], np.float32), (t, n, 1)),
         "raw": np.zeros((t, n, 12), np.float32), "qd1": np.zeros((t, n, 12), np.float32),
         "tau_c": np.zeros((t, n, 12), np.float32), "tau_a": np.zeros((t, n, 12), np.float32)}
    eps = [{"trunk_contact": False, "success": True} for _ in range(n)]
    rep = directional_report(z, eps, dt=0.02)["bins"]
    assert rep["strong_backward"]["success"] == 1.0 and rep["mild_backward"]["success"] == 1.0
    assert rep["strong_forward"]["success"] == 0.0
    assert rep["strong_forward"]["vx_achieved"] == 0.0
    assert rep["strong_forward"]["abs_vx_err"] == pytest.approx(0.75)
    assert rep["mild_forward"]["success"] == 1.0  # 0.2 m/s error is inside the 0.25 bound


def test_walk_primary_score_adds_tracking_to_the_stand_score():
    import numpy as np

    from phoenix.monitor.stand_metrics import walk_primary_score

    t, n = 200, 3
    cmd = np.zeros((t, n, 3), np.float32)
    cmd[:, :, 0] = 0.6
    linv = np.zeros((t, n, 3), np.float32)
    linv[:, 0, 0] = 0.6  # tracks
    linv[:, 1, 0] = 0.0  # stands still: only the settle window scores
    linv[:, 2, 0] = 0.6
    grav = np.tile(np.array([0, 0, -1.0], np.float32), (t, n, 1))
    grav[100:, 2] = np.array([0.9, 0, -0.44], np.float32)  # env 2 falls over at half time
    valid = np.ones((t, n), bool)
    s = walk_primary_score(grav=grav, cmd=cmd, linv=linv, valid=valid,
                           contact_term=np.zeros(n, bool), dt=0.02)
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(0.25, abs=0.01)  # only the 1 s settle window counts
    assert s[2] == pytest.approx(0.5, abs=0.01)
