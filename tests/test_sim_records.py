"""Sim arrays -> bridge-format records -> the same monitor (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest

from phoenix.monitor.layers import from_records, tracking_pairs
from phoenix.monitor.sim_records import records_from_arrays

T = 20


def arrays(clip_joint=None):
    rng = np.random.default_rng(0)
    req = rng.normal(0.0, 0.1, (T, 12))
    sent = req.copy()
    if clip_joint is not None:
        sent[:, clip_joint] += 0.05
    q = sent - 0.01
    kp = np.full(12, 25.0)
    kp[7] = 15.0  # a targeted, degraded RR_thigh
    return req, sent, q, np.zeros((T, 12)), kp, np.full(12, 0.5)


def test_round_trip_keeps_layers_and_per_joint_gains():
    req, sent, q, dq, kp, kd = arrays()
    lay = from_records(records_from_arrays(req, sent, q, dq, kp, kd))
    assert len(lay) == T
    assert np.allclose(lay.requested, req) and np.allclose(lay.sent, sent)
    assert np.allclose(lay.kp[:, 7], 15.0) and np.allclose(lay.kp[:, 0], 25.0)
    assert tracking_pairs(lay).valid.all()


def test_rate_limited_joint_is_excluded():
    req, sent, q, dq, kp, kd = arrays(clip_joint=3)
    pairs = tracking_pairs(from_records(records_from_arrays(req, sent, q, dq, kp, kd)))
    assert not pairs.valid[:, 3].any() and pairs.valid[:, 4].all()


def test_bad_shapes_and_nan_refused():
    req, sent, q, dq, kp, kd = arrays()
    with pytest.raises(ValueError):
        list(records_from_arrays(req, sent[:-1], q, dq, kp, kd))
    q[0, 0] = np.nan
    with pytest.raises(ValueError):
        list(records_from_arrays(req, sent, q, dq, kp, kd))
