"""Simulator rollouts as bridge-format tick records, so ONE monitor serves sim and robot.

Phase 1a of the experiment validates the monitor in Isaac Lab before any hardware run.
Rather than a second, sim-specific monitor, the simulator's per-step arrays are
written in the same tick-record schema the LowCmd bridge writes
(:mod:`phoenix.sim2real.bridge_telemetry`), and :func:`phoenix.monitor.layers.from_records`
reads both.

Mapping (policy steps at 50 Hz, arrays in Unitree motor order, shape ``(T, 12)``):

* ``requested``: ``default_q + action_scale * action`` before the rate limiter;
* ``sent``: the target the rate-limited action term applied (what the DCMotor PD
  law tracked during the step);
* ``q``/``dq``: joint state at the START of each step, as the bridge records the
  freshest LowState at the start of its tick;
* ``kp``/``kd``: the actuator's stiffness/damping for this env (per joint), which
  includes any targeted scaling, so the degradation is visible exactly as on the robot.

There is no policy-node layer in sim: the node target equals the request unless
``node_target`` is given. The adapter is pure numpy; calling it from
``phoenix.training.evaluate`` needs Isaac Lab and is not wired yet.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from phoenix.sim2real.motor_crc import unitree_to_phoenix

from .layers import N_JOINTS


def _rows(a: np.ndarray, name: str, t: int) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.broadcast_to(arr, (t, N_JOINTS))
    if arr.shape != (t, N_JOINTS):
        raise ValueError(f"{name}: expected shape ({t}, {N_JOINTS}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: non-finite values")
    return arr


def records_from_arrays(
    requested: np.ndarray,
    sent: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    control_dt: float = 0.02,
    node_target: np.ndarray | None = None,
    tau: np.ndarray | None = None,
) -> Iterator[dict]:
    """Yield a manifest record, then one ``policy``-mode tick record per step."""
    req = np.asarray(requested, dtype=np.float64)
    if req.ndim != 2:
        raise ValueError("requested must be (T, 12)")
    t = req.shape[0]
    req = _rows(req, "requested", t)
    snt = _rows(sent, "sent", t)
    qq = _rows(q, "q", t)
    dqq = _rows(dq, "dq", t)
    kpp = _rows(kp, "kp", t)
    kdd = _rows(kd, "kd", t)
    node = req if node_target is None else _rows(node_target, "node_target", t)
    tq = None if tau is None else _rows(tau, "tau", t)
    yield {"record": "manifest", "source": "simulation", "control_dt": float(control_dt)}
    for k in range(t):
        clipped = [bool(v) for v in np.abs(snt[k] - node[k]) > 0.0]
        yield {
            "record": "tick",
            "t_mono_ns": int(round(k * control_dt * 1e9)),
            "tick": k,
            "mode": "policy",
            "publish": True,
            "cmd_is_new": True,
            "q_unitree": qq[k].tolist(),
            "dq_unitree": dqq[k].tolist(),
            "tau_est_unitree": None if tq is None else tq[k].tolist(),
            "requested_target_unitree": node[k].tolist(),
            "final_target_unitree": snt[k].tolist(),
            "kp": float(np.max(kpp[k])),
            "kd": float(np.max(kdd[k])),
            "kp_unitree": kpp[k].tolist(),
            "kd_unitree": kdd[k].tolist(),
            "slew_clip": clipped,
            "limit_clip": [False] * N_JOINTS,
            "policy": {
                "requested_target": unitree_to_phoenix(req[k]),
                "target": unitree_to_phoenix(node[k]),
                "raw_action": None,
            },
        }


__all__ = ["records_from_arrays"]
