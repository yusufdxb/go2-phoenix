"""Robot-side runner for the active joint-friction probe (hung GO2, feet off the ground).

The protocol (sequencer, constants, estimator) lives in
:mod:`phoenix.sim2real.friction_probe`; this module only runs it. It is a command
SOURCE for the existing LowCmd bridge, in place of the policy node:

* the bridge is live with a stand-only deploy config, no ``--standup-s``, and is
  holding measured posture with the real deadman (the harness ``FP`` stage);
* this node waits for fresh ``/joint_states`` and a deadman heartbeat, lets the
  bridge hold for ``settle_s``, measures ``q0`` once (median of the last 0.5 s), then
  publishes wire-v2 :data:`~phoenix.sim2real.command_wire.KIND_POLICY` targets at
  ``TICK_HZ`` on the deploy config's ``topics.joint_command``, from one
  ``ProbeSequencer`` per selected joint (so a joint subset is possible: the
  sequencer itself always walks all 12). Every bridge gate (torque clip, hard limits,
  sustained-clip latch, estop, deadman, freshness) applies unchanged;
* probe arrays are in Unitree motor order (``friction_probe.PROBE_JOINT_ORDER``); the
  wire is in the deploy config's ``joint_order`` (policy order). The node permutes.

The node's own abort rules (pure, in :func:`check_abort`): estop asserted or heartbeat
stale/never seen, ``/joint_states`` stale, ``|q - q0| > MAX_OFFSET + 0.05`` on any
joint, any joint other than the active and the previously probed one more than
0.05 rad from ``q0``, a non-finite state, or the wall-clock cap. On abort it sends a
few :data:`~phoenix.sim2real.command_wire.KIND_ABORT` notices (the bridge latches a
hold, then its own sit) and stops publishing, so the bridge's stale-command path is
the backstop if the notices are lost. A complete run sends the same notice with
reason ``authority_window_complete``.

``python -m phoenix.sim2real.friction_probe_node estimate --trace X.npz --out Y.json``
runs the estimator on a recorded trace and prints the per-joint table.

Pure logic (``check_abort``, :class:`ProbeSession`) imports no ROS and is what
``tests/test_friction_probe_node.py`` covers. ROS is imported lazily in :func:`run_node`.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phoenix.sim2real import friction_probe as fp
from phoenix.sim2real.command_wire import KIND_ABORT, KIND_POLICY, encode
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER

EXIT_COMPLETE = 0
EXIT_ABORTED = 2
EXIT_REFUSED = 3

ABORT_NOTICE_REPEATS = 3


class SingleJointSequencer(fp.ProbeSequencer):
    """``ProbeSequencer`` that stops after one joint (+ then - excursion).

    ASSUMPTION: starts the inner sequencer at ``joint_index`` by setting its ``_joint``
    (no public start-joint parameter exists); ``tests/test_friction_probe_node.py``
    pins that the ramp lands on the intended joint.
    """

    def __init__(self, q0: np.ndarray, joint_index: int, **kw: Any) -> None:
        super().__init__(q0, **kw)
        self.joint_index = int(joint_index)
        self._joint = self.joint_index

    @property
    def finished(self) -> bool:
        return self._phase == "done" or self._joint != self.joint_index


@dataclass
class Snapshot:
    """What the runner knows at one tick, in probe (Unitree motor) joint order."""

    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    lowstate_age_s: float
    estop: bool | None
    estop_age_s: float


@dataclass(frozen=True)
class AbortLimits:
    max_offset: float = fp.MAX_OFFSET_RAD
    excursion_margin: float = 0.05
    other_joint_tol: float = 0.05
    sensor_timeout_s: float = 0.2
    estop_timeout_s: float = 0.5
    wall_cap_s: float = 120.0


#: q0 is captured only when every joint moved less than this over the last 0.5 s.
CAPTURE_STILL_RAD = 0.005
#: Seconds past the settle period to wait for stillness before refusing to probe.
CAPTURE_TIMEOUT_S = 10.0


def check_abort(
    snap: Snapshot,
    q0: np.ndarray | None,
    exempt: Collection[int],
    limits: AbortLimits,
    elapsed_s: float,
    names: Sequence[str] = fp.PROBE_JOINT_ORDER,
    ref: np.ndarray | None = None,
) -> str | None:
    """First abort reason, or None. Pure. Reasons start with a wire abort name.

    ``exempt`` are the joints allowed to move (the active one and the one probed just
    before it, which may sit off q0 by stiction). ``q0`` is None before it is measured.
    The excursion cap is measured from ``q0`` (the commanded hold pose) at all times. The
    other-joint check is measured from ``ref``, the pose the legs settled to under the
    probe's own gains; it is skipped while ``ref`` is None (the first settle, when a
    hanging leg sags until kp * error balances its weight).
    """
    if snap.estop is None:
        return "estop_publisher_missing"
    if snap.estop:
        return "external_estop:deadman_released_or_estop_true"
    if not math.isfinite(snap.estop_age_s) or snap.estop_age_s > limits.estop_timeout_s:
        return "estop_heartbeat_stale"
    if not math.isfinite(snap.lowstate_age_s) or snap.lowstate_age_s > limits.sensor_timeout_s:
        return "sensor_stale:joint_states"
    if not (np.all(np.isfinite(snap.q)) and np.all(np.isfinite(snap.dq))):
        return "nan_in_joint_state"
    if elapsed_s > limits.wall_cap_s:
        return "max_runtime:wall_clock_cap"
    if q0 is not None:
        dev = np.abs(snap.q - q0)
        bad = np.flatnonzero(dev > limits.max_offset + limits.excursion_margin)
        if bad.size:
            return f"other:excursion:{names[int(bad[0])]}:{dev[bad[0]]:.3f}rad"
    if q0 is not None and ref is not None:
        dev_ref = np.abs(snap.q - ref)
        for i in range(len(dev_ref)):
            if i not in exempt and dev_ref[i] > limits.other_joint_tol:
                return f"other:other_joint_moved:{names[i]}:{dev_ref[i]:.3f}rad"
    return None


@dataclass
class StepResult:
    phase: str  # settle | settle_hold | rise | fall | rest | done | abort
    target: np.ndarray | None
    reason: str | None = None
    done: bool = False
    row: dict[str, Any] | None = None


class ProbeSession:
    """The probe timeline as a pure state machine: ``step(t, snapshot) -> StepResult``.

    Phase 0 (``settle``, ``settle_s``): no command is sent, the bridge holds; q0 is
    the median of the last 0.5 s of ``q``. Then one sequencer per joint in ``joints``,
    in order; each starts with its own ``SETTLE_S`` hold at q0. ``protocol`` "v1" is
    :class:`SingleJointSequencer` (one slow + and - excursion), "v2" is
    :class:`~phoenix.sim2real.friction_probe.RateProbeSequencer` (``repeats`` x slow/fast
    +/- with dwells). The abort rules are the same for both.
    """

    def __init__(
        self,
        joints: Sequence[str],
        limits: AbortLimits,
        settle_s: float = 2.0,
        order: Sequence[str] = fp.PROBE_JOINT_ORDER,
        protocol: str = "v1",
        repeats: int = fp.V2_REPEATS,
    ) -> None:
        if protocol not in ("v1", "v2"):
            raise ValueError(f"protocol {protocol!r} not in ('v1', 'v2')")
        if protocol == "v2" and repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        self.protocol = protocol
        self.repeats = int(repeats)
        if not joints or any(j not in order for j in joints):
            raise ValueError(f"joints {list(joints)} not all in {list(order)}")
        self.order = tuple(order)
        self.idx = [self.order.index(j) for j in joints]
        self.limits = limits
        self.settle_s = settle_s
        self.q0: np.ndarray | None = None
        self.n_clipped = 0
        self._seqs_done = 0
        self._seq: SingleJointSequencer | fp.RateProbeSequencer | None = None
        self._samples: list[tuple[float, np.ndarray]] = []
        self._prev_idx: int | None = None
        # Pose the legs settle to under the probe's gains; reference for other-joint motion.
        self.ref: np.ndarray | None = None
        self._t_cmd0: float | None = None
        self._post: list[tuple[float, np.ndarray]] = []

    @property
    def total_s(self) -> float:
        if self.protocol == "v2":
            per_joint = fp.v2_worst_case_joint_s(self.repeats)
        else:
            per_joint = (fp.worst_case_duration_s() - fp.SETTLE_S) / fp.N_JOINTS + fp.SETTLE_S
        return self.settle_s + len(self.idx) * per_joint

    def _start_seq(self) -> SingleJointSequencer | fp.RateProbeSequencer:
        assert self.q0 is not None
        if self.protocol == "v2":
            return fp.RateProbeSequencer(self.q0, self.idx[self._seqs_done], repeats=self.repeats)
        return SingleJointSequencer(self.q0, self.idx[self._seqs_done])

    def step(self, t_s: float, snap: Snapshot, t_mono_ns: int = 0) -> StepResult:
        exempt: set[int] = set()
        if self.q0 is not None and self._seq is not None:
            exempt = {self._seq.joint_index}
            if self._prev_idx is not None:
                exempt.add(self._prev_idx)
        if self.q0 is not None and self.ref is None and self._t_cmd0 is not None:
            # First settle under the probe's own gains: hanging legs sag until kp * error
            # balances their weight (10-07 hung run: FR_calf 0.058 rad). Take the
            # other-joint reference only after that, from the last 0.5 s of it.
            self._post.append((t_s, snap.q.copy()))
            if t_s >= self._t_cmd0 + fp.SETTLE_S - 0.05:
                tail = [q for ts, q in self._post if ts >= t_s - 0.5]
                self.ref = np.median(np.stack(tail), axis=0)
        reason = check_abort(snap, self.q0, exempt, self.limits, t_s, self.order, ref=self.ref)
        if reason is not None:
            return StepResult("abort", None, reason)
        if self.q0 is None:
            self._samples.append((t_s, snap.q.copy()))
            if t_s < self.settle_s:
                return StepResult("settle", None)
            # Capture q0 only once the legs are still: before the first command the
            # bridge re-targets the measured position every tick, so a sagging leg
            # would leave q0 lagging and get tugged back on the first command
            # (safety review of 09cce76). Wait up to CAPTURE_TIMEOUT_S, then refuse.
            recent = [q for ts, q in self._samples if ts >= t_s - 0.5]
            spread = float(np.max(np.ptp(np.stack(recent), axis=0))) if len(recent) > 1 else 1.0
            if spread > CAPTURE_STILL_RAD:
                if t_s >= self.settle_s + CAPTURE_TIMEOUT_S:
                    return StepResult("abort", None, f"not_still_at_capture:spread_{spread:.4f}rad")
                return StepResult("settle", None)
            self.q0 = np.median(np.stack(recent), axis=0)
            self._t_cmd0 = t_s
            self._seq = self._start_seq()
        assert self._seq is not None and self.q0 is not None
        target = self._seq.step(t_s, snap.q)
        if self._seq.finished:
            self._prev_idx = self._seq.joint_index
            self._seqs_done += 1
            if self._seqs_done >= len(self.idx):
                return StepResult("done", self.q0.copy(), done=True)
            self._seq = self._start_seq()
            target = self._seq.step(t_s, snap.q)
        status = self._seq.status
        row = {
            "record": "tick",
            "t_mono_ns": int(t_mono_ns),
            "t_s": float(t_s),
            "q": snap.q.tolist(),
            "dq": snap.dq.tolist(),
            "target": target.tolist(),
            "tau_est": snap.tau.tolist(),
            "kp": fp.KP,
            "kd": fp.KD,
            "active": self.order[self._seq.joint_index],
            "phase": status.phase,
            "offset": float(status.offset),
            "broke": bool(status.broke),
        }
        if self.protocol == "v2":
            row["rate"] = float(self._seq.rate)
            row["excursion"] = int(self._seq.excursion)
        return StepResult(status.phase, target, row=row)


def encode_command(
    wire_order: Sequence[str],
    probe_order: Sequence[str],
    *,
    seq: int,
    kind: int,
    target: Sequence[float],
    snap: Snapshot | None,
    elapsed_s: float,
    abort_reason: str | None = None,
) -> tuple[str, list[float]]:
    """One wire-v2 command, probe-order arrays permuted to the wire (policy) order.

    Fields the bridge gate needs for KIND_POLICY: finite target, finite obs_source_code,
    stand_only 1.0, zero velocity commands (any nonzero would latch ``walking_blocked``).
    """
    nan = float("nan")
    perm = [list(probe_order).index(n) for n in wire_order]
    tgt = np.asarray(target, dtype=np.float64)[perm]
    return encode(
        wire_order,
        seq=seq,
        kind=kind,
        target=tgt.tolist(),
        abort_reason=abort_reason,
        obs_source_code=0.0,  # "zeros": no base-velocity estimate is used or claimed
        stand_only=1.0,
        velocity_command_fed=[0.0, 0.0, 0.0],
        cmd_vel_received=[0.0, 0.0, 0.0],
        joint_state_age_s=nan if snap is None else snap.lowstate_age_s,
        estop_age_s=nan if snap is None else snap.estop_age_s,
        estop_value=nan if snap is None or snap.estop is None else float(snap.estop),
        policy_elapsed_s=elapsed_s,
        q_policy=[nan] * len(wire_order) if snap is None else snap.q[perm].tolist(),
    )


def write_npz(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    if not rows:
        return
    np.savez(
        path,
        t=np.asarray([r["t_s"] for r in rows]),
        t_mono_ns=np.asarray([r["t_mono_ns"] for r in rows], dtype=np.int64),
        q=np.asarray([r["q"] for r in rows]),
        dq=np.asarray([r["dq"] for r in rows]),
        target=np.asarray([r["target"] for r in rows]),
        tau_est=np.asarray([r["tau_est"] for r in rows]),
        kp=np.asarray([r["kp"] for r in rows]),
        kd=np.asarray([r["kd"] for r in rows]),
        active=np.asarray([r["active"] for r in rows]),
        phase=np.asarray([r["phase"] for r in rows]),
        offset=np.asarray([r["offset"] for r in rows]),
        broke=np.asarray([r["broke"] for r in rows]),
        rate=np.asarray([r.get("rate", fp.RATE_RAD_S) for r in rows]),
        excursion=np.asarray([r.get("excursion", -1) for r in rows], dtype=np.int64),
        protocol=np.asarray(meta.get("protocol", "v1")),
        q0=np.asarray(meta["q0"]),
        joint_order=np.asarray(meta["probe_order"]),
    )


def load_trace_npz(path: Path) -> fp.ProbeTrace:
    z = np.load(path)
    return fp.ProbeTrace(
        t=z["t"],
        q=z["q"],
        target=z["target"],
        dq=z["dq"],
        kp=z["kp"][:, None],
        kd=z["kd"][:, None],
        tau_est=z["tau_est"],
    )


def estimate_main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(prog="friction_probe_node estimate")
    p.add_argument("--trace", type=Path, required=True, help="the .npz written by the node")
    p.add_argument("--out", type=Path, required=True, help="estimate JSON, never overwritten")
    a = p.parse_args(list(argv))
    est = fp.estimate(load_trace_npz(a.trace))
    with a.out.open("x") as fh:
        json.dump(est.to_dict(), fh, indent=1)

    def f(v: float | None) -> str:
        return "-" if v is None else f"{v:7.3f}"

    print(f"{'joint':<16}{'Fs Nm':>8}{'bias Nm':>9}{'kin Nm':>8}{'Fs>':>8}  flags")
    for je in est.joints:
        print(
            f"{je.name:<16}{f(je.fs_nm):>8}{f(je.bias_nm):>9}{f(je.kinetic_nm):>8}"
            f"{f(je.fs_lower_bound_nm):>8}  {','.join(je.flags) or '-'}"
        )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--config", type=Path, required=True, help="the STAND-ONLY deploy config")
    p.add_argument("--out", type=Path, required=True, help="trace JSONL path (never overwritten)")
    p.add_argument("--joints", nargs="*", default=None, help="joint names, default all 12")
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument("--max-wall-s", type=float, default=None, help="default: whole timeline + 10 s")
    p.add_argument(
        "--protocol",
        choices=("v1", "v2"),
        default="v1",
        help="v1: one slow +/- excursion per joint; v2: repeats x slow/fast +/- with dwells",
    )
    p.add_argument(
        "--repeats", type=int, default=fp.V2_REPEATS, help="v2 only: schedule repeats per joint"
    )
    return p.parse_args(argv)


def run_node(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    import yaml

    from phoenix.sim2real.deploy_contract import is_stand_only

    cfg = yaml.safe_load(args.config.read_text())
    if not is_stand_only(cfg):
        print(
            f"REFUSING: {args.config} is not stand-only (safety.stand_only: true)", file=sys.stderr
        )
        return EXIT_REFUSED
    if args.out.exists():
        print(f"REFUSING: {args.out} exists; a trace is never overwritten", file=sys.stderr)
        return EXIT_REFUSED
    probe_order = tuple(fp.PROBE_JOINT_ORDER)
    wire_order = tuple(cfg.get("joint_order") or POLICY_JOINT_ORDER)
    if sorted(wire_order) != sorted(probe_order):
        print("REFUSING: deploy joint_order is not the 12 GO2 joints", file=sys.stderr)
        return EXIT_REFUSED
    joints = list(args.joints) if args.joints else list(probe_order)
    safety = cfg.get("safety") or {}
    topics = cfg["topics"]
    first_msg_s = float(safety.get("first_message_timeout_s", 15.0))
    limits = AbortLimits(
        sensor_timeout_s=float(safety.get("sensor_timeout_s", 0.2)),
        estop_timeout_s=float(safety.get("estop_timeout_s", 0.5)),
    )
    try:
        session = ProbeSession(
            joints, limits, settle_s=args.settle_s, protocol=args.protocol, repeats=args.repeats
        )
    except ValueError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    wall_cap = args.max_wall_s if args.max_wall_s is not None else session.total_s + 10.0
    session.limits = AbortLimits(
        sensor_timeout_s=limits.sensor_timeout_s,
        estop_timeout_s=limits.estop_timeout_s,
        wall_cap_s=wall_cap,
    )

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension

    stop = {"signal": False}

    def _on_signal(_signum: int, _frame: Any) -> None:
        stop["signal"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    rclpy.init(args=None, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = Node("phoenix_friction_probe")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    latest: dict[str, Any] = {
        "q": None,
        "dq": None,
        "tau": None,
        "js_ns": None,
        "estop": None,
        "estop_ns": None,
    }

    def on_js(msg: Any) -> None:
        pos = dict(zip(msg.name, msg.position, strict=False))
        if not all(n in pos for n in probe_order):
            return
        vel = dict(zip(msg.name, msg.velocity, strict=False))
        eff = dict(zip(msg.name, msg.effort, strict=False))
        latest["q"] = np.asarray([pos[n] for n in probe_order], dtype=np.float64)
        latest["dq"] = np.asarray([vel.get(n, math.nan) for n in probe_order], dtype=np.float64)
        latest["tau"] = np.asarray([eff.get(n, math.nan) for n in probe_order], dtype=np.float64)
        latest["js_ns"] = time.monotonic_ns()

    def on_estop(msg: Any) -> None:
        latest["estop"] = bool(msg.data)
        latest["estop_ns"] = time.monotonic_ns()

    node.create_subscription(JointState, topics["joint_states"], on_js, qos)
    node.create_subscription(Bool, safety["emergency_stop_topic"], on_estop, qos)
    pub = node.create_publisher(Float64MultiArray, topics["joint_command"], qos)
    seq = {"n": 0}

    def publish(
        kind: int,
        target: Sequence[float],
        snap: Snapshot | None,
        elapsed: float,
        reason: str | None = None,
    ) -> None:
        label, data = encode_command(
            wire_order,
            probe_order,
            seq=seq["n"],
            kind=kind,
            target=target,
            snap=snap,
            elapsed_s=elapsed,
            abort_reason=reason,
        )
        seq["n"] += 1
        msg = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.label, dim.size, dim.stride = label, len(data), len(data)
        msg.layout.dim = [dim]
        msg.data = data
        pub.publish(msg)

    def snapshot() -> Snapshot | None:
        if latest["q"] is None:
            return None
        now = time.monotonic_ns()
        return Snapshot(
            q=latest["q"],
            dq=latest["dq"],
            tau=latest["tau"],
            lowstate_age_s=(now - latest["js_ns"]) / 1e9,
            estop=latest["estop"],
            estop_age_s=(
                math.inf if latest["estop_ns"] is None else (now - latest["estop_ns"]) / 1e9
            ),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fh = args.out.open("x")
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "record": "header",
        "probe_order": list(probe_order),
        "wire_order": list(wire_order),
        "joints_probed": joints,
        "kp": fp.KP,
        "kd": fp.KD,
        "tick_hz": fp.TICK_HZ,
        "rate_rad_s": fp.RATE_RAD_S,
        "protocol": args.protocol,
        "repeats": args.repeats if args.protocol == "v2" else None,
        "v2_schedule": [list(e) for e in fp.v2_schedule(args.repeats)]
        if args.protocol == "v2"
        else None,
        "dwell_peak_s": fp.DWELL_PEAK_S if args.protocol == "v2" else None,
        "dwell_zero_s": fp.DWELL_ZERO_S if args.protocol == "v2" else None,
        "break_delta_rad": fp.BREAK_DELTA_RAD,
        "max_offset_rad": fp.MAX_OFFSET_RAD,
        "settle_s": args.settle_s,
        "wall_cap_s": wall_cap,
        "config": str(args.config),
        "q0": None,
        "t_start_mono_ns": time.monotonic_ns(),
    }
    fh.write(json.dumps(meta) + "\n")  # first record; q0 is in the summary and the .header.json
    outcome = {"v": "aborted", "reason": "startup"}
    print(
        f"[fp] waiting up to {first_msg_s:.0f} s for /joint_states and the deadman heartbeat",
        flush=True,
    )
    deadline = time.monotonic() + first_msg_s
    while time.monotonic() < deadline and not stop["signal"]:
        rclpy.spin_once(node, timeout_sec=0.05)
        if latest["q"] is not None and latest["estop"] is not None:
            break
    state: dict[str, Any] = {"t0": None, "done": False}

    def tick() -> None:
        if state["done"]:
            return
        now_ns = time.monotonic_ns()
        if state["t0"] is None:
            state["t0"] = now_ns
        elapsed = (now_ns - state["t0"]) / 1e9
        snap = snapshot()
        if stop["signal"]:
            res = StepResult("abort", None, "node_shutdown:signal")
        elif snap is None:
            res = StepResult("abort", None, "sensor_stale:no_joint_states")
        else:
            res = session.step(elapsed, snap, now_ns)
        if res.row is not None:
            rows.append(res.row)
            fh.write(json.dumps(res.row) + "\n")
            fh.flush()
            if meta["q0"] is None and session.q0 is not None:
                meta["q0"] = session.q0.tolist()
        if res.done or res.phase == "abort":
            state["done"] = True
            reason = "authority_window_complete" if res.done else (res.reason or "other")
            if session.q0 is not None:
                final = session.q0
            else:
                final = snap.q if snap is not None else np.zeros(len(probe_order))
            for _ in range(ABORT_NOTICE_REPEATS):
                publish(KIND_ABORT, final.tolist(), snap, elapsed, reason)
            outcome["v"] = "complete" if res.done else "aborted"
            outcome["reason"] = reason
            return
        if res.target is not None:
            publish(KIND_POLICY, res.target.tolist(), snap, elapsed)

    timer = node.create_timer(1.0 / fp.TICK_HZ, tick)
    print(f"[fp] probing {joints} (settle {args.settle_s} s, cap {wall_cap:.0f} s)", flush=True)
    try:
        if latest["q"] is None or latest["estop"] is None:
            print(
                "[fp] no joint states or no deadman heartbeat in time; nothing published",
                flush=True,
            )
            outcome["reason"] = "first_message_timeout"
        else:
            while rclpy.ok() and not state["done"]:
                rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        timer.cancel()
        summary = {
            "record": "summary",
            "outcome": outcome["v"],
            "reason": outcome["reason"],
            "n_ticks": len(rows),
            "t_end_mono_ns": time.monotonic_ns(),
            "q0": meta["q0"],
        }
        fh.write(json.dumps(summary) + "\n")
        fh.close()
        args.out.with_suffix(".header.json").write_text(json.dumps(meta, indent=1))
        write_npz(args.out.with_suffix(".npz"), rows, meta)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(
        f"[fp] {outcome['v']}: {outcome['reason']} ({len(rows)} ticks) trace {args.out}", flush=True
    )
    return EXIT_COMPLETE if outcome["v"] == "complete" else EXIT_ABORTED


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "estimate":
        return estimate_main(argv[1:])
    return run_node(argv)


if __name__ == "__main__":
    sys.exit(main())
