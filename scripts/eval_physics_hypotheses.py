"""Closed-loop MuJoCo replay of the hardware walking profiles under physics hypotheses (SIM ONLY).

For one deployed policy (deploy config: ONNX + manifest, kp 25 / kd 0.5 as on the robot),
every (hypothesis, profile) pair is simulated with phoenix.sim2sim.gate_runner, preceded by a
3 s zero-command stand like the harness. Per run it reports, with the SAME definitions as
phoenix.sim2real.walk_run_metrics on hardware:

* yaw change (quaternion) from 0.2 s before the first nonzero command to 1 s after the last,
* forward / lateral displacement from motion start to rest (profile end + 1 s),
* per joint, over [first nonzero command, last nonzero + 0.5 s]: swing ratio
  (p95-p5 of q over p95-p5 of the target sent) and RMS(target - q),
* falls, roll / pitch.

With ``--hardware`` (walk_run_metrics JSONs) every hypothesis is scored against the robot,
separately on a FIT set and a HELD-OUT set of profiles, so a hypothesis that matches one
trajectory is not mistaken for one that predicts another.

Hypothesis grammar (``;``-separated): ``name:key=val,key=val`` with keys
``scale`` (x measured Coulomb), ``vh``/``vt``/``vc`` (viscous hip/thigh/calf N m s/rad),
``load`` (k_load), ``soft`` (1 = MuJoCo default soft friction), ``rope`` (trunk yaw damping,
N m s/rad), ``ff`` (foot friction), ``lat`` (actuator latency ms). ``none`` = frictionless.

    ~/Sim/isaac-sim-venv/bin/python scripts/eval_physics_hypotheses.py \\
        --deploy-config results/friction_physics/I/deploy.yaml --grid ident --out-dir OUT \\
        --hardware HW1/metrics.json HW2/metrics.json HW3/metrics.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

GATE_CFG = REPO / "configs/sim2sim/gate_v3.yaml"
CTRL_HZ = 50
STAND_S = 3.0
TAIL_EXTRA_S = 1.0
#: Hardware-matched profiles + command magnitudes never run on the robot (generalisation).
PROFILES = {
    "vx03_4s": ("preset", "vx03_4s"),
    "yaw05_3s": ("preset", "yaw05_3s"),
    "vx03_2s": ("preset", "vx03_2s"),
    "vx05_3s": ("preset", "vx05_3s"),
    "yawm05_3s": ("custom", (0.0, 0.0, -0.5, 3.0)),
    "yaw08_3s": ("custom", (0.0, 0.0, 0.8, 3.0)),
    "vx02_yaw03_3s": ("custom", (0.2, 0.0, 0.3, 3.0)),
    "vx07_3s": ("custom", (0.7, 0.0, 0.0, 3.0)),
}
HW_PROFILES = ("vx03_4s", "yaw05_3s", "vx03_2s")
KEYS = {"scale", "vh", "vt", "vc", "load", "soft", "rope", "ff", "lat"}


def parse_hypothesis(text: str) -> tuple[str, dict[str, float]]:
    text = text.strip()
    name, _, body = text.partition(":")
    if not body:
        if name == "none":
            return "none", {}
        raise ValueError(f"hypothesis {text!r} needs name:key=val,...")
    out: dict[str, float] = {}
    for part in body.split(","):
        k, v = part.split("=")
        if k.strip() not in KEYS:
            raise ValueError(f"unknown key {k!r} in {text!r}")
        out[k.strip()] = float(v)
    return name, out


def friction_model(h: dict[str, float]):
    from phoenix.physics.joint_friction import FrictionModel

    if not any(k in h for k in ("scale", "vh", "vt", "vc", "load")):
        return None
    return FrictionModel.hypothesis(
        coulomb_scale=h.get("scale", 0.0),
        viscous_by_group={"hip": h.get("vh", 0.0), "thigh": h.get("vt", 0.0), "calf": h.get("vc", 0.0)},
        load_coeff=h.get("load", 0.0),
    )


def grid(name: str) -> list[str]:
    if name == "ident":
        out = ["none"]
        for s, v, k, rope in itertools.product((0.75, 1.0, 1.25, 1.5, 2.0), (0.0, 0.02, 0.05), (0.0, 0.05, 0.1), (0.0, 1.0)):
            out.append(f"s{s}_v{v}_k{k}_r{rope}:scale={s},vh={v},vt={v},vc={v},load={k},rope={rope}")
        out += [f"soft{s}:scale={s},soft=1" for s in (1.0, 1.25, 1.5)]
        out += [f"rope{r}:rope={r}" for r in (0.5, 1.0, 2.0, 4.0)]
        return out
    if name == "eval":
        return [
            "none",
            "meas1.0:scale=1.0",
            "meas1.25:scale=1.25",
            "meas1.5:scale=1.5",
            "meas1.25_visc:scale=1.25,vh=0.02,vt=0.02,vc=0.02",
            "meas1.0_load0.1:scale=1.0,load=0.1",
            "meas2.0:scale=2.0",
            "meas1.25_ff0.4:scale=1.25,ff=0.4",
            "meas1.25_lat20:scale=1.25,lat=20",
        ]
    raise ValueError(f"unknown grid {name!r}")


def build_scenario(profile: str, foot_friction: float | None):
    from phoenix.sim2real import walk_command as wc
    from phoenix.sim2sim.gate import GateScenario

    kind, val = PROFILES[profile]
    if kind == "preset":
        preset = wc.PRESETS[val]
    else:
        vx, vy, wz, motion_s = val
        preset = wc.WalkPreset(profile, vx, vy, wz, motion_s)
    dur = wc.profile_duration(preset)
    n = int(round((STAND_S + dur + TAIL_EXTRA_S) * CTRL_HZ))
    segs: list = []
    for k in range(n):
        t = k / CTRL_HZ
        tp = max(0.0, t - STAND_S)
        vx_, vy_, wz_, _ = wc.command_at(preset, tp)
        cmd = (float(vx_), float(vy_), float(wz_))
        if not segs or segs[-1][1] != cmd:
            segs.append((t, cmd))
    return GateScenario(f"replay_{profile}", "replay", n / CTRL_HZ, tuple(segs), foot_friction=foot_friction)


_CTX: dict = {}


def _init(deploy_cfg: str) -> None:
    import yaml

    from phoenix.sim2sim.deploy_spec import load_deploy_spec
    from phoenix.sim2sim.gate import load_gate_config
    from phoenix.sim2sim.gate_runner import OnnxPolicy

    dep = yaml.safe_load(Path(deploy_cfg).read_text())["policy"]
    cfg = load_gate_config(GATE_CFG)
    spec = load_deploy_spec(dep["manifest_path"], required_envelope=cfg.required_envelope)
    _CTX.update(cfg=cfg, spec=spec, policy=OnnxPolicy(dep["onnx_path"], spec.obs_dim))


def _yaw(qw: np.ndarray) -> np.ndarray:
    w, x, y, z = qw.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run_one(task: tuple[str, str, str]) -> dict:
    from phoenix.sim2sim.gate_runner import JOINT_ORDER, RunOptions, run_gate_scenario

    hname, htext, profile = task
    _, h = parse_hypothesis(htext)
    cfg, spec, policy = _CTX["cfg"], _CTX["spec"], _CTX["policy"]
    scn = build_scenario(profile, h.get("ff"))
    trace: list = []
    opts = RunOptions(
        latency_ms=float(h.get("lat", 0.0)),
        trace=trace,
        friction=friction_model(h),
        friction_stiff=not bool(h.get("soft", 0.0)),
        rope_yaw_damping=float(h.get("rope", 0.0)),
    )
    res = run_gate_scenario(spec, policy, scn, cfg, opts)
    m = res["metrics"]
    out: dict = {"hypothesis": hname, "spec": htext, "profile": profile, "fell": bool(m["fell"]), "fall_reason": m["fall_reason"]}
    if not trace:
        return out
    t = np.array([r["t"] for r in trace])
    cmd = np.array([r["cmd"] for r in trace])
    q = np.array([r["q"] for r in trace])
    tgt = np.array([r["target"] for r in trace])
    xy = np.array([r["base_xy"] for r in trace])
    yaw = _yaw(np.array([r["quat_wxyz"] for r in trace]))
    nz = np.flatnonzero(np.abs(cmd).sum(axis=1) > 1e-6)
    if nz.size == 0:
        return out
    a, b = max(nz[0] - 10, 0), min(nz[-1] + 50, len(t) - 1)
    k0 = max(nz[0] - 1, 0)
    kf = len(t) - 1
    h0 = yaw[k0]
    dx, dy = xy[kf] - xy[k0]
    win = (t >= t[nz[0]]) & (t <= t[nz[-1]] + 0.5)

    def swing(x: np.ndarray) -> np.ndarray:
        return np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0)

    sq, st = swing(q[win]), swing(tgt[win])
    rms = np.sqrt(np.mean((tgt[win] - q[win]) ** 2, axis=0))
    steady = (t > t[nz[0]] + 1.5) & (t <= t[nz[-1]] - 0.5)
    out.update(
        yaw_deg=math.degrees(float(yaw[b] - yaw[a])),
        fwd_m=float(dx * math.cos(h0) + dy * math.sin(h0)),
        lat_m=float(-dx * math.sin(h0) + dy * math.cos(h0)),
        cmd_yaw_deg=math.degrees(float(np.sum(cmd[a:b, 2]) / CTRL_HZ)),
        cmd_fwd_m=float(np.sum(cmd[:, 0]) / CTRL_HZ),
        steady_wz=float(np.mean([r["wz"] for r, s_ in zip(trace, steady, strict=True) if s_])) if steady.any() else None,
        max_abs_roll_deg=math.degrees(max(abs(r["roll"]) for r in trace)),
        mean_pitch_deg_motion=math.degrees(float(np.mean([r["pitch"] for r, w_ in zip(trace, win, strict=True) if w_]))),
        joints={n: {"swing_ratio": float(sq[i] / st[i]) if st[i] > 1e-6 else None, "rms": float(rms[i])}
                for i, n in enumerate(JOINT_ORDER)},
    )
    return out


def load_hardware(paths: list[Path]) -> dict[str, dict]:
    """profile -> averaged hardware metrics (body + per joint) over the runs of that profile."""
    by: dict[str, list[dict]] = {}
    for p in paths:
        m = json.loads(p.read_text())
        name = Path(m["run_dir"]).name
        prof = next(pr for pr in HW_PROFILES if name.startswith(f"W_{pr}_"))
        by.setdefault(prof, []).append(m)
    out = {}
    for prof, ms in by.items():
        joints = {}
        for n in ms[0]["motor_tracking"]["joints"]:
            joints[n] = {
                "swing_ratio": float(np.mean([x["motor_tracking"]["joints"][n]["swing_ratio"] for x in ms])),
                "rms": float(np.mean([x["motor_tracking"]["joints"][n]["rms_target_minus_q_rad"] for x in ms])),
            }
        b = [x["body_tracking"] for x in ms]
        out[prof] = {
            "n_runs": len(ms),
            "yaw_deg": float(np.mean([x["yaw_imu_quat_deg"] for x in b if x.get("yaw_imu_quat_deg") is not None])),
            "fwd_m": next((x["forward_measured_m"] for x in b if x.get("forward_measured_m") is not None), None),
            "joints": joints,
        }
    return out


def score(sim: dict, hw: dict) -> dict:
    """Body and joint discrepancies of one sim run against the hardware mean of its profile."""
    ratio_err = [abs(sim["joints"][n]["swing_ratio"] - hw["joints"][n]["swing_ratio"]) for n in hw["joints"]]
    rms_err = [abs(sim["joints"][n]["rms"] - hw["joints"][n]["rms"]) for n in hw["joints"]]
    out = {"joint_ratio_mae": float(np.mean(ratio_err)), "joint_rms_mae_rad": float(np.mean(rms_err))}
    if abs(sim.get("cmd_yaw_deg", 0.0)) > 1.0:
        out["yaw_err_frac"] = abs(sim["yaw_deg"] - hw["yaw_deg"]) / abs(sim["cmd_yaw_deg"])
    if hw.get("fwd_m") is not None and abs(sim.get("cmd_fwd_m", 0.0)) > 0.05:
        out["fwd_err_frac"] = abs(sim["fwd_m"] - hw["fwd_m"]) / abs(sim["cmd_fwd_m"])
    body = [v for k, v in out.items() if k.endswith("_frac")]
    out["body_err"] = float(np.mean(body)) if body else None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deploy-config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--grid", default=None, help="ident | eval")
    ap.add_argument("--hypotheses", default=None, help="';'-separated, overrides --grid")
    ap.add_argument("--profiles", default=",".join(HW_PROFILES))
    ap.add_argument("--hardware", type=Path, nargs="*", default=[])
    ap.add_argument("--fit", default="yaw05_3s", help="profiles used to rank hypotheses")
    ap.add_argument("--heldout", default="vx03_4s", help="profiles never used for ranking")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--tag", default="hyp")
    args = ap.parse_args(argv)
    hyps = [h for h in (args.hypotheses.split(";") if args.hypotheses else grid(args.grid or "eval")) if h.strip()]
    parsed = [(parse_hypothesis(h)[0], h) for h in hyps]
    profiles = [p for p in args.profiles.split(",") if p]
    tasks = [(n, h, p) for n, h in parsed for p in profiles]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(str(args.deploy_config),)) as pool:
        rows = pool.map(run_one, tasks, chunksize=1)
    hw = load_hardware(args.hardware) if args.hardware else {}
    fit = [p for p in args.fit.split(",") if p in hw]
    held = [p for p in args.heldout.split(",") if p in hw]
    summary = []
    for n, h in parsed:
        rs = {r["profile"]: r for r in rows if r["hypothesis"] == n}
        rec: dict = {"hypothesis": n, "spec": h, "fell": [p for p, r in rs.items() if r["fell"]]}
        for p, r in rs.items():
            rec[f"{p}.yaw_deg"] = r.get("yaw_deg")
            rec[f"{p}.fwd_m"] = r.get("fwd_m")
            if p in hw and "joints" in r:
                r["score"] = score(r, hw[p])
        for split, ps in (("fit", fit), ("heldout", held)):
            sc = [rs[p]["score"] for p in ps if p in rs and "score" in rs[p]]
            if sc:
                rec[f"{split}.body_err"] = float(np.mean([s["body_err"] for s in sc if s["body_err"] is not None] or [np.nan]))
                rec[f"{split}.joint_ratio_mae"] = float(np.mean([s["joint_ratio_mae"] for s in sc]))
                rec[f"{split}.joint_rms_mae_rad"] = float(np.mean([s["joint_rms_mae_rad"] for s in sc]))
        summary.append(rec)
    (args.out_dir / f"{args.tag}_runs.json").write_text(json.dumps(rows, indent=2))
    (args.out_dir / f"{args.tag}_hardware.json").write_text(json.dumps(hw, indent=2))
    (args.out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
    key = "fit.body_err" if fit else None
    if key:
        summary.sort(key=lambda r: (r.get(key) if r.get(key) is not None else 9, r.get("fit.joint_ratio_mae", 9)))
    cols = [c for c in ("fit.body_err", "fit.joint_ratio_mae", "fit.joint_rms_mae_rad", "heldout.body_err",
                        "heldout.joint_ratio_mae", "heldout.joint_rms_mae_rad") if any(c in r for r in summary)]
    lines = ["| hypothesis | " + " | ".join(f"{p} yaw/fwd" for p in profiles) + " | " + " | ".join(cols) + " | fell |",
             "|---" * (2 + len(profiles) + len(cols)) + "|"]
    for r in summary:
        cells = []
        for p in profiles:
            y, f = r.get(f"{p}.yaw_deg"), r.get(f"{p}.fwd_m")
            cells.append(f"{y:.1f} deg / {f:.2f} m" if y is not None and f is not None else "n/a")
        cells += [f"{r[c]:.3f}" if isinstance(r.get(c), float) else "-" for c in cols]
        lines.append(f"| {r['hypothesis']} | " + " | ".join(cells) + f" | {','.join(r['fell']) or 'no'} |")
    if hw:
        lines.append("")
        lines.append("hardware: " + "; ".join(
            f"{p}: yaw {v['yaw_deg']:.1f} deg, fwd {v['fwd_m'] if v['fwd_m'] is not None else 'n/a'} m (n={v['n_runs']})"
            for p, v in hw.items()))
    table = "\n".join(lines)
    (args.out_dir / f"{args.tag}_table.md").write_text(table + "\n")
    print(table)
    print(f"\n{len(tasks)} runs in {time.time() - t0:.0f} s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
