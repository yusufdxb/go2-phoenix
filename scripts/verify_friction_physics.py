"""Runtime check that the joint-friction model reaches the MuJoCo dynamics (SIM ONLY).

Base-fixed GO2 (feet in the air, contacts off), the model from
phoenix.physics.joint_friction applied through GateSim's MujocoFriction. Each check drives
every leg joint with a PD ramp up then down over the same span and estimates the friction
from the applied torque while moving (gravity cancels between the two legs of the ramp):

    F_hat(v) = (mean tau_up - mean tau_down) / 2

* coulomb:  measured x1.0, ramp 0.05 rad/s           -> F_hat = F0_j
* viscous:  b per group, ramps 0.5/1/2 rad/s          -> slope of F_hat over v = b
* load:     k_load with a constant external joint load -> F_hat = F0_j + k * mean|tau|

Run with ``--soft`` too: MuJoCo's default soft frictionloss creeps and reads low.

    ~/Sim/isaac-sim-venv/bin/python scripts/verify_friction_physics.py --out results/friction_physics/mujoco_verify.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phoenix.physics.joint_friction import FrictionModel, MujocoFriction, load_measured  # noqa: E402
from phoenix.sim2real.go2_model import TRAINING_DEFAULT_JOINT_POS  # noqa: E402
from phoenix.sim2sim.model import PROFILE_REAL, load_go2_model  # noqa: E402

PHYSICS_HZ = 1000
SPAN = 0.30
VISCOUS = {"hip": 0.05, "thigh": 0.08, "calf": 0.10}
LOAD_K = 0.10
LOAD_NM = 4.0
TOL_ABS, TOL_REL = 0.05, 0.15


class Rig:
    def __init__(self, fm: FrictionModel, stiff: bool) -> None:
        import mujoco

        self.mj = mujoco
        self.m, imap, _ = load_go2_model(PROFILE_REAL, timestep=1.0 / PHYSICS_HZ)
        self.m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        b = imap.base_dof_adr
        self.m.dof_armature[b : b + 6] = 1.0e6
        self.names = list(imap.joint_names)
        self.qadr = np.asarray(imap.qpos_adr)
        self.dadr = np.asarray(imap.dof_adr)
        self.bq, self.bd = imap.base_qpos_adr, imap.base_dof_adr
        self.fric = MujocoFriction(fm, self.m, self.dadr, self.names, stiff=stiff)
        self.d = mujoco.MjData(self.m)

    def _fix(self) -> None:
        self.d.qpos[self.bq : self.bq + 7] = (0, 0, 1.0, 1, 0, 0, 0)
        self.d.qvel[self.bd : self.bd + 6] = 0.0

    def ramp(self, speed: float, kp: float, kd: float, ext: np.ndarray) -> np.ndarray:
        """F_hat per joint for one up/down ramp at ``speed`` rad/s with external load ``ext``.

        Also sets ``self.mean_abs_tau``: mean |actuator torque| over the moving samples of both
        legs (the load the load-dependent term sees, gravity included)."""
        d, m = self.d, self.m
        self.mj.mj_resetData(m, d)
        q0 = np.array([TRAINING_DEFAULT_JOINT_POS[n] for n in self.names])
        start = q0 - SPAN / 2
        d.qpos[self.qadr] = start
        self._fix()
        self.mj.mj_forward(m, d)
        dt = m.opt.timestep
        n = int(SPAN / speed / dt)
        settle = int(0.5 / dt)
        traj = [start + k * speed * dt for k in range(n)]
        sums = {"up": [], "down": []}

        def step(target: np.ndarray, rec: str | None) -> None:
            q, qd = d.qpos[self.qadr], d.qvel[self.dadr]
            tau = kp * (target - q) - kd * qd
            self.fric.update(tau)
            d.qfrc_applied[self.dadr] = tau + ext
            self.mj.mj_step(m, d)
            self._fix()
            if rec:
                sign = 1.0 if rec == "up" else -1.0
                moving = sign * d.qvel[self.dadr] > 0.5 * speed
                sums[rec].append(np.where(moving, tau, np.nan))

        for _ in range(settle):
            step(traj[0], None)
        skip = int(0.25 * n)  # skip the breakaway transient at the start of each leg
        for k, t in enumerate(traj):
            step(t, "up" if k >= skip else None)
        for _ in range(settle):
            step(traj[-1], None)
        for k, t in enumerate(reversed(traj)):
            step(t, "down" if k >= skip else None)
        up = np.nanmean(np.asarray(sums["up"]), axis=0)
        dn = np.nanmean(np.asarray(sums["down"]), axis=0)
        self.mean_abs_tau = (
            np.nanmean(np.abs(np.asarray(sums["up"])), axis=0) + np.nanmean(np.abs(np.asarray(sums["down"])), axis=0)
        ) / 2.0
        return (up - dn) / 2.0


def ok(est: float, truth: float) -> bool:
    return bool(abs(est - truth) <= max(TOL_ABS, TOL_REL * abs(truth)))


def run(stiff: bool) -> dict:
    meas = load_measured()
    out: dict = {"stiff": stiff}
    # 1. Coulomb at probe speed
    fm = FrictionModel.hypothesis(1.0, label="measured x1.0")
    rig = Rig(fm, stiff)
    est = rig.ramp(0.05, 25.0, 0.5, np.zeros(12))
    rows = [{"joint": n, "set": meas[n]["static"], "est": float(e), "ok": ok(float(e), meas[n]["static"])}
            for n, e in zip(rig.names, est, strict=True)]
    out["coulomb"] = {"rows": rows, "pass": all(r["ok"] for r in rows)}
    # 2. Viscous: slope over speed, Coulomb on as well (the realistic case)
    fm = FrictionModel.hypothesis(1.0, VISCOUS, label="measured + viscous")
    rig = Rig(fm, stiff)
    speeds = (0.5, 1.0, 2.0)
    ests = np.stack([rig.ramp(v, 150.0, 2.0, np.zeros(12)) for v in speeds])
    rows = []
    for j, n in enumerate(rig.names):
        slope, icpt = np.polyfit(speeds, ests[:, j], 1)
        b = VISCOUS[n.split("_")[1]]
        rows.append({"joint": n, "b_set": b, "b_est": float(slope), "intercept": float(icpt),
                     "coulomb_set": meas[n]["static"], "ok": ok(float(slope), b)})
    out["viscous"] = {"speeds": speeds, "rows": rows, "pass": all(r["ok"] for r in rows)}
    # 3. Load: k * |tau| with a constant external load the actuator has to hold
    fm = FrictionModel.hypothesis(1.0, load_coeff=LOAD_K, label="measured + load")
    rig = Rig(fm, stiff)
    est = rig.ramp(0.05, 25.0, 0.5, np.full(12, LOAD_NM))
    est0 = Rig(FrictionModel.hypothesis(1.0), stiff).ramp(0.05, 25.0, 0.5, np.full(12, LOAD_NM))
    rows = []
    for j, n in enumerate(rig.names):
        truth = meas[n]["static"] + LOAD_K * float(rig.mean_abs_tau[j])
        rows.append({"joint": n, "set": truth, "mean_abs_tau": float(rig.mean_abs_tau[j]), "est": float(est[j]), "est_without_load_term": float(est0[j]),
                     "ok": ok(float(est[j]), truth)})
    out["load"] = {"k": LOAD_K, "external_nm": LOAD_NM, "rows": rows, "pass": all(r["ok"] for r in rows)}
    out["pass"] = out["coulomb"]["pass"] and out["viscous"]["pass"] and out["load"]["pass"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = {"stiff": run(True), "soft": run(False)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    for mode, r in res.items():
        print(f"== {mode}: coulomb {r['coulomb']['pass']} viscous {r['viscous']['pass']} load {r['load']['pass']}")
        for x in r["coulomb"]["rows"]:
            print(f"  coulomb {x['joint']:15s} set {x['set']:.3f} est {x['est']:.3f} {'ok' if x['ok'] else 'MISS'}")
        for x in r["viscous"]["rows"]:
            print(f"  viscous {x['joint']:15s} b set {x['b_set']:.3f} est {x['b_est']:.3f} (icpt {x['intercept']:.3f}) {'ok' if x['ok'] else 'MISS'}")
        for x in r["load"]["rows"]:
            print(f"  load    {x['joint']:15s} set {x['set']:.3f} est {x['est']:.3f} (no k: {x['est_without_load_term']:.3f}) {'ok' if x['ok'] else 'MISS'}")
    print("VERDICT", "PASS" if res["stiff"]["pass"] else "FAIL", "(stiff, the setting the evaluator uses)")
    return 0 if res["stiff"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
