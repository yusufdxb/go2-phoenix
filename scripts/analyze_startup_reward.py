"""Reward accounting for the first seconds after command onset from a still stance.

Question (2026-09-26): walk_vnext / walk_vnext2 policies refuse to START at low
speed (|v| 0.2-0.3, worst backward) from a still stance, and the refusal grows
with training. Does the reward make "stay still" competitive with "start"
during the start-up transient, and which SINGLE reward change most increases
the advantage of starting?

Protocol (Isaac, same env/spec as the checkpoint's training, pushes off):
  * every env: exact default stance, zero joint and root velocity
    (isaac_terms.write_cold_state), zero command for SETTLE_S, then the case's
    command pinned for POST_S; case ``moving_0.5_to_0.3`` walks 0.5 from cold
    for 4 s, and its onset is the switch to 0.3;
  * per env, per 50 Hz step, for WINDOW_S after onset: every reward term's RAW
    value (reward_manager._step_reward / weight) and the squared lin / yaw
    tracking errors (so the tracking kernel can be recomputed for any std);
  * outcome per env, from the mean velocity on the commanded axis over
    [WINDOW_S, POST_S] after onset: started (>= 0.5 x command), stationary
    (|mean| <= 0.1 x |command|), partial (neither, excluded), fell (excluded).

Offline, on those exact trajectories (the policy's actions do NOT change):
  R = sum_t sum_i w_i * raw_i(t) * dt  over the first WINDOW_S,
recomputed under ONE change at a time: action_l2 weight in ACTION_L2_WEIGHTS,
or tracking std (both tracking terms) in TRACKING_STDS.
advantage = mean R(started) - mean R(stationary), same command, pooled over
checkpoints. Also reported with a 0.99-per-step discount.
Limitation: a static counterfactual on fixed trajectories, not what a retrained
policy would do.

    PYTHONPATH=src python scripts/analyze_startup_reward.py --checkpoints A.pt B.pt --out out.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

SETTLE_S = 1.0
WINDOW_S = 2.0
POST_S = 4.0
MOVING_LEAD_S = 4.0
GAMMA = 0.99
ACTION_L2_WEIGHTS = (-0.01, -0.005, -0.0025, 0.0)
TRACKING_STDS = (0.5, 0.4, 0.35)
CASES = {
    "vx_p0.3": (0.3, 0.0, 0.0),
    "vx_m0.3": (-0.3, 0.0, 0.0),
    "vx_p0.2": (0.2, 0.0, 0.0),
    "vx_m0.2": (-0.2, 0.0, 0.0),
    "moving_0.5_to_0.3": (0.3, 0.0, 0.0),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless).app
    from phoenix.sim_app_exit import run_isaac_main

    return run_isaac_main(lambda: _run(args), app, label="analyze_startup_reward")


def _load_actor(ckpt_path: Path, device: str):
    import torch

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["actor_state_dict"]
    names = sorted(
        {k.rsplit(".", 1)[0] for k in sd if k.startswith("mlp.") and k.endswith(".weight")},
        key=lambda k: int(k.split(".")[1]),
    )
    layers = []
    for i, k in enumerate(names):
        w, b = sd[k + ".weight"], sd[k + ".bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0]).to(device)
        lin.weight.data, lin.bias.data = w, b
        layers += [lin] + ([torch.nn.ELU()] if i < len(names) - 1 else [])
    return torch.nn.Sequential(*layers)


def _run(args) -> int:
    import gymnasium as gym
    import torch

    from phoenix.velocity import env_cfg as pv
    from phoenix.velocity.isaac_terms import write_cold_state
    from phoenix.velocity.mdp import as_torch
    from phoenix.velocity.spec import TASK_ID, spec_from_dict

    pv.register()
    specs = {c: spec_from_dict(json.loads((c.parent / "spec.json").read_text())) for c in args.checkpoints}
    base = next(iter(specs.values()))
    for c, s in specs.items():
        if s.reward_scales() != base.reward_scales() or s.obs_schema != base.obs_schema:
            raise SystemExit(f"{c}: rewards/obs differ from {args.checkpoints[0]}; run separately")
    spec = base.replace(sim=dataclasses.replace(base.sim, num_envs=args.num_envs))
    cfg = pv.build_velocity_env_cfg(spec)
    cfg.events.push_robot = None
    env = gym.make(TASK_ID, cfg=cfg)
    u = env.unwrapped
    dev, n, dt = u.device, u.num_envs, u.step_dt
    robot = u.scene["robot"]
    cmd_term = u.command_manager.get_term("base_velocity")
    rm = u.reward_manager
    names = list(rm.active_terms)
    weights = torch.tensor([rm.get_term_cfg(t).weight for t in names], device=dev)
    all_ids = torch.arange(n, device=dev)
    win = int(round(WINDOW_S / dt))
    post = int(round(POST_S / dt))

    def pin(c):
        cmd_term.is_standing_env[:] = False
        cmd_term.time_left[:] = 1e9
        if hasattr(cmd_term, "cold_pending"):
            cmd_term.cold_pending[:] = False
        cmd_term.vel_command_b[:] = torch.tensor(c, device=dev)

    records = []
    for ck in args.checkpoints:
        actor = _load_actor(ck, dev)
        for case, cmd in CASES.items():
            obs, _ = u.reset()
            write_cold_state(u, all_ids)
            u.scene.write_data_to_sim()
            u.sim.forward()
            u.scene.update(dt=0.0)
            obs = u.observation_manager.compute()
            lead = int(round((SETTLE_S + (MOVING_LEAD_S if case.startswith("moving") else 0.0)) / dt))
            raw = torch.zeros(n, win, len(names), device=dev)
            lin_e2 = torch.zeros(n, win, device=dev)
            yaw_e2 = torch.zeros(n, win, device=dev)
            axis_v = torch.zeros(n, device=dev)
            fell = torch.zeros(n, dtype=torch.bool, device=dev)
            axis = max(range(3), key=lambda i: abs(cmd[i]))
            for k in range(lead + post):
                if k < int(round(SETTLE_S / dt)):
                    c = (0.0, 0.0, 0.0)
                elif k < lead:
                    c = (0.5, 0.0, 0.0)
                else:
                    c = cmd
                pin(c)
                with torch.no_grad():
                    a = actor(obs["policy"]).clamp(-spec.action.clip_actions, spec.action.clip_actions)
                obs, _r, term, trunc, _e = u.step(a)
                pin(c)
                j = k - lead  # step index after onset (reward of this step)
                if j >= 0:
                    fell |= term.bool() & ~trunc.bool()
                    vb = as_torch(robot.data.root_lin_vel_b)
                    wb = as_torch(robot.data.root_ang_vel_b)
                    ct = torch.tensor(c, device=dev)
                    if j < win:
                        raw[:, j] = rm._step_reward / weights
                        lin_e2[:, j] = ((ct[:2] - vb[:, :2]) ** 2).sum(1)
                        yaw_e2[:, j] = (ct[2] - wb[:, 2]) ** 2
                    elif j < post:
                        axis_v += torch.stack([vb[:, 0], vb[:, 1], wb[:, 2]], 1)[:, axis]
            axis_v /= max(post - win, 1)
            ratio = axis_v / cmd[axis]
            outcome = []
            for e in range(n):
                if bool(fell[e]):
                    outcome.append("fell")
                elif ratio[e] >= 0.5:
                    outcome.append("started")
                elif abs(float(ratio[e])) <= 0.1:
                    outcome.append("stationary")
                else:
                    outcome.append("partial")
            records.append(
                {
                    "checkpoint": str(ck),
                    "case": case,
                    "raw": raw.cpu(),
                    "lin_e2": lin_e2.cpu(),
                    "yaw_e2": yaw_e2.cpu(),
                    "outcome": outcome,
                }
            )
            counts = {o: outcome.count(o) for o in ("started", "stationary", "partial", "fell")}
            print(f"[startup] {ck.parent.parent.name}/{ck.stem} {case}: {counts}", flush=True)
    env.close()
    result = _analyze(records, names, weights.cpu(), base, dt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    return 0


def _analyze(records, names, weights, spec, dt) -> dict:
    import torch

    ti = {t: i for i, t in enumerate(names)}
    std0 = float(spec.reward("track_lin_vel_xy_exp").params["std"])
    w = weights.clone()
    win = records[0]["raw"].shape[1]
    disc = torch.tensor([GAMMA**k for k in range(win)])

    # sanity: recomputed kernels at the training std must equal the recorded terms
    check = max(
        float((torch.exp(-r["lin_e2"] / std0**2) - r["raw"][:, :, ti["track_lin_vel_xy_exp"]]).abs().max())
        for r in records
    )
    check_yaw = max(
        float((torch.exp(-r["yaw_e2"] / std0**2) - r["raw"][:, :, ti["track_ang_vel_z_exp"]]).abs().max())
        for r in records
    )

    def per_step(r, action_l2=None, std=None):
        raw = r["raw"].clone()
        ww = w.clone()
        if action_l2 is not None:
            ww[ti["action_l2"]] = action_l2
        if std is not None:
            raw[:, :, ti["track_lin_vel_xy_exp"]] = torch.exp(-r["lin_e2"] / std**2)
            raw[:, :, ti["track_ang_vel_z_exp"]] = torch.exp(-r["yaw_e2"] / std**2)
        return raw * ww  # (n, win, terms) weighted per-second reward

    variants = {"baseline": {}}
    variants.update({f"action_l2={a:g}": {"action_l2": a} for a in ACTION_L2_WEIGHTS if a != float(w[ti["action_l2"]])})
    variants.update({f"tracking_std={s:g}": {"std": s} for s in TRACKING_STDS if abs(s - std0) > 1e-12})

    by_case: dict = {}
    for r in records:
        by_case.setdefault(r["case"], []).append(r)

    detail: dict = {}
    summary: dict = {"kernel_recompute_max_abs_err": {"lin": check, "yaw": check_yaw}, "cases": {}}
    for case, recs in by_case.items():
        groups = {}
        for g in ("started", "stationary"):
            sel = [(r, [i for i, o in enumerate(r["outcome"]) if o == g]) for r in recs]
            groups[g] = sel
        n_s = sum(len(ix) for _, ix in groups["started"])
        n_z = sum(len(ix) for _, ix in groups["stationary"])
        case_out = {"n_started": n_s, "n_stationary": n_z, "per_checkpoint": {}}
        for r in recs:
            case_out["per_checkpoint"][Path(r["checkpoint"]).stem + "@" + Path(r["checkpoint"]).parent.parent.name] = {
                o: r["outcome"].count(o) for o in ("started", "stationary", "partial", "fell")
            }
        # per-term cumulative contribution over the window (baseline weights), per group
        contrib = {}
        for g, sel in groups.items():
            rows = [per_step(r)[ix] for r, ix in sel if ix]
            if rows:
                x = torch.cat(rows)  # (m, win, terms)
                contrib[g] = {t: float(x[:, :, i].sum(1).mean() * dt) for t, i in ti.items()}
                contrib[g]["TOTAL"] = float(x.sum(2).sum(1).mean() * dt)
        case_out["cumulative_contribution_2s"] = contrib
        adv = {}
        for vname, kw in variants.items():
            tot = {}
            for g, sel in groups.items():
                rows = [per_step(r, **kw)[ix].sum(2) for r, ix in sel if ix]  # (m, win)
                if rows:
                    x = torch.cat(rows)
                    tot[g] = (float(x.sum(1).mean() * dt), float((x * disc).sum(1).mean() * dt))
            if "started" in tot and "stationary" in tot:
                adv[vname] = {
                    "R_started": tot["started"][0],
                    "R_stationary": tot["stationary"][0],
                    "advantage": tot["started"][0] - tot["stationary"][0],
                    "advantage_discounted": tot["started"][1] - tot["stationary"][1],
                }
            elif tot:
                adv[vname] = {f"R_{g}": v[0] for g, v in tot.items()}
        case_out["variants"] = adv
        if "baseline" in adv and "advantage" in adv["baseline"]:
            b = adv["baseline"]["advantage"]
            case_out["delta_advantage_vs_baseline"] = {
                k: v["advantage"] - b for k, v in adv.items() if k != "baseline" and "advantage" in v
            }
        detail[case] = case_out
        summary["cases"][case] = {
            "n_started": n_s,
            "n_stationary": n_z,
            "baseline_advantage": adv.get("baseline", {}).get("advantage"),
            "delta_advantage_vs_baseline": case_out.get("delta_advantage_vs_baseline"),
        }
    # best single change, pooled over cases that have both groups (sum of deltas)
    pooled: dict = {}
    for c in summary["cases"].values():
        for k, v in (c["delta_advantage_vs_baseline"] or {}).items():
            pooled[k] = pooled.get(k, 0.0) + v
    summary["pooled_delta_advantage"] = pooled
    summary["best_single_change"] = max(pooled, key=pooled.get) if pooled else None
    return {"summary": summary, "cases": detail, "window_s": WINDOW_S, "dt": dt, "terms": names}


if __name__ == "__main__":
    sys.exit(main())
