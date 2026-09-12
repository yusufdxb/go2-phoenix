"""Diagnostic: does a terminated environment terminate AGAIN one tick after reset?

The historical harvest recorded every physical fall twice: a long window at step
s, then a one-row window for the same environment at step s + 1, both with
``base_contact``. This script tests the candidate mechanism interventionally.

Candidate mechanism. ``TerminationManager.compute`` rebuilds its buffers from
scratch every step, and ``ContactSensor.reset`` zeroes the contact history, so
neither survives a reset on its own. But the contact sensor fills its history
LAZILY, on the first ``.data`` access after it is marked outdated, from whatever
PhysX last simulated. ``_reset_idx`` marks the reset environment outdated without
stepping physics. A ``.data`` read between that reset and the next physics step
therefore writes the PRE-reset base contact force into the fresh history, and
``illegal_contact`` (a max over that history) fires on the next tick. The
harvest's post-step ``snapshot_manager_state`` performs exactly such a read.

Three phases in one process, same environment, no policy (random actions plus the
harvest's interval push, to make falls happen):

* ``access``         read contact data after every step, as the harvest does
* ``no_access``      never read contact data after a step
* ``access_rereset`` read it, then ``sensor.reset(env_ids)`` for environments reset
                     this step, undoing the side effect

Prediction if the mechanism is right: ``access`` shows a second termination on
the first step after nearly every reset, ``no_access`` and ``access_rereset`` show
none. This is an evaluation rollout. Nothing is trained.

    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ~/Sim/isaac-sim-venv/bin/python \
        scripts/diag_post_reset_termination.py --out <path>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PHASES = ("access", "no_access", "access_rereset")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-config", type=Path, default=REPO_ROOT / "configs/env/flat.yaml")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--steps-per-phase", type=int, default=400)
    p.add_argument("--action-std", type=float, default=2.0)
    p.add_argument("--push-velocity", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        return _run(args)
    except BaseException:
        # simulation_app.close() can exit the process before an in-flight
        # traceback reaches stderr, so print it first.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close()


def _run(args) -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    import warp as wp
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg

    from phoenix.sim_env import build_env_cfg, load_layered_config

    loaded = load_layered_config(args.env_config)
    cfg = build_env_cfg(loaded)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    push = {
        "x": (-args.push_velocity, args.push_velocity),
        "y": (-args.push_velocity, args.push_velocity),
    }
    if getattr(cfg.events, "push_robot", None) is None:
        cfg.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={"velocity_range": push},
        )
    else:
        cfg.events.push_robot.mode = "interval"
        cfg.events.push_robot.interval_range_s = (2.0, 4.0)
        cfg.events.push_robot.params["velocity_range"] = push

    task = loaded.to_container()["env"]["task_name"]
    env = gym.make(task, cfg=cfg, render_mode=None).unwrapped
    sensor = env.scene["contact_forces"]
    base_ids = [i for i, n in enumerate(sensor.body_names) if n == "base"]
    assert len(base_ids) == 1, sensor.body_names
    base_id = base_ids[0]
    manager = env.termination_manager
    origins = env.scene.env_origins

    def base_history_norms() -> np.ndarray:
        # Called only where the sensor is already up to date (inside _reset_idx,
        # right after termination compute accessed it), so it rolls nothing.
        hist = wp.to_torch(sensor.data.net_forces_w_history)[:, :, base_id]
        return torch.linalg.norm(hist, dim=-1).cpu().numpy()

    step_holder = {"step": -1, "phase": None}
    resets: list[dict] = []
    original = env._reset_idx

    def hooked(env_ids, *a, **kw):
        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        if ids:
            norms = base_history_norms()
            ep_len = env.episode_length_buf.cpu().numpy()
            term = manager.terminated.cpu().numpy()
            base_contact = manager.get_term("base_contact").cpu().numpy()
            # Articulation data are warp arrays in Isaac Lab 3.0.
            root_pos = wp.to_torch(env.scene["robot"].data.root_pos_w)
            z = (root_pos[:, 2] - torch.as_tensor(origins)[:, 2]).cpu().numpy()
            for i in ids:
                resets.append(
                    {
                        "phase": step_holder["phase"],
                        "step": step_holder["step"],
                        "env": int(i),
                        "episode_length_at_reset": int(ep_len[i]),
                        "terminated": bool(term[i]),
                        "base_contact": bool(base_contact[i]),
                        "base_contact_history_norm_N": [float(v) for v in norms[i]],
                        "base_height_m": float(z[i]),
                    }
                )
        return original(env_ids, *a, **kw)

    env._reset_idx = hooked
    env.reset()
    action_dim = env.action_manager.total_action_dim
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    step = 0
    with torch.inference_mode():
        for phase in PHASES:
            step_holder["phase"] = phase
            for _ in range(args.steps_per_phase):
                step_holder["step"] = step
                n_before = len(resets)
                actions = (
                    torch.randn((env.num_envs, action_dim), device=args.device, generator=gen)
                    * args.action_std
                )
                env.step(actions)
                reset_now = sorted({r["env"] for r in resets[n_before:]})
                if phase in ("access", "access_rereset"):
                    # The harvest's post-step read, reduced to the part that matters.
                    _ = wp.to_torch(sensor.data.net_forces_w).sum()
                    if phase == "access_rereset" and reset_now:
                        sensor.reset(reset_now)
                step += 1
    env._reset_idx = original
    env.close()

    summary = {}
    for phase in PHASES:
        rows = [r for r in resets if r["phase"] == phase and r["terminated"]]
        by_env_step = {(r["env"], r["step"]) for r in rows}
        followers = [r for r in rows if (r["env"], r["step"] - 1) in by_env_step]
        leaders = [r for r in rows if (r["env"], r["step"] + 1) in by_env_step]
        summary[phase] = {
            "terminations": len(rows),
            "terminations_on_step_after_own_reset": len(followers),
            "leaders_followed_by_one": len(leaders),
            "follower_episode_length_at_reset": sorted(
                {r["episode_length_at_reset"] for r in followers}
            ),
            "follower_min_base_height_m": min(
                (r["base_height_m"] for r in followers), default=None
            ),
            "follower_base_contact_history_N_example": (
                followers[0]["base_contact_history_norm_N"] if followers else None
            ),
            "non_follower_episode_length_min": min(
                (r["episode_length_at_reset"] for r in rows if r not in followers), default=None
            ),
        }
    payload = {
        "script": "scripts/diag_post_reset_termination.py",
        "task": task,
        "env_config": str(args.env_config.relative_to(REPO_ROOT)),
        "num_envs": args.num_envs,
        "steps_per_phase": args.steps_per_phase,
        "action_std": args.action_std,
        "push_velocity": args.push_velocity,
        "seed": args.seed,
        "contact_history_length": int(sensor.cfg.history_length),
        "summary": summary,
        "resets": resets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    print(f"[diag] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
