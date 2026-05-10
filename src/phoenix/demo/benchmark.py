"""Run the baseline + adapted checkpoints for the demo video.

Produces:

* ``<render-dir>/sim_baseline.mp4`` — baseline policy rollout
* ``<render-dir>/sim_adapted.mp4`` — adapted (Phoenix-loop) policy rollout
* ``<render-dir>/metrics_baseline.json`` / ``metrics_adapted.json``

Called from ``scripts/demo.sh`` inside Isaac Lab's Python context.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("phoenix.demo.benchmark")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline + adapted demo rollouts.")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--adapted", type=Path, required=True)
    p.add_argument("--render-dir", type=Path, required=True)
    p.add_argument("--env-config", type=Path, default=Path("configs/env/rough.yaml"))
    p.add_argument("--video-length", type=int, default=500)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.render_dir.mkdir(parents=True, exist_ok=True)

    # Run each evaluation in its own subprocess. Isaac Sim's SimulationApp is
    # a process-global singleton — once closed, you cannot relaunch it in the
    # same Python process, so doing both rollouts back-to-back in-process
    # silently kills the second run (this exact bug was hit on 2026-04-30).
    # Subprocess isolation guarantees each rollout gets a fresh app.
    isaaclab_path = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
    isaaclab_sh = Path(isaaclab_path) / "isaaclab.sh"

    rc = 0
    for tag, ckpt in (("baseline", args.baseline), ("adapted", args.adapted)):
        if not ckpt.exists():
            logger.warning("Skipping %s — checkpoint %s does not exist", tag, ckpt)
            continue
        video = args.render_dir / f"sim_{tag}.mp4"
        metrics = args.render_dir / f"metrics_{tag}.json"
        logger.info("Evaluating %s checkpoint -> %s", tag, video)
        cmd = [
            str(isaaclab_sh),
            "-p",
            "-m",
            "phoenix.training.evaluate",
            "--checkpoint",
            str(ckpt),
            "--env-config",
            str(args.env_config),
            "--num-envs",
            str(args.num_envs),
            "--num-episodes",
            "8",
            "--video-out",
            str(video),
            "--video-length",
            str(args.video_length),
            "--metrics-out",
            str(metrics),
            "--device",
            args.device,
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("%s eval exited with code %d", tag, result.returncode)
            rc = result.returncode

    return rc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
