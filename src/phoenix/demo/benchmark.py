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

    # Dispatch into the training.evaluate module twice, once per checkpoint.
    from phoenix.training import evaluate as eval_mod

    for tag, ckpt in (("baseline", args.baseline), ("adapted", args.adapted)):
        if not ckpt.exists():
            logger.warning("Skipping %s — checkpoint %s does not exist", tag, ckpt)
            continue
        video = args.render_dir / f"sim_{tag}.mp4"
        metrics = args.render_dir / f"metrics_{tag}.json"
        logger.info("Evaluating %s checkpoint -> %s", tag, video)
        eval_mod.main(
            [
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
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
