"""Demo pipeline — produces the side-by-side SIM/REAL/SIM+Phoenix video.

* :mod:`phoenix.demo.benchmark` — evaluates baseline + adapted checkpoints,
  records rollout videos, and writes metrics JSON.
* :mod:`phoenix.demo.video_compose` — stitches three clips side-by-side with
  labels. Pure-ffmpeg; no Isaac Lab needed.
"""

from .video_compose import compose_side_by_side

__all__ = ["compose_side_by_side"]
