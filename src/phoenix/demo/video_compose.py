"""Compose a three-panel side-by-side video: SIM | REAL | SIM-after-Phoenix.

Pure ffmpeg — runs in system Python, no Isaac Lab dependency. Missing
inputs are substituted with a placeholder colour bar so the pipeline
always produces an artifact (useful on first run before the real clip
has been recorded).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("phoenix.demo.video_compose")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compose a 3-panel side-by-side demo video.")
    p.add_argument("--sim", type=Path, required=True, help="Left panel (baseline sim)")
    p.add_argument("--real", type=Path, required=True, help="Middle panel (real deployment clip)")
    p.add_argument("--improved", type=Path, required=True, help="Right panel (adapted sim)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--labels",
        nargs=3,
        default=["SIM", "REAL", "SIM + Phoenix"],
        help="Three overlay labels (left, middle, right)",
    )
    p.add_argument("--height", type=int, default=540, help="Per-panel height in px")
    return p.parse_args(argv)


def compose_side_by_side(
    sim: Path,
    real: Path,
    improved: Path,
    out: Path,
    *,
    labels: tuple[str, str, str] = ("SIM", "REAL", "SIM + Phoenix"),
    height: int = 540,
) -> int:
    """Build a 3-panel horizontal stack using ffmpeg. Returns the exit code."""
    inputs = [
        _resolve_or_placeholder(sim, height, label="missing sim"),
        _resolve_or_placeholder(real, height, label="missing real"),
        _resolve_or_placeholder(improved, height, label="missing adapted"),
    ]

    # Each input is scaled to the same height, labelled via drawtext, then hstack'd.
    filter_parts = []
    for i, label in enumerate(labels):
        filter_parts.append(
            f"[{i}:v]scale=-2:{height},"
            f"drawtext=text='{_escape(label)}':fontcolor=white:fontsize=28:"
            "box=1:boxcolor=black@0.5:boxborderw=8:x=20:y=20[v{i}]".format(i=i)
        )
    filter_parts.append("[v0][v1][v2]hstack=inputs=3[out]")
    filter_complex = ";".join(filter_parts)

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    for src in inputs:
        cmd += ["-i", str(src)]
    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    logger.info("Running ffmpeg: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def _resolve_or_placeholder(path: Path, height: int, *, label: str) -> Path:
    """If ``path`` exists return it; otherwise synthesize a 5s placeholder."""
    if path.exists():
        return path
    placeholder = path.with_name(path.stem + "_placeholder.mp4")
    if placeholder.exists():
        return placeholder
    logger.warning("Missing clip %s — generating placeholder %s", path, placeholder)
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=gray:s=960x{height}:d=5",
        "-vf",
        f"drawtext=text='{_escape(label)}':fontcolor=white:fontsize=36:x=(w-tw)/2:y=(h-th)/2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(placeholder),
    ]
    subprocess.run(cmd, check=True)
    return placeholder


def _escape(text: str) -> str:
    """Escape characters that ffmpeg's drawtext filter interprets specially."""
    return text.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return compose_side_by_side(
        args.sim,
        args.real,
        args.improved,
        args.out,
        labels=tuple(args.labels),
        height=args.height,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    sys.exit(main())
