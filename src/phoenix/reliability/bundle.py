"""The executable controller bundle: what actually determines behaviour.

The shield artifact pins the detector and the arbiter. It does not pin the
*controller*, and the controller is what the study measures. Change the action
scale, the default pose, the slew cap, or the control rate, and you have a
different system producing different fall rates against the same artifact, with
nothing in the pipeline noticing.

So a bundle manifest records a content hash for every input that can change the
closed-loop behaviour, and :func:`verify_bundle` refuses to run when any of them
has moved since the manifest was written. This is what makes "we ran arm A and
arm B against the same controller" a checkable statement rather than a claim.

The rate check deserves its own mention. The artifact stores ``arming_ticks``,
``trip_persistence`` and the ramp lengths in **ticks**, not seconds. Running the
same artifact at 100 Hz would halve every one of those durations while every
number in the manifest still matched. :func:`verify_bundle` therefore requires
the control period to equal the artifact's ``control_dt_s`` exactly.

Nothing here imports torch, Isaac or rclpy: the manifest is checkable in CI and
on the robot with equal ease.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_VERSION = 1

# Every controller knob that changes closed-loop behaviour. Recorded by value
# (not by file hash) because they come from a merged config, and it is the
# resolved value that acts on the robot.
CONTROLLER_KEYS = (
    "control_dt_s",
    "action_scale",
    "max_delta_per_step_rad",
    "default_joint_pos",
    "joint_order",
)


def file_sha256(path: str | Path) -> str:
    """Streaming SHA-256 of a file (checkpoints and ONNX payloads are large)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value) -> str:
    """Stable hash of a JSON-serialisable value (sorted keys, no whitespace drift)."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class BundleManifest:
    """Content hashes for every input that can change closed-loop behaviour."""

    files: dict[str, str]  # role -> sha256
    controller: dict  # resolved controller values (see CONTROLLER_KEYS)
    code_commit: str
    code_dirty: bool
    versions: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "manifest_version": MANIFEST_VERSION,
            "files": dict(self.files),
            "controller": self.controller,
            "controller_hash": value_sha256(self.controller),
            "code_commit": self.code_commit,
            "code_dirty": self.code_dirty,
            "versions": self.versions,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BundleManifest:
        version = int(d.get("manifest_version", -1))
        if version != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version {version}")
        return cls(
            files=dict(d["files"]),
            controller=d["controller"],
            code_commit=str(d["code_commit"]),
            code_dirty=bool(d["code_dirty"]),
            versions=d.get("versions", {}),
            note=d.get("note", ""),
        )

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def read(cls, path: str | Path) -> BundleManifest:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def bundle_id(self) -> str:
        """Short, stable identifier for this exact executable configuration."""
        return value_sha256(
            {"files": self.files, "controller": self.controller, "commit": self.code_commit}
        )[:16]


def git_state(repo: str | Path = ".") -> tuple[str, bool]:
    """Return ``(commit, dirty)``; ``("unknown", True)`` outside a git checkout.

    A dirty tree is reported as such rather than hidden: results produced from
    uncommitted code cannot be reconstructed, and the manifest should say so.
    """
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def build_manifest(
    *,
    files: dict[str, str | Path],
    controller: dict,
    versions: dict | None = None,
    note: str = "",
    repo: str | Path = ".",
) -> BundleManifest:
    """Hash every file in ``files`` (role -> path) and snapshot the controller."""
    missing = [k for k in CONTROLLER_KEYS if k not in controller]
    if missing:
        raise ValueError(f"controller snapshot is missing required keys: {missing}")
    hashes = {}
    for role, path in files.items():
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"bundle file '{role}' not found: {p}")
        hashes[role] = file_sha256(p)
    commit, dirty = git_state(repo)
    return BundleManifest(
        files=hashes,
        controller={k: controller[k] for k in CONTROLLER_KEYS},
        code_commit=commit,
        code_dirty=dirty,
        versions=versions or {},
        note=note,
    )


def verify_bundle(
    manifest: BundleManifest,
    *,
    files: dict[str, str | Path],
    controller: dict,
    artifact_control_dt_s: float | None = None,
    allow_dirty: bool = True,
) -> None:
    """Raise unless the current inputs match ``manifest`` exactly.

    Checks, in order: every recorded file's hash, every controller value, and
    the control period against the artifact's own ``control_dt_s``. The rate
    check is separate because a mismatch there is invisible to hashing: the
    artifact's arming, persistence and ramp windows are counted in ticks, so
    running at a different rate silently rescales all of them.
    """
    problems: list[str] = []

    for role, expected in manifest.files.items():
        if role not in files:
            problems.append(f"missing bundle file '{role}'")
            continue
        path = Path(files[role])
        if not path.is_file():
            problems.append(f"bundle file '{role}' not found: {path}")
            continue
        actual = file_sha256(path)
        if actual != expected:
            problems.append(f"'{role}' hash {actual[:12]} != manifest {expected[:12]} ({path})")

    for key in CONTROLLER_KEYS:
        expected_value = manifest.controller.get(key)
        actual_value = controller.get(key)
        if value_sha256(expected_value) != value_sha256(actual_value):
            problems.append(f"controller '{key}' is {actual_value!r}, manifest says {expected_value!r}")

    if artifact_control_dt_s is not None:
        dt = controller.get("control_dt_s")
        if dt is None or abs(float(dt) - float(artifact_control_dt_s)) > 1e-12:
            problems.append(
                f"control period {dt} != artifact control_dt_s {artifact_control_dt_s}; "
                "the artifact's arming/persistence/ramp windows are counted in ticks, so "
                "another rate silently rescales every one of them"
            )

    if manifest.code_dirty and not allow_dirty:
        problems.append("manifest was built from a dirty working tree")

    if problems:
        raise ValueError(
            "bundle verification FAILED, the controller is not the one that was "
            "calibrated:\n  - " + "\n  - ".join(problems)
        )
