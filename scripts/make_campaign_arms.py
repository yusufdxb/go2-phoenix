"""Write (or load) the arms file of a hardware campaign (scripts/hw_campaign.sh).

The arms file pins, per arm, the ONNX / data / checkpoint sha256, the stand and walk lock
paths and optionally the bundle directories, plus the block order and the per-block slot
sequence. The hashes are COMPUTED from the stand bundle's files, never typed:

    python3 scripts/make_campaign_arms.py --name abc_2026-10-17 --out configs/sim2real/campaigns/abc_2026-10-17.arms.json \\
        --arm A:incumbent:~/phoenix/abA-stand-1a2b3c4d:configs/sim2real/locks/A_stand.lock.yaml:configs/sim2real/locks/A_walk.lock.yaml \\
        --arm B:friction_b:...:...:... --arm C:friction_c:...:...:...[:WALK_BUNDLE_DIR]

``--arm`` is ID:LABEL:STAND_BUNDLE:STAND_LOCK:WALK_LOCK[:WALK_BUNDLE]. Each stand bundle must hold
SHA256SUMS (verified here), policy.onnx, policy.onnx.data, one model_*.pt and phoenix_manifest.json.
``--emit-shell FILE`` validates an arms file and prints the shell assignments hw_campaign.sh evals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from phoenix.sim2real.walk_grant import GANTRY_PRESETS  # noqa: E402

DEFAULT_PREREG = "docs/preregistration/2026-10-07-gantry-ab-exploratory.md"
DEFAULT_ORDER = "A B C C B A B C A"  # thirds: forward, reversed, rotated (each arm 3 blocks)
DEFAULT_SLOTS = "G vx03_4s yaw05_3s yawm05_3s vx05_3s yaw08_3s vx02_yaw03_3s"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ARM_KEYS = ("onnx_sha256", "data_sha256", "ckpt_sha256")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_hashes(bundle: Path) -> dict[str, str]:
    """Verify SHA256SUMS and return the three pinned hashes of a stand bundle."""
    sums = bundle / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError(f"{bundle}: no SHA256SUMS")
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        want, _, name = line.partition("  ")
        name = name.lstrip("*")
        if sha256(bundle / name) != want:
            raise ValueError(f"{bundle}: SHA256SUMS does not verify ({name})")
    for name in ("policy.onnx", "policy.onnx.data", "phoenix_manifest.json"):
        if not (bundle / name).is_file():
            raise ValueError(f"{bundle}: missing {name}")
    json.loads((bundle / "phoenix_manifest.json").read_text())
    ckpts = sorted(p for p in bundle.glob("*.pt") if p.name != "policy.pt")
    if len(ckpts) != 1:
        raise ValueError(f"{bundle}: expected exactly one model_*.pt checkpoint, found {len(ckpts)}")
    return {
        "onnx_sha256": sha256(bundle / "policy.onnx"),
        "data_sha256": sha256(bundle / "policy.onnx.data"),
        "ckpt_sha256": sha256(ckpts[0]),
    }


def validate(doc: dict) -> None:
    """Raise ValueError unless the arms file is well formed (used by make and by the runner)."""
    arms = doc.get("arms")
    if not isinstance(arms, dict) or len(arms) < 2:
        raise ValueError("arms: need at least two arms")
    for arm, spec in arms.items():
        if not re.fullmatch(r"[A-Z]", arm):
            raise ValueError(f"arm id {arm!r} must be one capital letter")
        for k in ARM_KEYS:
            if not HEX64.match(str(spec.get(k, ""))):
                raise ValueError(f"arm {arm}: {k} is not a sha256 hex digest")
        for k in ("stand_lock", "walk_lock"):
            if not spec.get(k):
                raise ValueError(f"arm {arm}: {k} is required")
    if len({tuple(s[k] for k in ARM_KEYS) for s in arms.values()}) != len(arms):
        raise ValueError("two arms share the same artifact hashes")
    order, slots = doc.get("order"), doc.get("slots")
    if not order or any(a not in arms for a in order):
        raise ValueError(f"order {order!r} must be a non-empty list of arm ids from {sorted(arms)}")
    if set(order) != set(arms):
        raise ValueError("every arm must appear in the order")
    if not slots or slots[0] != "G" or "G" in slots[1:] or len(set(slots)) != len(slots):
        raise ValueError("slots: start with G, no repeats")
    for s in slots[1:]:
        if s not in GANTRY_PRESETS:
            raise ValueError(f"slot {s!r} is not a preset W may authorize ({GANTRY_PRESETS})")
    if not doc.get("name") or not re.fullmatch(r"[A-Za-z0-9_.-]+", doc["name"]):
        raise ValueError("name must be a simple identifier")


def emit_shell(path: Path) -> str:
    doc = json.loads(path.read_text())
    validate(doc)
    q = shlex.quote
    out = [
        f"CAMPAIGN_NAME={q(doc['name'])}",
        f"PREREG={q(doc.get('prereg') or '')}",
        f"ARM_IDS=({' '.join(q(a) for a in sorted(doc['arms']))})",
        f"ORDER=({' '.join(q(a) for a in doc['order'])})",
        f"SLOTS=({' '.join(q(s) for s in doc['slots'])})",
    ]
    last = {a: max(i for i, x in enumerate(doc["order"], 1) if x == a) for a in doc["arms"]}
    out.append(f"STOP_DRILL_BLOCKS=({' '.join(str(b) for b in sorted(last.values()))})")
    tables = {
        "ONNX_SHA": "onnx_sha256",
        "DATA_SHA": "data_sha256",
        "CKPT_SHA": "ckpt_sha256",
        "STAND_LOCK": "stand_lock",
        "WALK_LOCK": "walk_lock",
        "STAND_DIR": "stand_bundle",
        "WALK_DIR": "walk_bundle",
        "ARM_LABEL": "label",
    }
    for var, key in tables.items():
        out.append(f"declare -A {var}=()")
        for arm, spec in doc["arms"].items():
            out.append(f"{var}[{arm}]={q(spec.get(key) or '')}")
    return "\n".join(out)


def build(args: argparse.Namespace) -> dict:
    arms = {}
    for spec in args.arm:
        parts = spec.split(":")
        if len(parts) not in (5, 6):
            raise ValueError(f"--arm {spec!r}: need ID:LABEL:STAND_BUNDLE:STAND_LOCK:WALK_LOCK[:WALK_BUNDLE]")
        arm, label, stand, stand_lock, walk_lock = parts[:5]
        walk = parts[5] if len(parts) == 6 else None
        hashes = bundle_hashes(Path(stand).expanduser())
        arms[arm] = {
            "label": label,
            **hashes,
            "stand_lock": stand_lock,
            "walk_lock": walk_lock,
            "stand_bundle": stand,
            "walk_bundle": walk,
        }
    doc = {
        "name": args.name,
        "prereg": args.prereg,
        "order": args.order.split(),
        "slots": args.slots.split(),
        "arms": arms,
    }
    validate(doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emit-shell", type=Path, metavar="ARMS_FILE")
    ap.add_argument("--name")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--order", default=DEFAULT_ORDER)
    ap.add_argument("--slots", default=DEFAULT_SLOTS)
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    args = ap.parse_args(argv)
    try:
        if args.emit_shell is not None:
            print(emit_shell(args.emit_shell))
            return 0
        if not (args.name and args.out and args.arm):
            ap.error("--name, --out and at least two --arm are required")
        doc = build(args)
    except (OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {args.out}: arms {' '.join(doc['arms'])}, order {' '.join(doc['order'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
