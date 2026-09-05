"""Activation: make the bundle that reached the payload the policy that runs.

Transfer is not activation. ``scripts/stage_payload_bundle.sh`` copies a
verified bundle to the payload and re-checks its SHA256SUMS there, which proves
the *bytes arrived*. It proves nothing about which file the policy node opens,
and the two have been different: the deploy configs ship a workstation-relative
``policy.onnx_path`` (``checkpoints/<run>/policy.onnx``), and
``ros2_policy_node.main`` resolves that with a bare ``Path(cfg_onnx)``, i.e.
against the payload's current working directory. Started from the payload's
repo checkout, that path lands on whatever the payload's own ``checkpoints/``
tree holds, which is the older ``phoenix-stand-v3`` era export. The session then
runs a policy nobody deployed while every transfer check reports success.

There is no symlink, no systemd unit, no launch file and no ROS parameter in
this tree that selects a checkpoint. The only selector is the path the operator
hands the node, so activation here means exactly one thing: rewrite the bundle's
own copy of the deploy config so every path the node will open is an absolute
path *inside the bundle*, then read those paths back and hash them.

Two entry points, both fail closed:

``pin``    rewrites the staged config in place (comments preserved: they carry
           the provenance of the export) and refuses if a path the node will
           actually open has no corresponding file in the bundle.
``verify`` re-reads the config the way the node does, resolves every path it
           will open, and compares each file's SHA-256 against the bundle's
           SHA256SUMS. Any mismatch, any relative path, any path escaping the
           bundle, and it exits non-zero.

This module is deliberately stdlib + PyYAML only, with no imports from the rest
of ``phoenix``, so the staging script can drop this single file into the bundle
and run it on the payload with no repo checkout and no PYTHONPATH.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

SUMS_NAME = "SHA256SUMS"

#: Config keys whose value is a path the node may open. Pinning rewrites these
#: to absolute in-bundle paths; verification requires the ones that are
#: actually reachable (see :func:`required_paths`) to be pinned and hashed.
PATH_KEYS = (
    "onnx_path",
    "torchscript_path",
    "stand_onnx_path",
    "walk_onnx_path",
    "artifact",
)

_KEY_LINE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<sep>[ \t]+)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s#][^#]*?)(?P<trail>[ \t]*(?:#.*)?)$"
)


def file_sha256(path: str | Path) -> str:
    """Streaming SHA-256 (ONNX external-data payloads are hundreds of MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse a ``sha256sum`` manifest into ``{basename: hexdigest}``.

    ``stage_payload_bundle.sh`` writes entries as ``./name``; the leading
    ``./`` is stripped so lookups are by plain file name.
    """
    sums: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts[0], parts[1].strip()
        if name.startswith("*"):  # sha256sum binary-mode marker
            name = name[1:]
        sums[Path(name).name] = digest.lower()
    return sums


def required_paths(cfg: dict) -> dict[str, str]:
    """Return ``{role: declared path}`` for every file the node will open.

    Gated on the same flags the node checks, so a disabled mode-switch or a
    disabled shield does not make an unstaged artifact a deploy blocker. The
    torchscript path is included because the parity gate
    (``phoenix.sim2real.verify_deploy``) opens it from the same config, and a
    torchscript that points somewhere else than the ONNX is exactly the
    mismatch that gate exists to detect.
    """
    policy = cfg.get("policy") or {}
    out: dict[str, str] = {}
    if policy.get("onnx_path"):
        out["policy.onnx_path"] = str(policy["onnx_path"])
    if policy.get("torchscript_path"):
        out["policy.torchscript_path"] = str(policy["torchscript_path"])

    mode_switch = policy.get("mode_switch") or {}
    if mode_switch.get("enabled"):
        for key in ("stand_onnx_path", "walk_onnx_path"):
            if mode_switch.get(key):
                out[f"policy.mode_switch.{key}"] = str(mode_switch[key])

    reliability = cfg.get("reliability") or {}
    if reliability.get("enabled"):
        if reliability.get("artifact"):
            out["reliability.artifact"] = str(reliability["artifact"])
    return out


def pin_config_text(
    text: str, bundle_dir: str | Path, present: set[str]
) -> tuple[str, dict[str, str]]:
    """Rewrite every path key whose file is in the bundle to an absolute path.

    Line-based on purpose. The deploy configs carry the export's provenance in
    their comments (parity numbers, what the checkpoint supersedes, the
    external-data warning), and a YAML round-trip would drop all of it.

    ``present`` is the set of file names actually in the bundle. A key whose
    current value names a file that is not in the bundle is left untouched and
    reported as unpinned by its absence from the returned mapping, so the caller
    can decide whether that path is one the node will open.
    """
    bundle = Path(bundle_dir)
    lines = text.splitlines(keepends=True)
    pinned: dict[str, str] = {}
    for i, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped) :]
        match = _KEY_LINE.match(stripped)
        if match is None:
            continue
        key = match.group("key")
        if key not in PATH_KEYS:
            continue
        value = match.group("value").strip().strip("\"'")
        name = Path(value).name
        if name not in present:
            continue
        target = str(bundle / name)
        lines[i] = (
            f'{match.group("indent")}{key}:{match.group("sep")}"{target}"'
            f'{match.group("trail")}{newline}'
        )
        pinned[key] = target
    return "".join(lines), pinned


def unpinned_roles(cfg: dict, target_dir: str | Path) -> list[str]:
    """Roles the node will open that are not absolute paths directly in ``target_dir``.

    Pinning is line-based, so a path key written in a form the line matcher does
    not recognise (a flow mapping, a value on the following line, a key with no
    space after the colon) is silently left alone: ``pin_config_text`` reports
    what it rewrote, never what it failed to find. Re-parsing the rewritten text
    and asking this question is what turns that silence into a refusal, so a
    relative path cannot survive a pin that exits zero.
    """
    target = Path(target_dir)
    bad: list[str] = []
    for role, declared in sorted(required_paths(cfg).items()):
        path = Path(declared)
        if not path.is_absolute() or path.parent != target:
            bad.append(f"{role} is {declared!r}, not an absolute path inside {target}")
    return bad


def find_config(bundle_dir: str | Path) -> Path:
    """Locate the single deploy config in a bundle.

    Ambiguity is an error: two configs in one bundle means the operator picks,
    and the point of this module is that nothing about which policy runs is left
    to a choice made at the lab.
    """
    bundle = Path(bundle_dir)
    candidates = sorted(p for p in bundle.glob("*.yaml") if p.is_file())
    candidates += sorted(p for p in bundle.glob("*.yml") if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"no deploy config (*.yaml) in {bundle}")
    if len(candidates) > 1:
        raise ValueError(
            f"{bundle} holds {len(candidates)} configs "
            f"({', '.join(p.name for p in candidates)}); a bundle must name exactly one"
        )
    return candidates[0]


def verify_activation(
    bundle_dir: str | Path,
    cfg: dict,
    sums: dict[str, str],
    *,
    hasher=file_sha256,
) -> list[str]:
    """Return the list of reasons this bundle is NOT activated. Empty means it is.

    A non-empty return must be treated as fatal by every caller. The checks are
    deliberately about the *resolved* file rather than the declared one: a path
    that is absolute, inside the bundle, present, and hash-equal to the bundle's
    own manifest is the only state in which "the robot runs what we deployed"
    is a checkable statement.
    """
    bundle = Path(bundle_dir).resolve()
    problems: list[str] = []

    for role, declared in sorted(required_paths(cfg).items()):
        path = Path(declared)
        if not path.is_absolute():
            problems.append(
                f"{role} is {declared!r}, a relative path. The node resolves it against "
                f"the payload's working directory, not the bundle, so it can load an "
                f"older checkpoint that happens to sit at that path. Re-stage to pin it."
            )
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(bundle)
        except ValueError:
            problems.append(f"{role} points outside the bundle: {resolved} not under {bundle}")
            continue
        if not resolved.is_file():
            problems.append(f"{role} points at a missing file: {resolved}")
            continue
        expected = sums.get(resolved.name)
        if expected is None:
            problems.append(
                f"{role} resolves to {resolved.name}, which is absent from {SUMS_NAME}; "
                f"its identity is unverifiable"
            )
            continue
        actual = hasher(resolved)
        if actual != expected:
            problems.append(
                f"{role} file {resolved.name} hashes {actual[:12]} but {SUMS_NAME} "
                f"records {expected[:12]}"
            )

    # An external-data ONNX is two files. The sidecar is never named in the
    # config, so a check driven only by config keys would pass on a bundle whose
    # weights never arrived, and the node would fail at load on the robot.
    policy_onnx = (cfg.get("policy") or {}).get("onnx_path")
    if policy_onnx:
        sidecar = Path(str(policy_onnx)).name + ".data"
        if sidecar in sums:
            sidecar_path = bundle / sidecar
            if not sidecar_path.is_file():
                problems.append(f"external-data sidecar {sidecar} is in {SUMS_NAME} but missing")
            elif hasher(sidecar_path) != sums[sidecar]:
                problems.append(f"external-data sidecar {sidecar} does not match {SUMS_NAME}")

    return problems


def bringup_command(bundle_dir: str | Path, config_path: str | Path) -> str:
    """The exact command that activates this bundle, with no placeholders.

    Deliberately omits ``--onnx``: the point of pinning is that the config now
    names the bundle's own export, so passing a path by hand at the lab is the
    step that can silently disagree with what was deployed.
    """
    del bundle_dir
    return "python3 -m phoenix.sim2real.ros2_policy_node " f"--config {Path(config_path)}"


# ----------------------------------------------------------------- CLI


def _load_yaml_text(text: str) -> dict:
    import yaml

    return yaml.safe_load(text) or {}


def _load_yaml(path: Path) -> dict:
    return _load_yaml_text(path.read_text())


def _cmd_pin(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle)
    config = Path(args.config) if args.config else find_config(bundle)
    present = {p.name for p in bundle.iterdir() if p.is_file()}
    target = Path(args.target) if args.target else bundle.resolve()
    text = config.read_text()
    pinned_text, pinned = pin_config_text(text, target, present)

    cfg = _load_yaml_text(text)
    missing = [
        role for role, declared in required_paths(cfg).items() if Path(declared).name not in present
    ]
    if missing:
        print(
            "REFUSING TO PIN: the config enables "
            + ", ".join(missing)
            + " but no matching file was staged into "
            + str(bundle),
            file=sys.stderr,
        )
        return 1

    # Post-condition, checked against the rewritten text before it is written:
    # every path the node will open is now absolute and in the bundle. Without
    # this, a path key the line matcher did not recognise stays relative and
    # pin still exits zero, which is the exact silence this module exists to end.
    leftover = unpinned_roles(_load_yaml_text(pinned_text), target)
    if leftover:
        print(
            "REFUSING TO PIN: these paths did not get pinned, so the node would "
            "still resolve them against its working directory:",
            file=sys.stderr,
        )
        for problem in leftover:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    config.write_text(pinned_text)
    for key, target in sorted(pinned.items()):
        print(f"[activate] pinned {key} -> {target}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle)
    sums_path = bundle / SUMS_NAME
    if not sums_path.is_file():
        print(f"ACTIVATION UNVERIFIED: no {sums_path}", file=sys.stderr)
        return 1
    sums = parse_sha256sums(sums_path.read_text())

    try:
        config = Path(args.config) if args.config else find_config(bundle)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ACTIVATION UNVERIFIED: {exc}", file=sys.stderr)
        return 1

    cfg = _load_yaml(config)
    problems = verify_activation(bundle, cfg, sums)
    if problems:
        print("ACTIVATION UNVERIFIED, the robot is NOT running this bundle:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nNothing was switched over. Whatever the payload was running before is "
            "still what it will run.",
            file=sys.stderr,
        )
        return 1

    onnx = Path(str((cfg.get("policy") or {})["onnx_path"]))
    print(f"[activate] bundle      {bundle.resolve()}")
    print(f"[activate] config      {config}")
    print(f"[activate] policy.onnx {onnx}")
    print(f"[activate] sha256      {sums[onnx.name]}")
    print("[activate] ACTIVATION VERIFIED. Bring up with, verbatim:")
    print(f"    {bringup_command(bundle, config)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    pin = sub.add_parser("pin", help="rewrite a staged config's paths to absolute in-bundle paths")
    pin.add_argument("--bundle", required=True)
    pin.add_argument("--config", default=None, help="defaults to the single *.yaml in the bundle")
    pin.add_argument(
        "--target",
        default=None,
        help="absolute path the bundle will live at (defaults to --bundle). Set this when "
        "staging locally for a directory that will be copied to the payload.",
    )
    pin.set_defaults(func=_cmd_pin)

    verify = sub.add_parser("verify", help="fail closed unless the bundle is the activated policy")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--config", default=None)
    verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
