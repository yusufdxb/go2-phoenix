"""Hardware preflight driver: stage A, stage evaluations, the ledger, one GO / NO-GO.

Normally invoked through ``scripts/harness_preflight.sh``, which owns process
orchestration (launching bridges and probes) and calls this module to judge what
those processes recorded. Subcommands:

``A``         run every offline gate on THIS machine and record ``stage_A.json``.
              Workstation: deploy contract, lock hashes, joint model, observation
              parity, canonical bench, torch-versus-ONNX and TorchScript-versus-ONNX
              parity on the pinned reference inputs (writes ``parity_golden.npz``),
              the no-hardware test suite, lint and type checks.
              Payload (``--payload --bundle DIR``): the same contract, lock, joint
              model, observation parity and bench, plus bundle activation,
              the onnxruntime pin, and ONNX on the PAYLOAD runtime against the
              workstation's golden torch outputs; the workstation's own stage A
              record must travel in the bundle and match this commit and lock.
``evaluate``  judge a recorded stage (B to H) and record ``stage_<X>.json``.
``require``   exit 0 only if every stage before the named one counts.
``status``    print the ledger and answer, with the exit code, whether Phoenix is
              ready for the first stand-only live stage (F).

Nothing here launches a ROS process or moves a motor, and no stage is ever entered
automatically: each live stage is a separate, operator-typed invocation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import preflight_eval as pe
from .activation import SUMS_NAME, file_sha256, parse_sha256sums, verify_activation
from .bridge_telemetry import read_telemetry, summarize
from .deploy_contract import (
    artifact_paths,
    is_stand_only,
    load_lock,
    validate_deploy_contract,
    verify_lock,
)
from .go2_model import (
    JOINT_POSITION_LIMITS_RAD,
    TRAINING_DEFAULT_JOINT_POS,
    UNITREE_EXAMPLE_FOLDED_POSE,
    UNITREE_EXAMPLE_STAND_POSE,
    UNITREE_MOTOR_ORDER,
    limits_in_order,
    verify_default_pose,
    verify_joint_model,
)
from .motor_crc import PHOENIX_FOR_MOTOR
from .provenance import EXPECTED_BRANCH, identity_problems, resolve_code_identity
from .safety import MAX_DELTA_PER_STEP_RAD

REPO_ROOT = Path(__file__).resolve().parents[3]

#: ``pyproject.toml`` [real] pin, with its reason: 1.19+ crashes on the Orin's CPU part.
PAYLOAD_ORT_VERSION = "1.18.1"

#: Modules this pass owns end to end. Lint, format and type checks gate on them; the
#: rest of the repository has pre-existing black and mypy findings that are reported,
#: not silently counted as green.
SAFETY_CORE_MODULES = (
    "src/phoenix/sim2real/actuator_gate.py",
    "src/phoenix/sim2real/bridge_telemetry.py",
    "src/phoenix/sim2real/command_wire.py",
    "src/phoenix/sim2real/deploy_contract.py",
    "src/phoenix/sim2real/go2_model.py",
    "src/phoenix/sim2real/hw_probe.py",
    "src/phoenix/sim2real/preflight.py",
    "src/phoenix/sim2real/preflight_eval.py",
    "src/phoenix/sim2real/provenance.py",
)
SAFETY_CORE_FORMATTED = SAFETY_CORE_MODULES + (
    "src/phoenix/sim2real/lowcmd_bridge_node.py",
    "src/phoenix/sim2real/ros2_policy_node.py",
    "tests/test_actuator_gate.py",
    "tests/test_bridge_telemetry.py",
    "tests/test_command_wire.py",
    "tests/test_deploy_contract.py",
    "tests/test_go2_model.py",
    "tests/test_hw_probe.py",
    "tests/test_lowcmd_bridge.py",
    "tests/test_preflight_eval.py",
    "tests/test_provenance.py",
    "tests/test_ros2_policy_node.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _check(name: str, ok: bool, detail: str, gating: bool = True) -> pe.Check:
    return pe.Check(name=name, ok=bool(ok), detail=str(detail), gating=gating)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ----------------------------------------------------------------- context
class Context:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cfg_path = Path(args.config)
        self.cfg = yaml.safe_load(self.cfg_path.read_text())
        self.lock_path = Path(args.lock)
        self.lock = load_lock(self.lock_path)
        self.lock_sha = file_sha256(self.lock_path)
        self.identity = resolve_code_identity(REPO_ROOT)
        self.expected_sha = getattr(args, "expect_sha", None)
        self.id_problems = identity_problems(self.identity, expected_sha=self.expected_sha)
        self.session = Path(args.session)
        self.session.mkdir(parents=True, exist_ok=True)

    def identity_check(self) -> pe.Check:
        return _check(
            f"code identity: {self.identity.source} {self.identity.sha} on {self.identity.branch}",
            not self.id_problems,
            "; ".join(self.id_problems) or f"expected branch {EXPECTED_BRANCH}, clean",
        )


def write_stage(
    ctx: Context, stage: str, checks: list[pe.Check], extra: dict | None = None
) -> dict:
    record = {
        "schema": pe.STAGE_SCHEMA,
        "stage": stage,
        "title": pe.STAGE_TITLES[stage],
        "verdict": pe.verdict(checks),
        "utc": utc_now(),
        "host": socket.gethostname(),
        "motors": "live" if stage in pe.LIVE_STAGES else "off",
        "rehearsal": os.environ.get("PHOENIX_REHEARSAL") == "1",
        "code_identity": ctx.identity.to_dict(),
        "code_identity_problems": ctx.id_problems,
        "deploy_config_path": str(ctx.cfg_path),
        "lock_path": str(ctx.lock_path),
        "lock_file_sha256": ctx.lock_sha,
        "lock_semantic_sha256": (ctx.lock.get("deploy_config") or {}).get("semantic_sha256"),
        "checks": [c.to_dict() for c in checks],
        **(extra or {}),
    }
    path = ctx.session / f"stage_{stage}.json"
    if path.exists():
        history = ctx.session / "history"
        history.mkdir(exist_ok=True)
        stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y%m%dT%H%M%S")
        shutil.move(str(path), str(history / f"stage_{stage}_{stamp}.json"))
    path.write_text(json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n")
    print_checks(stage, checks, record["verdict"], path)
    return record


def print_checks(
    stage: str, checks: list[pe.Check], verdict: str, path: Path | None = None
) -> None:
    print(f"\n=== stage {stage}: {pe.STAGE_TITLES[stage]} ===")
    for c in checks:
        tag = "info" if not c.gating else ("PASS" if c.ok else "FAIL")
        print(f"  [{tag}] {c.name}: {c.detail}")
    banner = "GO" if verdict == "GO" else "NO-GO"
    print(f"STAGE {stage}: {banner}" + (f"   (evidence: {path})" if path else ""))


def load_records(session: Path) -> dict[str, dict]:
    records = {}
    for stage in pe.STAGES:
        path = session / f"stage_{stage}.json"
        if path.is_file():
            records[stage] = json.loads(path.read_text())
    return records


# ------------------------------------------------------------------ stage A
def a_contract(ctx: Context) -> list[pe.Check]:
    problems = validate_deploy_contract(ctx.cfg)
    source = (ctx.cfg.get("observation") or {}).get("base_lin_vel_source")
    return [
        _check("deploy contract", not problems, "; ".join(problems) or "passes"),
        _check(
            "config is STAND-ONLY",
            is_stand_only(ctx.cfg),
            f"safety.stand_only={is_stand_only(ctx.cfg)}",
        ),
        _check(
            "first live gate uses base_lin_vel_source zeros",
            source == "zeros",
            f"base_lin_vel_source={source!r}; odom stays hardware-unverified",
        ),
        _check(
            "reliability shield not enabled in the first stand gate",
            not (ctx.cfg.get("reliability") or {}).get("enabled"),
            f"reliability={ctx.cfg.get('reliability')}",
        ),
    ]


def a_lock(ctx: Context) -> tuple[list[pe.Check], dict]:
    roles = ("policy.onnx", "policy.onnx.data", "policy.pt", "checkpoint")
    problems, observed = verify_lock(ctx.lock, ctx.cfg, ctx.cfg_path, required_roles=roles)
    checks = [
        _check(
            "every artifact hashes to the lock (onnx, onnx.data, policy.pt, checkpoint, config)",
            not problems,
            "; ".join(problems) or "all match",
        )
    ]
    for role, entry in sorted(observed["artifacts"].items()):
        checks.append(
            _check(f"sha256 {role}", True, f"{entry['sha256']} {entry['resolved']}", gating=False)
        )
    checks.append(
        _check(
            "sha256 deploy config",
            True,
            f"raw {observed['deploy_config']['sha256']} semantic {observed['deploy_config']['semantic_sha256']}",
            gating=False,
        )
    )
    return checks, observed


def a_joint_model(ctx: Context) -> list[pe.Check]:
    order = ctx.cfg.get("joint_order") or []
    perm = verify_joint_model(order, PHOENIX_FOR_MOTOR)
    pose = verify_default_pose((ctx.cfg.get("control") or {}).get("default_joint_pos") or {})
    lo, hi = limits_in_order(UNITREE_MOTOR_ORDER)
    fixtures_ok = all(
        np.all(np.asarray(p) >= lo) and np.all(np.asarray(p) <= hi)
        for p in (UNITREE_EXAMPLE_FOLDED_POSE, UNITREE_EXAMPLE_STAND_POSE)
    )
    default_in = all(
        JOINT_POSITION_LIMITS_RAD[n][0] < v < JOINT_POSITION_LIMITS_RAD[n][1]
        for n, v in TRAINING_DEFAULT_JOINT_POS.items()
    )
    return [
        _check(
            "joint order through the Phoenix to Unitree permutation, by name",
            not perm,
            "; ".join(perm) or "all 12 motors map to their own joint",
        ),
        _check(
            "default joint pose exactly equals training",
            not pose,
            "; ".join(pose) or "all 12 exact",
        ),
        _check(
            "hard joint limits admit the training pose and Unitree's own poses",
            default_in and fixtures_ok,
            "ok" if default_in and fixtures_ok else "limit table rejects a known-legal pose",
        ),
    ]


def a_obs_parity(ctx: Context) -> list[pe.Check]:
    from .obs_parity import check_observation_parity, check_zeros_fallback_is_explicit
    from .observation import JointOrder, ObservationBuilder

    builder = ObservationBuilder(
        JointOrder(tuple(ctx.cfg["joint_order"])), ctx.cfg["control"]["default_joint_pos"]
    )
    report = check_observation_parity(builder)
    extra = check_zeros_fallback_is_explicit()
    return [
        _check(
            "sensor-to-observation parity (term order, slices, scales, no zeroed term)",
            report.passed,
            "; ".join(report.failures) or "all 7 terms pass",
        ),
        _check(
            "zeros base_lin_vel reachable only by explicit selection",
            not extra,
            "; ".join(extra) or "ok",
        ),
    ]


def _ort_session(onnx_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]), ort.__version__


def a_bench(ctx: Context, onnx_path: Path) -> list[pe.Check]:
    from .bench_export import build_canonical_stand_obs
    from .observation import JointOrder, ObservationBuilder

    bench = ctx.lock.get("canonical_bench") or {}
    builder = ObservationBuilder(
        JointOrder(tuple(ctx.cfg["joint_order"])), ctx.cfg["control"]["default_joint_pos"]
    )
    obs = build_canonical_stand_obs(
        builder, pad_zeros=int((ctx.cfg.get("policy") or {}).get("obs_pad_zeros", 0))
    )
    session, _ = _ort_session(onnx_path)
    action = np.asarray(
        session.run(None, {session.get_inputs()[0].name: obs[None].astype(np.float32)})[0]
    ).reshape(-1)
    finite = bool(np.all(np.isfinite(action)))
    a_inf = float(np.max(np.abs(action))) if finite else float("nan")
    want = bench.get("abs_action_inf")
    tol = float(bench.get("max_abs_tol", 1e-5))
    scale = float(ctx.cfg["control"]["action_scale"])
    return [
        _check("canonical-stand ONNX output finite", finite, f"|a|_inf={a_inf}"),
        _check(
            "canonical-stand output reproduces the locked value on this runtime",
            want is not None and finite and abs(a_inf - float(want)) <= tol,
            f"|a|_inf={a_inf:.10f} lock={want} tol={tol:g}",
        ),
        _check(
            "canonical-stand first target step within one slew cap (action_scale*|a|_inf <= 0.175 rad)",
            finite and scale * a_inf <= MAX_DELTA_PER_STEP_RAD,
            f"{scale}*{a_inf:.4f}={scale * a_inf:.4f} rad",
        ),
        _check(
            f"legacy bench threshold {bench.get('legacy_threshold')} (NOT a gate, see lock file)",
            True,
            f"|a|_inf={a_inf:.4f} would {'pass' if finite and a_inf < float(bench.get('legacy_threshold', 0.3)) else 'FAIL'} it",
            gating=False,
        ),
    ]


def _parity_gate_module():
    spec = importlib.util.spec_from_file_location(
        "phoenix_parity_gate", REPO_ROOT / "scripts" / "parity_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves string annotations through sys.modules[cls.__module__],
    # so the module must be registered before it executes. Stage A crashed on
    # exactly this ("'NoneType' object has no attribute '__dict__'").
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def a_parity_workstation(ctx: Context, onnx_path: Path) -> tuple[list[pe.Check], dict]:
    import torch

    pg = _parity_gate_module()
    parity = ctx.lock.get("parity") or {}
    tol = float(parity.get("max_abs_tol", 1e-5))
    cos_tol = float(parity.get("cos_tol", 0.9999))
    max_steps = int(parity.get("max_steps_per_input", 2000))
    checks: list[pe.Check] = []
    batches: list[tuple[str, np.ndarray]] = []
    for entry in ctx.lock.get("parity_reference_inputs") or []:
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            checks.append(_check(f"reference input present: {entry['path']}", False, "missing"))
            continue
        got = file_sha256(path)
        ok = got == entry["sha256"]
        checks.append(
            _check(f"reference input is the pinned file: {entry['path']}", ok, f"sha256 {got}")
        )
        if not ok:
            continue
        obs = (
            pg.obs_from_parquet(path, ctx.cfg, max_steps)
            if entry["kind"] == "parquet"
            else pg.obs_from_npz(path, entry["key"], max_steps)
        )
        batches.append((entry["path"], obs))
    if not batches:
        checks.append(_check("at least one pinned reference batch", False, "none usable"))
        return checks, {}

    paths = artifact_paths(ctx.cfg)
    policy, obs_dim, _ = pg.build_reference_policy(paths["checkpoint"])
    session, ort_version = _ort_session(onnx_path)
    in_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    checks.append(
        _check("ONNX first output is 'action'", out_names[0] == "action", f"outputs {out_names}")
    )
    scripted = torch.jit.load(str(paths["policy.pt"]), map_location="cpu").eval()
    golden: dict[str, np.ndarray] = {}
    sources = []
    for index, (source, obs) in enumerate(batches):
        ref = pg.run_torch(policy, obs)
        onnx_action = session.run(out_names, {in_name: obs})[0]
        with torch.no_grad():
            ts = scripted(torch.from_numpy(obs))
            ts = (ts[0] if isinstance(ts, (tuple, list)) else ts).cpu().numpy()
        for label, reference in (("checkpoint torch", ref), ("TorchScript policy.pt", ts)):
            out = pg.compare("action", reference, onnx_action, max_abs_tol=tol, cos_tol=cos_tol)
            checks.append(
                _check(
                    f"{label} vs ONNX on {source} (n={obs.shape[0]})",
                    out.passed,
                    f"max_abs={out.max_abs:.3e} cos={out.cos_global:.9f} cos_worst={out.cos_worst_sample:.9f} (tol {tol:g}, cos {cos_tol})",
                )
            )
        golden[f"obs_{index}"] = obs.astype(np.float32)
        golden[f"torch_action_{index}"] = np.asarray(ref, dtype=np.float32)
        sources.append(source)
    golden_path = ctx.session / "parity_golden.npz"
    arrays: dict[str, Any] = {"sources": np.asarray(sources), **golden}
    np.savez_compressed(golden_path, **arrays)
    info = {
        "golden_file": golden_path.name,
        "golden_sha256": file_sha256(golden_path),
        "obs_dim": obs_dim,
        "onnxruntime_version": ort_version,
        "torch_version": torch.__version__,
        "max_abs_tol": tol,
        "cos_tol": cos_tol,
    }
    checks.append(
        _check(
            "golden torch outputs written for the payload runtime check",
            True,
            f"{golden_path} sha256 {info['golden_sha256']}",
            gating=False,
        )
    )
    return checks, info


def a_parity_payload(
    ctx: Context, bundle: Path, onnx_path: Path, workstation: dict | None
) -> list[pe.Check]:
    pg = _parity_gate_module()
    checks: list[pe.Check] = []
    info = (workstation or {}).get("parity") or {}
    golden_path = bundle / str(info.get("golden_file", "parity_golden.npz"))
    if not golden_path.is_file() or not info.get("golden_sha256"):
        return [
            _check(
                "golden torch outputs present in the bundle",
                False,
                f"{golden_path} / recorded sha {info.get('golden_sha256')}",
            )
        ]
    got = file_sha256(golden_path)
    checks.append(
        _check(
            "golden file is the one the workstation recorded",
            got == info["golden_sha256"],
            f"sha256 {got}",
        )
    )
    data = np.load(golden_path, allow_pickle=False)
    session, _ = _ort_session(onnx_path)
    in_name = session.get_inputs()[0].name
    tol = float(info.get("max_abs_tol", 1e-5))
    cos_tol = float(info.get("cos_tol", 0.9999))
    index = 0
    while f"obs_{index}" in data.files:
        obs = data[f"obs_{index}"]
        onnx_action = session.run(None, {in_name: obs})[0]
        out = pg.compare(
            "action", data[f"torch_action_{index}"], onnx_action, max_abs_tol=tol, cos_tol=cos_tol
        )
        checks.append(
            _check(
                f"payload ONNX runtime vs workstation torch, batch {index} (n={obs.shape[0]})",
                out.passed,
                f"max_abs={out.max_abs:.3e} cos_worst={out.cos_worst_sample:.9f}",
            )
        )
        index += 1
    if index == 0:
        checks.append(_check("golden file holds at least one batch", False, "empty"))
    return checks


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"{path}: no testsuite element")
    return {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}


def a_tests_and_lint(ctx: Context) -> list[pe.Check]:
    env = dict(
        os.environ, PYTHONPATH=f"{REPO_ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )
    checks: list[pe.Check] = []
    xml = ctx.session / "A_pytest.xml"
    log = ctx.session / "A_pytest.log"
    with log.open("w") as fh:
        rc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "not sim and not ros",
                "-p",
                "no:cacheprovider",
                f"--junitxml={xml}",
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    counts = _junit_counts(xml) if xml.is_file() else {}
    checks.append(
        _check(
            "no-hardware test suite",
            rc == 0
            and counts.get("tests", 0) > 0
            and counts.get("failures", 1) == 0
            and counts.get("errors", 1) == 0,
            f"exit {rc}, {counts} (log {log})",
        )
    )
    for name, cmd in (
        (
            "ruff check src tests (CI scope)",
            [sys.executable, "-m", "ruff", "check", "src", "tests"],
        ),
        (
            "black --check on the safety core and its tests",
            [sys.executable, "-m", "black", "--check", *SAFETY_CORE_FORMATTED],
        ),
        ("mypy on the safety core modules", [sys.executable, "-m", "mypy", *SAFETY_CORE_MODULES]),
    ):
        res = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False
        )
        tail = (res.stdout + res.stderr).strip().splitlines()[-1:] or [""]
        checks.append(_check(name, res.returncode == 0, f"exit {res.returncode}: {tail[0]}"))
    return checks


def cmd_stage_a(args: argparse.Namespace) -> int:
    ctx = Context(args)
    checks: list[pe.Check] = [ctx.identity_check()]
    extra: dict[str, Any] = {"mode": "payload" if args.payload else "workstation"}
    checks += a_contract(ctx)
    lock_checks, observed = a_lock(ctx)
    checks += lock_checks
    extra["artifacts"] = observed
    checks += a_joint_model(ctx)
    checks += a_obs_parity(ctx)
    onnx_path = artifact_paths(ctx.cfg)["policy.onnx"]
    try:
        checks += a_bench(ctx, onnx_path)
    except Exception as exc:  # noqa: BLE001 - a crashed gate is a failed gate
        checks.append(_check("canonical-stand bench ran", False, repr(exc)))

    if args.payload:
        bundle = Path(args.bundle)
        sums_path = bundle / SUMS_NAME
        if sums_path.is_file():
            problems = verify_activation(bundle, ctx.cfg, parse_sha256sums(sums_path.read_text()))
            checks.append(
                _check(
                    "bundle activation verified (pinned in-bundle paths, SHA256SUMS)",
                    not problems,
                    "; ".join(problems) or "ok",
                )
            )
        else:
            checks.append(_check("bundle activation verified", False, f"no {sums_path}"))
        try:
            import onnxruntime

            version = onnxruntime.__version__
        except ImportError as exc:
            version = f"unavailable: {exc}"
        checks.append(
            _check(
                f"onnxruntime is the pinned {PAYLOAD_ORT_VERSION}",
                version == PAYLOAD_ORT_VERSION,
                version,
            )
        )
        ws_path = bundle / "workstation_stage_A.json"
        workstation = json.loads(ws_path.read_text()) if ws_path.is_file() else None
        checks.append(
            _check(
                "workstation stage A travelled in the bundle and is GO for this commit and lock",
                workstation is not None
                and workstation.get("verdict") == "GO"
                and (workstation.get("code_identity") or {}).get("sha") == ctx.identity.sha
                and workstation.get("lock_file_sha256") == ctx.lock_sha
                and not workstation.get("rehearsal"),
                (
                    "missing"
                    if workstation is None
                    else f"verdict {workstation.get('verdict')} commit {(workstation.get('code_identity') or {}).get('sha')}"
                ),
            )
        )
        try:
            checks += a_parity_payload(ctx, bundle, onnx_path, workstation)
        except Exception as exc:  # noqa: BLE001
            checks.append(_check("payload runtime parity ran", False, repr(exc)))
    else:
        try:
            parity_checks, parity_info = a_parity_workstation(ctx, onnx_path)
            checks += parity_checks
            extra["parity"] = parity_info
        except Exception as exc:  # noqa: BLE001
            checks.append(_check("torch vs ONNX parity ran", False, repr(exc)))
        if not args.skip_tests:
            checks += a_tests_and_lint(ctx)
        else:
            checks.append(
                _check(
                    "no-hardware tests, lint, types",
                    False,
                    "skipped with --skip-tests: stage A cannot be GO",
                )
            )
    write_stage(ctx, "A", checks, extra)
    return 0 if pe.verdict(checks) == "GO" else 1


# -------------------------------------------------------------- evaluations
def _prior_stages_check(ctx: Context, stage: str) -> pe.Check:
    status = pe.ledger_status(
        load_records(ctx.session), current_sha=ctx.identity.sha, current_lock_sha256=ctx.lock_sha
    )
    index = pe.STAGES.index(stage)
    # A localhost rehearsal may advance through its own rehearsal records so the
    # whole sequence can be exercised; those records still never count in status.
    accepted = {"GO", "REHEARSAL"} if os.environ.get("PHOENIX_REHEARSAL") == "1" else {"GO"}
    blocking = [r for r in status["stages"][:index] if r["state"] not in accepted]
    return _check(
        f"every stage before {stage} counts (same commit, same lock)",
        not blocking,
        "; ".join(f"{r['stage']}={r['state']}" for r in blocking) or "all prior GO",
    )


def _missing_evidence(args: argparse.Namespace) -> list[str]:
    """Evidence paths a stage needs that do not exist. A halted run leaves gaps."""
    stage = args.stage
    if stage == "B":
        run = Path(args.run_dir or "")
        needed = [run / "bridge.jsonl", run / "processes.json", run / "probe.json"]
    elif stage == "C":
        needed = [Path(args.trace or "")]
    elif stage == "D":
        needed = [Path(args.probe or "")]
    else:
        needed = [Path(t) for t in args.telemetry] or [Path("<no --telemetry given>")]
    return [str(path) for path in needed if not path.is_file()]


def cmd_evaluate(args: argparse.Namespace) -> int:
    ctx = Context(args)
    stage = args.stage
    checks = [ctx.identity_check(), _prior_stages_check(ctx, stage)]
    missing = _missing_evidence(args)
    if missing:
        checks.append(
            _check("all evidence files for this stage exist", False, f"missing: {missing}")
        )
        write_stage(ctx, stage, checks, {"missing_evidence": missing})
        return 1
    extra: dict[str, Any] = {}
    safety = ctx.cfg.get("safety") or {}
    if stage == "B":
        run = Path(args.run_dir)
        manifest, ticks, _ = read_telemetry(run / "bridge.jsonl")
        summary = summarize(manifest, ticks)
        checks += pe.dryrun_checks(
            processes=json.loads((run / "processes.json").read_text()),
            probe=json.loads((run / "probe.json").read_text()),
            manifest=manifest,
            ticks=ticks,
            summary=summary,
            cfg=ctx.cfg,
            lock=ctx.lock,
            expected_sha=ctx.identity.sha,
            watchdog_s=float((manifest.get("gate_params") or {}).get("watchdog_s", 0.2)),
        )
        extra.update(run_dir=str(run), telemetry_summary=summary)
    elif stage == "C":
        trace = json.loads(Path(args.trace).read_text())
        checks += pe.deadman_trace_checks(trace, estop_timeout_s=float(safety["estop_timeout_s"]))
        extra.update(trace=str(args.trace))
    elif stage == "D":
        probe = json.loads(Path(args.probe).read_text())
        for topic in pe.SENSOR_TOPICS:
            checks += pe.rate_checks(
                probe, topic, floor_hz=pe.RATE_FLOOR_HZ, max_gap_s=float(safety["sensor_timeout_s"])
            )
        checks += pe.sensor_content_checks(probe)
        extra.update(probe=str(args.probe))
    elif stage == "E":
        manifest, ticks, _ = read_telemetry(args.telemetry[0])
        summary = summarize(manifest, ticks)
        params = manifest.get("gate_params") or {}
        checks += pe.hold_test_checks(
            manifest=manifest,
            ticks=ticks,
            summary=summary,
            cfg=ctx.cfg,
            lock=ctx.lock,
            expected_sha=ctx.identity.sha,
            duration_s=float(args.duration),
            hold_kp=float(params.get("hold_kp", float("nan"))),
            hold_kd=float(params.get("hold_kd", float("nan"))),
            watchdog_s=float(params.get("watchdog_s", 0.2)),
        )
        extra.update(telemetry=str(args.telemetry[0]), telemetry_summary=summary)
    else:  # F, G, H
        expected_attempts = pe.H_ATTEMPTS if stage == "H" else 1
        confirmations = [c.lower() in ("y", "yes") for c in (args.operator_confirmed or [])]
        attempts = []
        checks.append(
            _check(
                f"{expected_attempts} attempt(s) recorded",
                len(args.telemetry) == expected_attempts
                and len(confirmations) == expected_attempts,
                f"{len(args.telemetry)} telemetry file(s), {len(confirmations)} operator confirmation(s)",
            )
        )
        for index, telemetry in enumerate(args.telemetry):
            manifest, ticks, _ = read_telemetry(telemetry)
            summary = summarize(manifest, ticks)
            attempt_checks = pe.stand_checks(
                stage=stage,
                manifest=manifest,
                ticks=ticks,
                summary=summary,
                cfg=ctx.cfg,
                lock=ctx.lock,
                expected_sha=ctx.identity.sha,
                authority_s=pe.STAND_AUTHORITY_S[stage],
                watchdog_s=float((manifest.get("gate_params") or {}).get("watchdog_s", 0.2)),
                operator_confirmed_stand=index < len(confirmations) and confirmations[index],
            )
            print_checks(stage, attempt_checks, pe.verdict(attempt_checks))
            attempts.append(
                {
                    "telemetry": str(telemetry),
                    "verdict": pe.verdict(attempt_checks),
                    "checks": [c.to_dict() for c in attempt_checks],
                    "telemetry_summary": summary,
                }
            )
            checks.append(
                _check(
                    f"attempt {index + 1} ({Path(telemetry).name})",
                    pe.verdict(attempt_checks) == "GO",
                    f"{sum(not c.ok for c in attempt_checks if c.gating)} failing gating checks",
                )
            )
        extra.update(attempts=attempts)
    write_stage(ctx, stage, checks, extra)
    return 0 if pe.verdict(checks) == "GO" else 1


# ------------------------------------------------------------ status/require
def cmd_status(args: argparse.Namespace) -> int:
    ctx = Context(args)
    status = pe.ledger_status(
        load_records(ctx.session), current_sha=ctx.identity.sha, current_lock_sha256=ctx.lock_sha
    )
    print(f"\nPhoenix hardware gate ledger   session {ctx.session}")
    print(
        f"code   {ctx.identity.source} {ctx.identity.sha} branch {ctx.identity.branch} dirty={ctx.identity.dirty}"
    )
    print(f"config {ctx.cfg_path}")
    print(f"lock   {ctx.lock_path} sha256 {ctx.lock_sha}")
    for problem in ctx.id_problems:
        print(f"  CODE IDENTITY PROBLEM: {problem}")
    for row in status["stages"]:
        print(
            f"  {row['stage']}  {row['state']:9s} {row['title']}"
            + (f"   <- {row['why']}" if row["why"] else "")
        )
    ready = status["ready_for_stand_F"] and not ctx.id_problems
    print(f"\nnext permitted stage: {status['next_stage'] or 'none (all stages GO)'}")
    print(
        f"ready for live hold stage E:                   {'YES' if status['ready_for_live_hold_E'] and not ctx.id_problems else 'NO'}"
    )
    print(f"READY FOR STAND-ONLY LIVE GO2 TEST (stage F):  {'YES' if ready else 'NO'}")
    if status["stand_gate_passed"] and not ctx.id_problems:
        print(
            "H25 stand gate passed (A to H). Walking stays blocked in code: see deploy_contract.WALKING_PREREQUISITES."
        )
    return 0 if ready else 1


def cmd_require(args: argparse.Namespace) -> int:
    ctx = Context(args)
    check = _prior_stages_check(ctx, args.stage)
    ok = check.ok and not ctx.id_problems
    print(
        f"[require {args.stage}] {'OK' if ok else 'REFUSED'}: {check.detail}"
        + ("" if not ctx.id_problems else f"; identity: {ctx.id_problems}")
    )
    return 0 if ok else 1


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--lock", required=True)
    common.add_argument("--session", required=True)
    common.add_argument("--expect-sha", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("A", parents=[common], help="offline gates on this machine")
    a.add_argument("--payload", action="store_true")
    a.add_argument("--bundle", default=None)
    a.add_argument("--skip-tests", action="store_true", help="debug only; forces NO-GO")
    a.set_defaults(func=cmd_stage_a)

    ev = sub.add_parser("evaluate", parents=[common], help="judge a recorded stage")
    ev.add_argument("stage", choices=pe.STAGES[1:])
    ev.add_argument("--run-dir")
    ev.add_argument("--trace")
    ev.add_argument("--probe")
    ev.add_argument("--telemetry", nargs="+", default=[])
    ev.add_argument("--duration", type=float, default=None)
    ev.add_argument("--operator-confirmed", nargs="+", default=[])
    ev.set_defaults(func=cmd_evaluate)

    req = sub.add_parser("require", parents=[common])
    req.add_argument("stage", choices=pe.STAGES)
    req.set_defaults(func=cmd_require)

    st = sub.add_parser("status", parents=[common])
    st.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "payload", False) and not getattr(args, "bundle", None):
        print("--payload requires --bundle", file=sys.stderr)
        return 2
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
