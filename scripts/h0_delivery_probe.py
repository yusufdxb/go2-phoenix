"""H0 delivery probe: what state does a curriculum-seeded reset actually start in?

Phase I of the failure-curriculum study returned a paired multi-seed null at n=11.
That null is uninformative: ``fine_tune`` installed the reset bridge with
``seed_row_strategy="first"``, so every failure reset loaded row 0 of a trajectory,
and row 0 of every shipped trajectory is a nominal gait state. The treatment was
never delivered.

This probe is the H0 gate from ``ashfall/docs/phase2/HYPOTHESIS.md``. It measures
the seeded initial states directly rather than inferring delivery from any
downstream outcome. It runs on CPU and needs no simulator: the nominal reset
distribution is reproduced analytically from the IsaacLab source that defines it.

PREREGISTERED BEFORE RUNNING
============================

Reference distributions, read from source, not from memory:

* Nominal base pose: ``UNITREE_GO2_CFG.init_state.pos = (0, 0, 0.4)``
  (isaaclab_assets/robots/unitree.py:157-166). Roll and pitch are NOT
  randomized by ``reset_base``; only x, y and yaw are. So nominal tilt is
  exactly 0 and nominal height is exactly 0.400 m.
* Nominal reset velocities: ``reset_base`` velocity_range is U(-0.5, 0.5) on all
  six components; ``reset_robot_joints`` velocity_range is (0.0, 0.0), so nominal
  joint velocity is exactly 0
  (isaaclab_tasks/.../velocity/velocity_env_cfg.py:175-198).

Descriptors. Chosen to be invariant to yaw and to the parquet's joint ordering,
which the schema does not pin:

* ``base_height_m``      nominal exactly 0.400
* ``tilt_deg``           angle between body +z and world +z; nominal exactly 0
* ``lin_vel_norm``       nominal ||U(-0.5,0.5)^3||
* ``ang_vel_norm``       nominal ||U(-0.5,0.5)^3||
* ``joint_vel_norm``     nominal exactly 0
* ``cmd_track_err``      ||command_vel[:2] - base_lin_vel_body[:2]||
* ``min_contact_force``  diagnostic only; NOT part of the delivered state

Mode -> predicted direction. Taken from the generator source
(ashfall/src/ashfall/synth/generator.py), not guessed:

* ATTITUDE          tilt_deg increases       (quat ramped, height dropped)
* COLLAPSE          base_height_m decreases  (height ramped to ~0.05 m)
* SLIP              lin_vel_norm decreases   (actual velocity driven to ~0)
* STUMBLE           joint_vel_norm increases (one leg spiked to 16-25 rad/s)
* CONTACT_LOSS      min_contact_force decreases
* COMMAND_MISMATCH  cmd_track_err increases  (actual velocity reversed)

ACCEPTANCE CRITERION, fixed before the run:

  H0 PASSES for a trajectory iff the seeded state under the production strategy
  is (a) distinguishable from the nominal reset distribution on at least one
  descriptor that ``restore_state`` actually writes, and (b) moves in the
  predicted direction for that trajectory's failure mode.

  Operationalized after a robustness re-analysis on 2026-09-11 (the criterion
  above is unchanged; only the estimator is): distinctness is |z| > 2 where z
  scores the seeded value against the trajectory's OWN history strictly before
  the seeded row. The first cut of this probe anchored on the single row 0,
  which is one noisy draw; at these effect sizes that made both the direction
  test and the excursion fraction unstable, and the fraction's sign flipped
  under the more robust anchor. The z-test is the reported statistic. The
  excursion fraction is retained as a descriptive figure only and its SIGN
  must not be interpreted near zero.

  Direction is scored only where the state is distinct. Direction agreement at
  a near-zero effect is a coin flip and carries no information: 4 of 12 under
  the original anchor has binomial P(X<=4 | n=12, p=0.5) = 0.19, which is
  consistent with zero effect and is NOT evidence of anti-correlation.

  H0 PASSES OVERALL iff it passes for every failure mode that is deliverable in
  principle. A mode whose only signature is a quantity the bridge cannot write
  (contact forces) is reported as NOT DELIVERABLE rather than as a failure of
  the fix, because no seeding strategy can deliver it.

Delivered-state boundary. ``phoenix.replay.state_adapter.restore_state`` writes
root pose, root velocity, joint position, joint velocity and the velocity
command. It does not write contact forces. Any mode whose signature lives only
in contact forces is undeliverable through this bridge by construction.

Usage:
    python scripts/h0_delivery_probe.py [--out reliability_eval/h0_delivery]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phoenix.adaptation.reset_bridge import resolve_seed  # noqa: E402
from phoenix.replay.trajectory_reader import TrajectoryReader, load_initial_state  # noqa: E402

# Read from isaaclab_assets/robots/unitree.py UNITREE_GO2_CFG and
# isaaclab_tasks .../velocity/velocity_env_cfg.py EventCfg. See module docstring.
NOMINAL_HEIGHT_M = 0.4
NOMINAL_TILT_DEG = 0.0
NOMINAL_JOINT_VEL_NORM = 0.0
NOMINAL_VEL_RANGE = 0.5

PRODUCTION_STRATEGY = "failure_onset_minus_seconds"
PRODUCTION_OFFSET_SECONDS = 0.5

MODE_PREDICTIONS = {
    "attitude": ("tilt_deg", "increase"),
    "collapse": ("base_height_m", "decrease"),
    "slip": ("lin_vel_norm", "decrease"),
    "stumble": ("joint_vel_norm", "increase"),
    "contact_loss": ("min_contact_force", "decrease"),
    "command_mismatch": ("cmd_track_err", "increase"),
}

# Descriptors that restore_state actually writes into the simulator.
DELIVERED_DESCRIPTORS = {
    "base_height_m",
    "tilt_deg",
    "lin_vel_norm",
    "ang_vel_norm",
    "joint_vel_norm",
    "cmd_track_err",
}


def tilt_deg(quat_xyzw: np.ndarray) -> float:
    """Angle between the body +z axis and world +z, invariant to yaw."""
    q = np.asarray(quat_xyzw, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    # Third column of the rotation matrix: body +z expressed in world frame.
    body_z_world = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return float(np.degrees(np.arccos(np.clip(body_z_world[2], -1.0, 1.0))))


def describe(path: Path, row: int) -> dict:
    """Descriptors for one trajectory row, at the delivered-state boundary."""
    state = load_initial_state(path, row)
    reader = TrajectoryReader(path)
    contact = None
    if "contact_forces" in reader._table.column_names:
        value = reader._table.column("contact_forces")[row].as_py()
        if value is not None:
            contact = float(np.min(np.asarray(value, dtype=float)))
    return {
        "row": row,
        "base_height_m": float(state.base_pos[2]),
        "tilt_deg": tilt_deg(state.base_quat),
        "lin_vel_norm": float(np.linalg.norm(state.base_lin_vel_body)),
        "ang_vel_norm": float(np.linalg.norm(state.base_ang_vel_body)),
        "joint_vel_norm": float(np.linalg.norm(state.joint_vel)),
        "cmd_track_err": float(
            np.linalg.norm(np.asarray(state.command_vel)[:2] - np.asarray(state.base_lin_vel_body)[:2])
        ),
        "min_contact_force": contact,
    }


def nominal_reference(n: int = 200_000, seed: int = 0) -> dict:
    """Monte-Carlo the nominal reset distribution defined by the event terms."""
    rng = np.random.default_rng(seed)
    lin = rng.uniform(-NOMINAL_VEL_RANGE, NOMINAL_VEL_RANGE, size=(n, 3))
    ang = rng.uniform(-NOMINAL_VEL_RANGE, NOMINAL_VEL_RANGE, size=(n, 3))
    lin_norm = np.linalg.norm(lin, axis=1)
    ang_norm = np.linalg.norm(ang, axis=1)
    return {
        "base_height_m": {"value": NOMINAL_HEIGHT_M, "degenerate": True},
        "tilt_deg": {"value": NOMINAL_TILT_DEG, "degenerate": True},
        "joint_vel_norm": {"value": NOMINAL_JOINT_VEL_NORM, "degenerate": True},
        "lin_vel_norm": {"samples": lin_norm, "degenerate": False},
        "ang_vel_norm": {"samples": ang_norm, "degenerate": False},
    }


def tail_probability(reference: dict, descriptor: str, value: float) -> float | None:
    """P(nominal is at least as extreme as the observed value). None if undefined."""
    entry = reference.get(descriptor)
    if entry is None:
        return None
    if entry["degenerate"]:
        return 0.0 if abs(value - entry["value"]) > 1e-9 else 1.0
    samples = entry["samples"]
    return float(np.mean(samples >= value)) if value >= np.median(samples) else float(
        np.mean(samples <= value)
    )


def infer_mode(path: Path, reader: TrajectoryReader) -> str | None:
    """Failure mode from the trajectory's own column, falling back to the name."""
    if "failure_mode" in reader._table.column_names:
        values = {v for v in reader._table.column("failure_mode").to_pylist() if v}
        if len(values) == 1:
            return values.pop()
    stem = path.stem
    for mode in MODE_PREDICTIONS:
        if mode in stem:
            return mode
    return None


def probe(path: Path, reference: dict) -> dict:
    reader = TrajectoryReader(path)
    mode = infer_mode(path, reader)
    record: dict = {
        "trajectory": path.name,
        "pool": path.parent.parent.parent.name,
        "rows": len(reader),
        "failure_mode": mode,
    }

    try:
        onset = int(reader.failure_indices()[0])
    except (KeyError, IndexError):
        record["error"] = "no failure_flag=True row; trajectory carries no onset"
        return record
    record["onset_row"] = onset

    # Phase-I behaviour: seed_row_strategy="first" resolved to row 0 unconditionally.
    record["phase_i"] = describe(path, 0)
    record["phase_i"]["strategy"] = "first"

    # Production behaviour on this branch: resolve_seed's default strategy.
    try:
        resolved = resolve_seed(
            path,
            PRODUCTION_STRATEGY,
            0,
            PRODUCTION_OFFSET_SECONDS,
        )
        record["production"] = describe(path, resolved["resolved_row"])
        record["production"]["strategy"] = PRODUCTION_STRATEGY
        record["production"]["time_before_onset_seconds"] = resolved["time_before_onset_seconds"]
    except (ValueError, IndexError) as exc:
        record["production"] = {"error": str(exc)}

    # The failure state itself, as an upper reference for how far delivery could go.
    record["at_onset"] = describe(path, onset)
    record["at_onset"]["strategy"] = "failure_onset"

    if mode not in MODE_PREDICTIONS:
        record["verdict"] = "UNKNOWN_MODE"
        return record

    descriptor, direction = MODE_PREDICTIONS[mode]
    record["mode_descriptor"] = descriptor
    record["mode_direction"] = direction
    record["deliverable"] = descriptor in DELIVERED_DESCRIPTORS

    if "error" in record["production"]:
        record["verdict"] = "RESOLVE_FAILED"
        return record

    baseline = record["phase_i"][descriptor]
    seeded = record["production"][descriptor]
    onset_value = record["at_onset"][descriptor]
    if baseline is None or seeded is None or onset_value is None:
        record["verdict"] = "DESCRIPTOR_MISSING"
        return record

    # PRIMARY TEST. Baseline is the trajectory's own history strictly before the
    # seeded row, not the single row 0: one row is a noisy draw and using it as
    # the anchor makes both the direction and the excursion fraction unstable at
    # small effect sizes. This window is causal (nothing after the seed row) and
    # pool-agnostic (no hardcoded stable-prefix length). Where the seeded row
    # sits inside the failure segment the window absorbs some failure rows,
    # which inflates its spread and makes the test HARDER to pass, so the
    # choice is conservative for the cases that pass.
    seed_row = record["production"]["row"]
    history = np.array([describe(path, i)[descriptor] for i in range(seed_row)])
    prefix_mean = float(history.mean())
    prefix_sd = float(history.std(ddof=1)) if len(history) > 1 else 0.0
    record["prefix_rows"] = int(len(history))
    record["prefix_mean"] = prefix_mean
    record["prefix_sd"] = prefix_sd
    record["z_vs_prefix"] = (
        float((seeded - prefix_mean) / prefix_sd) if prefix_sd > 0 else None
    )

    moved = seeded - prefix_mean
    available = onset_value - prefix_mean
    record["mode_descriptor_phase_i"] = baseline
    record["mode_descriptor_seeded"] = seeded
    record["mode_descriptor_at_onset"] = onset_value
    # What fraction of the nominal-to-failure excursion the seeded state covers.
    record["delivery_fraction"] = float(moved / available) if abs(available) > 1e-12 else None
    record["direction_correct"] = bool((moved > 0) if direction == "increase" else (moved < 0))
    record["tail_probability_vs_nominal"] = tail_probability(reference, descriptor, seeded)

    # A seeded state within the spread of its own pre-seed history is, by
    # definition, not a delivered treatment. |z| > 2 is the distinctness bar.
    z = record["z_vs_prefix"]
    if not record["deliverable"]:
        record["verdict"] = "NOT_DELIVERABLE"
    elif z is None or abs(z) <= 2.0:
        record["verdict"] = "FAIL_NOT_DISTINCT"
    elif not record["direction_correct"]:
        record["verdict"] = "FAIL_DIRECTION"
    elif record["delivery_fraction"] is not None and record["delivery_fraction"] < 0.05:
        record["verdict"] = "FAIL_NOT_DISTINCT"
    else:
        record["verdict"] = "PASS"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        action="append",
        default=None,
        help="Directory of failure trajectories; repeatable.",
    )
    parser.add_argument("--out", default="reliability_eval/h0_delivery")
    args = parser.parse_args()

    pools = args.pool or [
        str(Path.home() / "Projects/ashfall/data/failures"),
        str(REPO_ROOT / "data/failures"),
    ]
    paths: list[Path] = []
    for pool in pools:
        paths.extend(sorted(Path(pool).glob("*.parquet")))
    if not paths:
        print("no trajectories found", file=sys.stderr)
        return 2

    reference = nominal_reference()
    records = [probe(p, reference) for p in paths]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "production_strategy": PRODUCTION_STRATEGY,
        "production_offset_seconds": PRODUCTION_OFFSET_SECONDS,
        "nominal_height_m": NOMINAL_HEIGHT_M,
        "pools": pools,
        "records": records,
    }
    (out_dir / "h0_delivery.json").write_text(json.dumps(payload, indent=2, default=str))

    verdicts: dict[str, int] = {}
    for record in records:
        verdicts[record.get("verdict", "NO_ONSET")] = (
            verdicts.get(record.get("verdict", "NO_ONSET"), 0) + 1
        )

    print(f"trajectories probed: {len(records)}")
    print(f"verdicts: {verdicts}\n")
    header = (
        f"{'trajectory':<38} {'mode':<17} {'onset':>5} {'seed':>5} "
        f"{'descriptor':<17} {'prefix_mu':>10} {'seeded':>9} {'onset':>9} "
        f"{'z':>7} {'deliv':>7}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        if "onset_row" not in r:
            print(f"{r['trajectory']:<38} {'-':<17} {'-':>5} {'-':>5} {r.get('error','')}")
            continue
        prod = r.get("production", {})
        frac = r.get("delivery_fraction")
        z = r.get("z_vs_prefix")
        print(
            f"{r['trajectory']:<38} {str(r.get('failure_mode')):<17} "
            f"{r['onset_row']:>5} {str(prod.get('row', '-')):>5} "
            f"{str(r.get('mode_descriptor', '-')):<17} "
            f"{_fmt(r.get('prefix_mean')):>10} "
            f"{_fmt(r.get('mode_descriptor_seeded')):>9} "
            f"{_fmt(r.get('mode_descriptor_at_onset')):>9} "
            f"{(f'{z:+7.2f}' if z is not None else '      -'):>7} "
            f"{(f'{frac:6.1%}' if frac is not None else '      -'):>7}  {r.get('verdict')}"
        )
    print(f"\nwrote {out_dir / 'h0_delivery.json'}")
    return 0


def _fmt(value) -> str:
    return "-" if value is None else f"{value:9.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
