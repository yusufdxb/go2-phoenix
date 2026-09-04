"""Per-episode evaluation telemetry for Phoenix rollouts.

Phoenix's evaluator (:mod:`phoenix.training.evaluate`) builds per-episode
information inside the rollout loop and then averages it away when it
constructs :class:`~phoenix.training.evaluate.RolloutMetrics`, leaving about
a dozen scalars per evaluation cell in ``metrics_*.json``. Those scalars
cannot answer a per-failure-mode recurrence question, because the identity
of the individual episode (its seed, its env slot, when it died, how badly
it was tracking at the time) is gone by the time the JSON is written.

This module is the additive fix. It defines a typed, versioned, columnar
record with one row per finished episode, written next to the existing
metrics JSON. The metrics JSON schema and values are untouched, so the
Phase-I n=11 numbers keep reproducing bit for bit.

Design rules:

* Every field name carries its unit, and :data:`UNITS` maps field name to
  unit for the fields where a suffix would be awkward.
* :data:`SCHEMA_VERSION` is stamped both in the Arrow schema metadata and
  in a per-row column, so a single row read in isolation still identifies
  its own version.
* The loader is fail-closed. A version mismatch or a missing required
  column raises; nothing is silently defaulted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "1.0.0"

#: Arrow metadata key holding :data:`SCHEMA_VERSION`.
SCHEMA_VERSION_KEY = b"phoenix_episode_records_schema_version"

#: Arrow metadata key holding a JSON dump of :data:`UNITS`.
UNITS_KEY = b"phoenix_episode_records_units"

#: Sentinel written into ``time_to_failure_*`` for episodes that did not fail.
#: Chosen over null so that a reader cannot mistake "no failure" for
#: "column absent" or "value dropped".
NO_FAILURE = -1

#: Unit of every field. Fields whose name already ends in the unit are
#: repeated here so the map is total: a consumer never has to guess.
UNITS: dict[str, str] = {
    "schema_version": "semver",
    "run_id": "identifier",
    "episode_id": "count",
    "seed": "identifier",
    "env_index": "index",
    "terrain_id": "identifier",
    "challenge_id": "identifier",
    "success": "bool",
    "termination_reason": "enum",
    "time_to_failure_steps": "env_steps",
    "time_to_failure_s": "seconds",
    "episode_return": "reward",
    "episode_length_steps": "env_steps",
    "episode_length_s": "seconds",
    "mean_lin_vel_error_mps": "m/s",
    "max_lin_vel_error_mps": "m/s",
    "mean_ang_vel_error_radps": "rad/s",
    "max_ang_vel_error_radps": "rad/s",
    "control_dt_s": "seconds",
    "policy_path": "filesystem_path",
    "policy_sha256": "hex_digest",
}


@dataclass
class EpisodeRecord:
    """One finished evaluation episode.

    ``time_to_failure_steps`` / ``time_to_failure_s`` are :data:`NO_FAILURE`
    for a successful (timed-out) episode and equal to the episode length for
    a failed one, since a Phoenix episode terminates at its failure.
    """

    run_id: str
    episode_id: int
    seed: int
    env_index: int
    terrain_id: str
    challenge_id: str
    success: bool
    termination_reason: str
    time_to_failure_steps: int
    time_to_failure_s: float
    episode_return: float
    episode_length_steps: int
    episode_length_s: float
    mean_lin_vel_error_mps: float
    max_lin_vel_error_mps: float
    mean_ang_vel_error_radps: float
    max_ang_vel_error_radps: float
    control_dt_s: float
    policy_path: str
    policy_sha256: str
    schema_version: str = SCHEMA_VERSION


ARROW_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("run_id", pa.string()),
        ("episode_id", pa.int64()),
        ("seed", pa.int64()),
        ("env_index", pa.int64()),
        ("terrain_id", pa.string()),
        ("challenge_id", pa.string()),
        ("success", pa.bool_()),
        ("termination_reason", pa.string()),
        ("time_to_failure_steps", pa.int64()),
        ("time_to_failure_s", pa.float64()),
        ("episode_return", pa.float64()),
        ("episode_length_steps", pa.int64()),
        ("episode_length_s", pa.float64()),
        ("mean_lin_vel_error_mps", pa.float64()),
        ("max_lin_vel_error_mps", pa.float64()),
        ("mean_ang_vel_error_radps", pa.float64()),
        ("max_ang_vel_error_radps", pa.float64()),
        ("control_dt_s", pa.float64()),
        ("policy_path", pa.string()),
        ("policy_sha256", pa.string()),
    ],
    metadata={
        SCHEMA_VERSION_KEY: SCHEMA_VERSION.encode(),
        UNITS_KEY: json.dumps(UNITS, sort_keys=True).encode(),
    },
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(ARROW_SCHEMA.names)


class EpisodeRecordSchemaError(ValueError):
    """Raised when an episode-record artifact fails validation.

    Never caught internally to fall back on a default: a malformed or
    wrong-version artifact is an error, not a shrug.
    """


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, streamed so checkpoints do not blow RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_episode_record(
    *,
    run_id: str,
    episode_id: int,
    seed: int,
    env_index: int,
    terrain_id: str,
    challenge_id: str,
    timed_out: bool,
    termination_reason: str,
    episode_return: float,
    episode_length_steps: int,
    control_dt_s: float,
    lin_err_sum_mps: float,
    lin_err_max_mps: float,
    ang_err_sum_radps: float,
    ang_err_max_radps: float,
    tracking_steps: int,
    policy_path: str,
    policy_sha256: str,
) -> EpisodeRecord:
    """Assemble one record from the accumulators the rollout loop carries.

    ``timed_out`` is the Isaac Lab ``time_outs`` flag, which is Phoenix's
    existing definition of a successful episode; it is passed in rather
    than re-derived so this helper and the evaluator cannot disagree.
    """
    success = bool(timed_out)
    steps = int(episode_length_steps)
    denom = max(int(tracking_steps), 1)
    return EpisodeRecord(
        run_id=str(run_id),
        episode_id=int(episode_id),
        seed=int(seed),
        env_index=int(env_index),
        terrain_id=str(terrain_id),
        challenge_id=str(challenge_id),
        success=success,
        termination_reason=str(termination_reason),
        time_to_failure_steps=NO_FAILURE if success else steps,
        time_to_failure_s=(float(NO_FAILURE) if success else float(steps) * float(control_dt_s)),
        episode_return=float(episode_return),
        episode_length_steps=steps,
        episode_length_s=float(steps) * float(control_dt_s),
        mean_lin_vel_error_mps=float(lin_err_sum_mps) / denom,
        max_lin_vel_error_mps=float(lin_err_max_mps),
        mean_ang_vel_error_radps=float(ang_err_sum_radps) / denom,
        max_ang_vel_error_radps=float(ang_err_max_radps),
        control_dt_s=float(control_dt_s),
        policy_path=str(policy_path),
        policy_sha256=str(policy_sha256),
    )


def records_to_table(records: list[EpisodeRecord]) -> pa.Table:
    """Convert records to an Arrow table under :data:`ARROW_SCHEMA`."""
    rows = [asdict(r) for r in records]
    for row in rows:
        # Stamp the writer's version rather than trusting a hand-built record.
        row["schema_version"] = SCHEMA_VERSION
    return pa.Table.from_pylist(rows, schema=ARROW_SCHEMA)


def write_episode_records(path: str | Path, records: list[EpisodeRecord]) -> Path:
    """Write ``records`` to a parquet file and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(records_to_table(records), out, compression="zstd")
    return out


def validate_table(table: pa.Table, *, source: str = "<table>") -> None:
    """Fail-closed schema validation of an episode-record table.

    Raises :class:`EpisodeRecordSchemaError` on a missing required column or
    on any schema version other than :data:`SCHEMA_VERSION`.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in table.schema.names]
    if missing:
        raise EpisodeRecordSchemaError(
            f"{source}: missing required column(s): {', '.join(sorted(missing))}"
        )

    versions: set[str] = set()
    meta = table.schema.metadata or {}
    if SCHEMA_VERSION_KEY in meta:
        versions.add(meta[SCHEMA_VERSION_KEY].decode())
    for value in table.column("schema_version").to_pylist():
        if value is None:
            raise EpisodeRecordSchemaError(f"{source}: null schema_version in a row")
        versions.add(str(value))

    if not versions:
        raise EpisodeRecordSchemaError(f"{source}: no schema version found")
    unexpected = sorted(v for v in versions if v != SCHEMA_VERSION)
    if unexpected:
        raise EpisodeRecordSchemaError(
            f"{source}: schema version mismatch, reader expects {SCHEMA_VERSION!r} "
            f"but artifact declares {unexpected!r}"
        )


def table_units(table: pa.Table, *, source: str = "<table>") -> dict[str, str]:
    """Return the units map stamped into the artifact.

    Raises if the artifact carries no units map: a consumer that silently
    invented units would be exactly the failure this module exists to stop.
    """
    meta = table.schema.metadata or {}
    if UNITS_KEY not in meta:
        raise EpisodeRecordSchemaError(f"{source}: units map missing from schema metadata")
    return json.loads(meta[UNITS_KEY].decode())


def load_episode_records(path: str | Path) -> list[EpisodeRecord]:
    """Load and validate an episode-record parquet into typed records."""
    src = Path(path)
    if not src.exists():
        raise EpisodeRecordSchemaError(f"{src}: episode-record artifact does not exist")
    table = pq.read_table(src)
    validate_table(table, source=str(src))
    _ = table_units(table, source=str(src))
    field_names = {f.name for f in fields(EpisodeRecord)}
    out: list[EpisodeRecord] = []
    for row in table.to_pylist():
        out.append(EpisodeRecord(**{k: v for k, v in row.items() if k in field_names}))
    return out


__all__ = [
    "ARROW_SCHEMA",
    "NO_FAILURE",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_KEY",
    "UNITS",
    "UNITS_KEY",
    "EpisodeRecord",
    "EpisodeRecordSchemaError",
    "build_episode_record",
    "load_episode_records",
    "records_to_table",
    "sha256_file",
    "table_units",
    "validate_table",
    "write_episode_records",
]
