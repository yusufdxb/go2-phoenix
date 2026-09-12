"""The policy-node to LowCmd-bridge command message, version 2.

Why the command topic carries more than twelve numbers now
----------------------------------------------------------
The final bridge is the safety boundary, and it is also the only place that sees
what the motors were actually told. A hardware run is only interpretable if one
record holds, per tick, what the policy wanted, what the policy node let through,
and what the bridge finally emitted. Splitting that across two processes' logs
invites exactly the silent disagreement this project keeps finding (the policy
fed zeros while the log held odometry). So the policy node ships its side of the
record inside the command itself, atomically, and the bridge writes one file.

The wire is ``std_msgs/Float64MultiArray``:

* ``layout.dim[0].label`` is :func:`wire_label` of the sender's joint order. The
  bridge builds the same label from ITS config and rejects anything else. That is
  the runtime joint-ORDERING check across the process boundary, and it also
  rejects version-1 senders (an old node publishing twelve bare numbers) instead
  of actuating them.
* ``data`` is a fixed-length float64 vector laid out by :data:`FIELDS`.

``target`` must be finite. Every other field may be NaN, meaning "not applicable
this tick" (for example ``raw_action`` on a default-pose message).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

WIRE_VERSION = 2

#: Message kinds. Only :data:`KIND_POLICY` carries a target the bridge may follow.
KIND_POLICY = 1
#: The policy node is waiting for its first messages and publishes its nominal
#: pose. The bridge HOLDS measured posture instead of driving towards that pose:
#: an unpoliced stand-up before the policy has run is not a behaviour anyone gated.
KIND_STARTUP_DEFAULT = 2
#: The policy node latched an abort. The bridge latches a hold at measured posture.
KIND_ABORT = 3
KINDS = (KIND_POLICY, KIND_STARTUP_DEFAULT, KIND_ABORT)

#: ``(name, width)`` in wire order. ``None`` width means one per joint (12).
FIELDS: tuple[tuple[str, int | None], ...] = (
    ("wire_version", 1),
    ("seq", 1),
    ("kind", 1),
    ("abort_code", 1),
    ("target", None),  # policy order, AFTER the policy node's slew clip
    ("requested_target", None),  # policy order, default_q + action_scale * action, BEFORE clip
    ("raw_action", None),  # policy order, the ONNX output
    ("q_policy", None),  # policy order, measured q the policy node clipped against
    ("base_lin_vel_fed", 3),  # what obs dims 0..2 actually held
    ("cmd_vel_received", 3),  # last /cmd_vel as received, NaN if never
    ("velocity_command_fed", 3),  # what obs dims 9..11 actually held
    ("obs_source_code", 1),  # OBS_SOURCE_CODES
    ("stand_only", 1),  # 1.0 or 0.0
    ("imu_age_s", 1),
    ("joint_state_age_s", 1),
    ("estop_age_s", 1),  # as seen by the policy node
    ("estop_value", 1),  # as seen by the policy node, NaN if never
    ("cmd_vel_age_s", 1),
    ("roll_rad", 1),
    ("pitch_rad", 1),
    ("policy_elapsed_s", 1),  # since node start
    ("authority_elapsed_s", 1),  # since the first RUN_POLICY tick, NaN before
)

OBS_SOURCE_CODES: dict[str, int] = {"zeros": 0, "odom": 1}

#: Abort reasons, by stable code. A reason not listed maps to ``other``; the full
#: string is still logged by the policy node.
ABORT_REASONS: tuple[str, ...] = (
    "none",
    "max_runtime",
    "authority_window_complete",
    "external_estop",
    "estop_publisher_missing",
    "estop_heartbeat_stale",
    "sensor_missing",
    "sensor_stale",
    "nan_in_joint_state",
    "nan_in_imu",
    "attitude",
    "first_message_timeout",
    "base_lin_vel_unavailable",
    "walking_command_blocked_stand_only",
    "node_shutdown",
    "other",
)


class WireError(ValueError):
    """A command that must not be actuated. ``code`` is a stable short reason."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


def field_slices(n_joints: int = 12) -> dict[str, slice]:
    out: dict[str, slice] = {}
    start = 0
    for name, width in FIELDS:
        w = n_joints if width is None else width
        out[name] = slice(start, start + w)
        start += w
    return out


def wire_length(n_joints: int = 12) -> int:
    return sum(n_joints if w is None else w for _, w in FIELDS)


def wire_label(joint_order: Sequence[str]) -> str:
    order = ",".join(joint_order)
    return f"phoenix_cmd/v{WIRE_VERSION};len={wire_length(len(joint_order))};order={order}"


def abort_code(reason: str | None) -> int:
    """Stable code for an abort reason string. Prefix match, so detail suffixes survive."""
    if not reason:
        return 0
    for index, name in enumerate(ABORT_REASONS):
        if name not in ("none", "other") and reason.startswith(name):
            return index
    return ABORT_REASONS.index("other")


def encode(
    joint_order: Sequence[str],
    *,
    seq: int,
    kind: int,
    target: Sequence[float],
    abort_reason: str | None = None,
    **fields: float | Sequence[float] | None,
) -> tuple[str, list[float]]:
    """Build ``(label, data)``. Unknown field names raise; omitted fields are NaN."""
    n = len(joint_order)
    slices = field_slices(n)
    data = np.full(wire_length(n), np.nan, dtype=np.float64)
    unknown = set(fields) - set(slices)
    if unknown:
        raise KeyError(f"unknown wire fields {sorted(unknown)}")
    if kind not in KINDS:
        raise ValueError(f"unknown command kind {kind}")
    data[slices["wire_version"]] = WIRE_VERSION
    data[slices["seq"]] = float(seq)
    data[slices["kind"]] = float(kind)
    data[slices["abort_code"]] = float(abort_code(abort_reason))
    data[slices["target"]] = np.asarray(target, dtype=np.float64).reshape(n)
    for name, value in fields.items():
        if value is None:
            continue
        sl = slices[name]
        data[sl] = np.asarray(value, dtype=np.float64).reshape(sl.stop - sl.start)
    return wire_label(joint_order), data.tolist()


@dataclass(frozen=True)
class DecodedCommand:
    seq: int
    kind: int
    abort_reason: str
    fields: dict[str, np.ndarray]

    @property
    def target(self) -> np.ndarray:
        return self.fields["target"]

    def telemetry(self) -> dict[str, object]:
        """JSON-ready view, NaN as ``None``, arrays as lists, in policy joint order."""
        out: dict[str, object] = {
            "seq": self.seq,
            "kind": self.kind,
            "abort_reason": self.abort_reason,
        }
        for name, value in self.fields.items():
            if name in ("wire_version", "seq", "kind", "abort_code"):
                continue
            arr = [None if not np.isfinite(v) else float(v) for v in value.tolist()]
            out[name] = arr[0] if value.size == 1 else arr
        return out


def decode(label: str, data: Sequence[float], expected_label: str) -> DecodedCommand:
    """Validate and unpack a command. Raises :class:`WireError` for anything unsafe."""
    if label != expected_label:
        raise WireError(
            "label_mismatch",
            f"got {label!r}, expected {expected_label!r} (wrong wire version or joint order)",
        )
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    n_joints = len(expected_label.split("order=", 1)[1].split(","))
    if arr.size != wire_length(n_joints):
        raise WireError(
            "length_mismatch", f"got {arr.size} values, expected {wire_length(n_joints)}"
        )
    slices = field_slices(n_joints)
    fields = {name: arr[sl].copy() for name, sl in slices.items()}
    if fields["wire_version"][0] != WIRE_VERSION:
        raise WireError("version_mismatch", f"wire_version {fields['wire_version'][0]}")
    kind_value = fields["kind"][0]
    if not np.isfinite(kind_value) or int(kind_value) not in KINDS:
        raise WireError("unknown_kind", f"kind {kind_value}")
    seq_value = fields["seq"][0]
    if not np.isfinite(seq_value) or seq_value < 0 or seq_value != int(seq_value):
        raise WireError("bad_seq", f"seq {seq_value}")
    if not np.all(np.isfinite(fields["target"])):
        raise WireError("non_finite_target", f"target {fields['target'].tolist()}")
    code = fields["abort_code"][0]
    reason = (
        ABORT_REASONS[int(code)]
        if np.isfinite(code) and 0 <= int(code) < len(ABORT_REASONS)
        else "other"
    )
    return DecodedCommand(
        seq=int(seq_value), kind=int(kind_value), abort_reason=reason, fields=fields
    )


def obs_source_code(source: str) -> float:
    return float(OBS_SOURCE_CODES[source])


def source_name(code: float) -> str | None:
    for name, value in OBS_SOURCE_CODES.items():
        if np.isfinite(code) and int(code) == value:
            return name
    return None


def _as_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return value


__all__ = [
    "ABORT_REASONS",
    "FIELDS",
    "KINDS",
    "KIND_ABORT",
    "KIND_POLICY",
    "KIND_STARTUP_DEFAULT",
    "OBS_SOURCE_CODES",
    "WIRE_VERSION",
    "DecodedCommand",
    "WireError",
    "abort_code",
    "decode",
    "encode",
    "field_slices",
    "obs_source_code",
    "source_name",
    "wire_label",
    "wire_length",
]
