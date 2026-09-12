"""Policy-node to bridge wire: ordering label, versioning, fail-closed decoding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phoenix.sim2real.command_wire import (
    ABORT_REASONS,
    KIND_ABORT,
    KIND_POLICY,
    WireError,
    abort_code,
    decode,
    encode,
    field_slices,
    wire_label,
    wire_length,
)
from phoenix.sim2real.go2_model import POLICY_JOINT_ORDER

ORDER = POLICY_JOINT_ORDER
LABEL = wire_label(ORDER)


def _policy_msg(**over):
    fields = dict(
        seq=7,
        kind=KIND_POLICY,
        target=np.linspace(-1, 1, 12),
        requested_target=np.linspace(-1.1, 1.1, 12),
        raw_action=np.arange(12) * 0.1,
        q_policy=np.zeros(12),
        base_lin_vel_fed=[0.0, 0.0, 0.0],
        velocity_command_fed=[0.0, 0.0, 0.0],
        obs_source_code=0.0,
        stand_only=1.0,
    )
    fields.update(over)
    return encode(ORDER, **fields)


def test_roundtrip_preserves_every_field() -> None:
    label, data = _policy_msg()
    assert label == LABEL
    # 4 header + 4 x 12 joint vectors + 3 x 3 velocity vectors + 11 scalars.
    assert len(data) == wire_length() == 72
    cmd = decode(label, data, LABEL)
    assert cmd.seq == 7 and cmd.kind == KIND_POLICY
    assert np.allclose(cmd.target, np.linspace(-1, 1, 12))
    assert np.allclose(cmd.fields["raw_action"], np.arange(12) * 0.1)
    assert math.isnan(cmd.fields["imu_age_s"][0])  # omitted -> NaN, allowed


def test_label_carries_the_joint_order_and_rejects_a_swap() -> None:
    swapped = list(ORDER)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    label, data = encode(swapped, seq=1, kind=KIND_POLICY, target=np.zeros(12))
    with pytest.raises(WireError) as exc:
        decode(label, data, LABEL)
    assert exc.value.code == "label_mismatch"


def test_version_one_twelve_float_command_is_rejected() -> None:
    with pytest.raises(WireError) as exc:
        decode("", [0.0] * 12, LABEL)
    assert exc.value.code == "label_mismatch"


def test_length_mismatch_is_rejected() -> None:
    _, data = _policy_msg()
    with pytest.raises(WireError) as exc:
        decode(LABEL, data[:-1], LABEL)
    assert exc.value.code == "length_mismatch"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("joint", range(12))
def test_non_finite_target_on_any_joint_is_rejected(bad: float, joint: int) -> None:
    label, data = _policy_msg()
    data[field_slices()["target"].start + joint] = bad
    with pytest.raises(WireError) as exc:
        decode(label, data, LABEL)
    assert exc.value.code == "non_finite_target"


def test_unknown_kind_and_bad_seq_are_rejected() -> None:
    label, data = _policy_msg()
    sl = field_slices()
    bad_kind = list(data)
    bad_kind[sl["kind"].start] = 9.0
    with pytest.raises(WireError):
        decode(label, bad_kind, LABEL)
    bad_seq = list(data)
    bad_seq[sl["seq"].start] = 1.5
    with pytest.raises(WireError):
        decode(label, bad_seq, LABEL)
    bad_version = list(data)
    bad_version[sl["wire_version"].start] = 1.0
    with pytest.raises(WireError):
        decode(label, bad_version, LABEL)


@pytest.mark.parametrize(
    "reason,expected",
    [
        (None, "none"),
        ("max_runtime", "max_runtime"),
        ("attitude pitch=0.91 roll=0.02", "attitude"),
        ("first_message_timeout_imu,joint_state", "first_message_timeout"),
        ("base_lin_vel_unavailable: stale", "base_lin_vel_unavailable"),
        ("something_new", "other"),
    ],
)
def test_abort_reason_codes(reason, expected) -> None:
    assert ABORT_REASONS[abort_code(reason)] == expected
    label, data = encode(ORDER, seq=3, kind=KIND_ABORT, target=np.zeros(12), abort_reason=reason)
    assert decode(label, data, LABEL).abort_reason == expected


def test_telemetry_view_is_json_ready() -> None:
    label, data = _policy_msg()
    view = decode(label, data, LABEL).telemetry()
    assert view["imu_age_s"] is None
    assert view["stand_only"] == 1.0
    assert len(view["target"]) == 12


def test_encode_rejects_unknown_field() -> None:
    with pytest.raises(KeyError):
        encode(ORDER, seq=1, kind=KIND_POLICY, target=np.zeros(12), made_up=1.0)
