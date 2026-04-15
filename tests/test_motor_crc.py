"""Tests for the Unitree LowCmd CRC and joint-order permutation.

The CRC is validated against a locked set of values computed by the current
implementation. These values also serve as regression sentinels: if someone
edits ``crc32_core`` or the ``LowCmdRaw`` struct layout, this test fails.

The ultimate validator is the GO2 firmware itself, which rejects LowCmd
messages with incorrect CRCs. This test cannot substitute for that, but it
does catch refactor-induced drift cheaply.
"""

from __future__ import annotations

import ctypes

from phoenix.sim2real.motor_crc import (
    PHOENIX_FOR_MOTOR,
    LowCmdRaw,
    build_raw_from_motor_values,
    compute_crc,
    crc32_core,
    phoenix_to_unitree,
    unitree_to_phoenix,
)


def test_struct_size_is_812_bytes() -> None:
    # Matches the C layout on aarch64/x86-64: 24-byte header + 20*36-byte
    # motor array + 4-byte bms + 55 bytes of LEDs/fan/gpio/padding/reserve
    # + 4-byte crc = 812 bytes.
    assert ctypes.sizeof(LowCmdRaw) == 812


def test_crc32_core_known_bitwise_behaviour() -> None:
    # Single-word input exercises the inner bit loop. Against a zero word
    # the CRC still advances because the outer shift always runs.
    assert crc32_core([0]) == 0xC704DD7B
    # With init=0xFFFFFFFF and data=0xFFFFFFFF, the shift-with-poly and the
    # data-bit-xor-poly paths cancel on every iteration, so the register
    # clears to zero.
    assert crc32_core([0xFFFFFFFF]) == 0x00000000


def test_all_zero_lowcmd_has_locked_crc() -> None:
    raw = LowCmdRaw()
    assert compute_crc(raw) == 0x4B5A4880


def test_unitree_stand_pose_matches_locked_crc() -> None:
    # target_pos_2_ from go2_stand_example.cpp — the Unitree-order stand pose
    # with its example's kp=60, kd=5. Already in motor order (FR,FL,RR,RL).
    q = [0.0, 0.67, -1.3] * 4
    raw = build_raw_from_motor_values(q, [60.0] * 12, [5.0] * 12)
    assert compute_crc(raw) == 0xBA838141


def test_phoenix_stand_pose_matches_locked_crc() -> None:
    # deploy.yaml:default_joint_pos in Phoenix order, with conservative
    # first-deploy gains kp=25 kd=0.5 — remapped to Unitree motor order
    # before the CRC is computed.
    phoenix_vec = [
        0.0, 0.0, 0.0, 0.0,    # FL_hip, FR_hip, RL_hip, RR_hip
        0.8, 0.8, 1.0, 1.0,    # FL_thigh, FR_thigh, RL_thigh, RR_thigh
        -1.5, -1.5, -1.5, -1.5,  # FL_calf, FR_calf, RL_calf, RR_calf
    ]
    unitree_vec = phoenix_to_unitree(phoenix_vec)
    raw = build_raw_from_motor_values(unitree_vec, [25.0] * 12, [0.5] * 12)
    assert compute_crc(raw) == 0x67886500


def test_phoenix_to_unitree_permutation() -> None:
    # Inject markers so we can read off which phoenix slot landed where.
    phoenix_vec = [float(i) for i in range(12)]
    expected_unitree = [
        1.0, 5.0, 9.0,    # FR: hip=P1, thigh=P5, calf=P9
        0.0, 4.0, 8.0,    # FL: hip=P0, thigh=P4, calf=P8
        3.0, 7.0, 11.0,   # RR: hip=P3, thigh=P7, calf=P11
        2.0, 6.0, 10.0,   # RL: hip=P2, thigh=P6, calf=P10
    ]
    assert phoenix_to_unitree(phoenix_vec) == expected_unitree


def test_roundtrip_is_identity() -> None:
    phoenix_vec = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 12.2]
    assert unitree_to_phoenix(phoenix_to_unitree(phoenix_vec)) == phoenix_vec


def test_permutation_covers_all_indices_exactly_once() -> None:
    assert sorted(PHOENIX_FOR_MOTOR) == list(range(12))


def test_build_raw_rejects_wrong_length() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_raw_from_motor_values([0.0] * 11, [25.0] * 12, [0.5] * 12)
    with pytest.raises(ValueError):
        build_raw_from_motor_values([0.0] * 12, [25.0] * 13, [0.5] * 12)
    with pytest.raises(ValueError):
        build_raw_from_motor_values([0.0] * 12, [25.0] * 12, [0.5] * 11)


def test_crc_sensitive_to_single_joint_change() -> None:
    # Flipping one motor's target must change the CRC.
    q = [0.0] * 12
    kp = [25.0] * 12
    kd = [0.5] * 12
    baseline = compute_crc(build_raw_from_motor_values(q, kp, kd))
    q_perturbed = list(q)
    q_perturbed[0] = 0.01
    perturbed = compute_crc(build_raw_from_motor_values(q_perturbed, kp, kd))
    assert baseline != perturbed


def test_crc_sensitive_to_gain_change() -> None:
    q = [0.0] * 12
    baseline = compute_crc(build_raw_from_motor_values(q, [25.0] * 12, [0.5] * 12))
    stiffer = compute_crc(build_raw_from_motor_values(q, [26.0] * 12, [0.5] * 12))
    assert baseline != stiffer
