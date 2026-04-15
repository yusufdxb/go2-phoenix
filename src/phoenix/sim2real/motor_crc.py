"""Unitree GO2 LowCmd CRC helper.

Ports ``get_crc`` and ``crc32_core`` from::

    ~/go2_ws/src/unitree_ros2/example/src/src/common/motor_crc.cpp

to pure Python. The GO2 firmware rejects LowCmd messages whose ``crc`` field
does not match this exact byte-layout / polynomial, so the on-robot behaviour
is the ultimate test of correctness. The ctypes structs mirror the packed C
layout (mode: u8 + 3 pad; then five floats + 3×u32 reserve per motor,
36 bytes each; array of 20 motors).

This module is intentionally ROS-free so it can be imported on the PC for
unit tests without an rclpy install.
"""

from __future__ import annotations

import ctypes
import struct
from typing import Iterable, Sequence


# --- joint index constants (matching motor_crc.h) ---------------------------
FR_0, FR_1, FR_2 = 0, 1, 2
FL_0, FL_1, FL_2 = 3, 4, 5
RR_0, RR_1, RR_2 = 6, 7, 8
RL_0, RL_1, RL_2 = 9, 10, 11


# Phoenix policy (deploy.yaml:joint_order) emits indices:
#   0:FL_hip 1:FR_hip 2:RL_hip 3:RR_hip
#   4:FL_thigh 5:FR_thigh 6:RL_thigh 7:RR_thigh
#   8:FL_calf 9:FR_calf 10:RL_calf 11:RR_calf
#
# Unitree motor_cmd[k] expects a per-leg grouping with FR first:
#   motor_cmd[0..2] = FR (hip, thigh, calf)
#   motor_cmd[3..5] = FL (hip, thigh, calf)
#   motor_cmd[6..8] = RR (hip, thigh, calf)
#   motor_cmd[9..11] = RL (hip, thigh, calf)
#
# So for motor_cmd[k] we pull phoenix[PHOENIX_FOR_MOTOR[k]].
PHOENIX_FOR_MOTOR: tuple[int, ...] = (
    1, 5, 9,    # FR: hip, thigh, calf
    0, 4, 8,    # FL: hip, thigh, calf
    3, 7, 11,   # RR: hip, thigh, calf
    2, 6, 10,   # RL: hip, thigh, calf
)


# --- ctypes mirror of the packed C LowCmd struct ----------------------------
class _MotorCmd(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint8),
        ("q", ctypes.c_float),
        ("dq", ctypes.c_float),
        ("tau", ctypes.c_float),
        ("Kp", ctypes.c_float),
        ("Kd", ctypes.c_float),
        ("reserve", ctypes.c_uint32 * 3),
    ]


class _BmsCmd(ctypes.Structure):
    _fields_ = [
        ("off", ctypes.c_uint8),
        ("reserve", ctypes.c_uint8 * 3),
    ]


class LowCmdRaw(ctypes.Structure):
    """Byte-for-byte mirror of the C ``LowCmd`` struct. Public so tests can
    construct one explicitly without a ROS message. The ``crc`` field is the
    last 4 bytes and is excluded from the CRC computation."""

    _fields_ = [
        ("head", ctypes.c_uint8 * 2),
        ("levelFlag", ctypes.c_uint8),
        ("frameReserve", ctypes.c_uint8),
        ("SN", ctypes.c_uint32 * 2),
        ("version", ctypes.c_uint32 * 2),
        ("bandWidth", ctypes.c_uint16),
        ("motorCmd", _MotorCmd * 20),
        ("bms", _BmsCmd),
        ("wirelessRemote", ctypes.c_uint8 * 40),
        ("led", ctypes.c_uint8 * 12),
        ("fan", ctypes.c_uint8 * 2),
        ("gpio", ctypes.c_uint8),
        ("reserve", ctypes.c_uint32),
        ("crc", ctypes.c_uint32),
    ]


def crc32_core(words: Iterable[int]) -> int:
    """Unitree custom CRC32 (non-standard: poly 0x04C11DB7, init 0xFFFFFFFF,
    no reflection, no final XOR). Processes a stream of uint32 words."""
    crc = 0xFFFFFFFF
    poly = 0x04C11DB7
    mask = 0xFFFFFFFF
    for data in words:
        xbit = 1 << 31
        while xbit:
            if crc & 0x80000000:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
            if data & xbit:
                crc ^= poly
            xbit >>= 1
    return crc & mask


def compute_crc(raw: LowCmdRaw) -> int:
    """Compute the CRC over all bytes of ``raw`` except the final ``crc``
    uint32. Mirrors ``get_crc``'s ``(sizeof(LowCmd) >> 2) - 1`` length."""
    size = ctypes.sizeof(LowCmdRaw)
    assert size == 812, f"LowCmd struct size changed: {size} (expected 812)"
    buf = bytes(raw)[: size - 4]
    words = struct.unpack(f"<{len(buf) // 4}I", buf)
    return crc32_core(words)


def build_raw_from_motor_values(
    q: Sequence[float],
    kp: Sequence[float],
    kd: Sequence[float],
    *,
    mode: int = 0x01,
    level_flag: int = 0xFF,
    head: tuple[int, int] = (0xFE, 0xEF),
    dq: Sequence[float] | None = None,
    tau: Sequence[float] | None = None,
) -> LowCmdRaw:
    """Construct a LowCmdRaw given 12 per-motor targets already in Unitree
    motor order (FR, FL, RR, RL × hip/thigh/calf). Unused motor_cmd[12..19]
    slots (arm/auxiliary) are left at zero. ``dq`` and ``tau`` default to 0.
    """
    if len(q) != 12 or len(kp) != 12 or len(kd) != 12:
        raise ValueError("q/kp/kd must have exactly 12 elements")
    dq_ = [0.0] * 12 if dq is None else list(dq)
    tau_ = [0.0] * 12 if tau is None else list(tau)
    if len(dq_) != 12 or len(tau_) != 12:
        raise ValueError("dq/tau must have exactly 12 elements")

    raw = LowCmdRaw()
    raw.head[0] = head[0]
    raw.head[1] = head[1]
    raw.levelFlag = level_flag
    for i in range(12):
        raw.motorCmd[i].mode = mode
        raw.motorCmd[i].q = float(q[i])
        raw.motorCmd[i].dq = float(dq_[i])
        raw.motorCmd[i].tau = float(tau_[i])
        raw.motorCmd[i].Kp = float(kp[i])
        raw.motorCmd[i].Kd = float(kd[i])
    return raw


def phoenix_to_unitree(phoenix_vec: Sequence[float]) -> list[float]:
    """Reorder 12 floats from Phoenix joint order to Unitree motor order."""
    if len(phoenix_vec) != 12:
        raise ValueError(f"expected 12 elements, got {len(phoenix_vec)}")
    return [float(phoenix_vec[PHOENIX_FOR_MOTOR[k]]) for k in range(12)]


def unitree_to_phoenix(unitree_vec: Sequence[float]) -> list[float]:
    """Inverse of phoenix_to_unitree — reorder 12 floats from Unitree motor
    order back to Phoenix joint order."""
    if len(unitree_vec) != 12:
        raise ValueError(f"expected 12 elements, got {len(unitree_vec)}")
    out = [0.0] * 12
    for k, src in enumerate(PHOENIX_FOR_MOTOR):
        out[src] = float(unitree_vec[k])
    return out
