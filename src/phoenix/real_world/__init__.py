"""Real-robot data capture and failure detection.

Runs on the robot side of the loop:

* :mod:`phoenix.real_world.failure_detector` — rule-based classifier that flags
  pitch / height / foot-slip anomalies in real-robot telemetry.
* :mod:`phoenix.real_world.trajectory_logger` — appends observations, actions,
  commands, and detected failure flags into an Apache Parquet file that can
  be re-played in Isaac Sim.
"""

from .failure_detector import FailureDetector, FailureEvent, FailureThresholds
from .trajectory_logger import TrajectoryLogger, TrajectoryStep

__all__ = [
    "FailureDetector",
    "FailureEvent",
    "FailureThresholds",
    "TrajectoryLogger",
    "TrajectoryStep",
]
