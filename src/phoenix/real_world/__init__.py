"""Real-robot data capture and failure detection.

Runs on the robot side of the loop:

* :mod:`phoenix.real_world.failure_detector`, rule-based classifier that flags
  attitude loss, body collapse, and sustained commanded-velocity tracking
  stalls in real-robot telemetry. Read its module docstring before comparing
  its labels to a simulator termination: they are different ontologies.
* :mod:`phoenix.real_world.trajectory_logger`, appends observations, actions,
  commands, provenance, and detected failure flags into an Apache Parquet file
  that can be re-played in Isaac Sim.
"""

from .failure_detector import (
    MODE_DEFINITIONS,
    FailureDetector,
    FailureEvent,
    FailureMode,
    FailureThresholds,
)
from .trajectory_logger import (
    CAPTURE_SOURCE_HARDWARE,
    CAPTURE_SOURCE_SIM,
    CAPTURE_SOURCE_UNKNOWN,
    TrajectoryLogger,
    TrajectoryStep,
)

__all__ = [
    "CAPTURE_SOURCE_HARDWARE",
    "CAPTURE_SOURCE_SIM",
    "CAPTURE_SOURCE_UNKNOWN",
    "MODE_DEFINITIONS",
    "FailureDetector",
    "FailureEvent",
    "FailureMode",
    "FailureThresholds",
    "TrajectoryLogger",
    "TrajectoryStep",
]
