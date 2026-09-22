"""Live DETECT: the health vector updated once per window while the robot runs.

:class:`LiveMonitor` consumes bridge tick records one at a time, in the order the
bridge writes them, and emits a health report every ``window_ticks`` records. It
runs the same code as the offline path (``tracking_pairs`` -> ``window_stats`` ->
``authority_ratio`` -> :class:`~phoenix.monitor.health.HealthMonitor`), on one window
at a time, so a live run and an offline re-analysis of the same file agree.

It deliberately lives outside the final actuator gate: it only reads telemetry the
bridge has already written (``scripts/phoenix_live_monitor.py`` tails
``bridge.jsonl``), it can never change a motor command, and it failing or lagging
cannot affect the robot.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .health import DEFAULT_PERSISTENCE, HealthMonitor, JointHealth, PersistenceConfig
from .layers import from_records, tracking_pairs
from .residual import Baseline, WindowConfig, authority_ratio, window_stats


class LiveMonitor:
    def __init__(
        self, baseline: Baseline, persistence: PersistenceConfig = DEFAULT_PERSISTENCE
    ) -> None:
        self.cfg = WindowConfig(**baseline.window) if baseline.window else WindowConfig()
        self.baseline = baseline
        self.health = HealthMonitor(baseline, persistence)
        # One extra record so the last tick of a window can pair with the next one.
        self._buf: list[Mapping[str, Any]] = []
        self.last: list[JointHealth] = self.health.report()

    def feed(self, record: Mapping[str, Any]) -> list[JointHealth] | None:
        """Add one bridge record; return a new report when a window completes."""
        if record.get("record", "tick") != "tick":
            return None
        self._buf.append(record)
        w = self.cfg.window_ticks
        if len(self._buf) < w + 1:
            return None
        window, self._buf = self._buf[: w + 1], self._buf[w:]
        st = window_stats(tracking_pairs(from_records(window)), self.cfg)
        if st.n_windows == 0:
            return None
        s = authority_ratio(st, self.baseline, self.cfg)
        self.last = self.health.update(s[0], st.torque_gain_ratio[0])
        return self.last


__all__ = ["LiveMonitor"]
