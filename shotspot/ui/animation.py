"""The wavefront animation: the sound spreading out from the shot, slowed down enough to watch.

Real sound crosses a 1 km field in about 3 seconds, which is just about watchable,
but a compact array is crossed in a fraction of a second. So the animation runs
SLOWDOWN times slower than real time, adjusted to last between MIN_DURATION and
MAX_DURATION seconds, and says how much it is slowed down.

The animation only keeps a clock: the simulated time t (absolute, like the
arrival times), advanced from the real time elapsed since it started. Each tick
asks the window to redraw, and the window puts the clock in the Snapshot; the
field then draws the wavefront at radius c·(t − t₀) and flashes each sensor as
the wavefront reaches it, and the Per-sensor table fills in arrival times live.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from ..core.scenario import Scenario

SLOWDOWN = 4.0  # animated seconds per real second of sound travel
MIN_DURATION, MAX_DURATION = 3.0, 8.0  # seconds the wave takes to reach the farthest sensor, on screen
OVERSHOOT = 1.08  # keep expanding a little past the farthest sensor
HOLD = 1.0  # seconds to linger on the final frame
FRAME_MS = 33  # about 30 redraws a second (fewer if a redraw takes longer)


class WavefrontAnimation(QObject):
    frame = Signal()  # redraw with the new clock
    finished = Signal()

    def __init__(self, parent: QObject | None = None, now: Callable[[], float] = time.monotonic) -> None:
        super().__init__(parent)
        self._now = now  # injectable, so tests can control time
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self.clock: float | None = None  # simulated time, or None when not animating
        self.slowdown = SLOWDOWN
        self._t0 = 0.0
        self._end = 0.0  # simulated time at which the wave stops expanding
        self._started = 0.0

    @property
    def running(self) -> bool:
        return self.clock is not None

    def start(self, scenario: Scenario) -> bool:
        """Start from the moment of the shot. Returns False if there is no shot to animate."""
        if scenario.shot is None:
            return False
        if scenario.sensors:
            span = float(np.max(scenario.distances())) / scenario.c  # real seconds to the farthest sensor
        else:
            span = float(np.hypot(scenario.width, scenario.height)) / scenario.c
        span = max(span, 1e-3)
        self.slowdown = float(np.clip(SLOWDOWN, MIN_DURATION / span, MAX_DURATION / span))
        self._t0 = scenario.t0
        self._end = scenario.t0 + OVERSHOOT * span
        self._started = self._now()
        self.clock = self._t0
        self._timer.start()
        self.frame.emit()
        return True

    def stop(self) -> None:
        if self.clock is None:
            return
        self._timer.stop()
        self.clock = None
        self.frame.emit()
        self.finished.emit()

    def _tick(self) -> None:
        elapsed = self._now() - self._started
        clock = self._t0 + elapsed / self.slowdown
        if clock > self._end + HOLD / self.slowdown:
            self.stop()
            return
        self.clock = min(clock, self._end)
        self.frame.emit()
