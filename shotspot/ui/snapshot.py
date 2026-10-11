"""A snapshot of everything derived from the scenario, computed once per redraw.

The field, the math panel and the active mode all read from the same
snapshot, so they can never disagree about a number (or a color).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.scenario import Scenario
from .style import Theme


@dataclass
class Snapshot:
    scenario: Scenario
    positions: np.ndarray  # (N, 2)
    labels: list[str]
    colors: list[str]
    reference: int
    selected: int | None
    theme: Theme
    # Only available once a shot has been placed:
    shot: np.ndarray | None = None
    distances: np.ndarray | None = None  # true |p − sᵢ|
    true_times: np.ndarray | None = None  # t₀ + |p − sᵢ|/c
    times: np.ndarray | None = None  # what the sensors report (true + noise)
    # While the wavefront animation runs: the simulated time, and how much slower than real time it runs.
    clock: float | None = None
    slowdown: float = 1.0

    @classmethod
    def take(cls, scenario: Scenario, selected: int | None, theme: Theme, clock: float | None = None,
             slowdown: float = 1.0) -> Snapshot:
        labels = scenario.labels()
        snap = cls(
            scenario=scenario,
            positions=scenario.positions(),
            labels=labels,
            colors=[theme.sensor_color(label) for label in labels],
            reference=scenario.reference_index(),
            selected=selected if selected is not None and selected < len(labels) else None,
            theme=theme,
        )
        if scenario.shot is not None and labels:
            snap.shot = scenario.shot_position()
            snap.distances = scenario.distances()
            snap.true_times = scenario.true_times()
            snap.times = scenario.measured_times()
            snap.clock, snap.slowdown = clock, slowdown
        return snap

    @property
    def c(self) -> float:
        return self.scenario.c

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def animating(self) -> bool:
        return self.clock is not None

    def reached(self) -> np.ndarray:
        """Which sensors the wavefront has reached by now (all of them when not animating)."""
        if self.clock is None:
            return np.ones(self.n, bool)
        return self.true_times <= self.clock

    @property
    def ready(self) -> bool:
        """A shot exists and enough sensors heard it to solve for (x, y, t₀)."""
        return self.shot is not None and self.n >= 3
