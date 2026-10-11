"""The mode interface, and the plain "Field" view that every mode builds on.

A mode is consulted on every redraw, in this order:

    update(snap)        compute whatever it needs from the snapshot; set self.fix
    draw(ax, snap)      add overlays to the field (sensors and shot are drawn for it)
    method_html(snap)   the Method tab's content for the current step

Modes can also contribute widgets to the control strip (`controls`), draggable
points on the field (`handles` / `move_handle`), and a sequence of steps
for the Method tab's Prev/Next buttons (`step_count`). When a mode's own state
changes (a slider moves, say) it emits `changed` and the window redraws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget

from ...core.accuracy import Fix, locate
from ..snapshot import Snapshot


@dataclass
class Handle:
    """A point on the field the user can drag, owned by a mode."""

    name: str
    position: np.ndarray
    color: str
    marker: str = "D"


class Mode(QObject):
    changed = Signal()

    title = "Field"
    estimate_source = "closed form, polished by Gauss-Newton"
    summary = (
        "Each sensor knows only where it is and when it heard the bang. Where the shot was "
        "and the moment it was fired are both unknown: three unknowns (x, y, t₀), so at "
        "least three sensors must hear it. Pick a mode to see different ways of solving for them."
    )

    def __init__(self) -> None:
        super().__init__()
        self.fix: Fix | None = None
        self.step = 0

    # ---- computation and drawing -----------------------------------------

    def update(self, snap: Snapshot) -> None:
        self.fix = locate(snap.positions, snap.times, snap.c, snap.reference) if snap.ready else None

    def draw(self, ax, snap: Snapshot) -> None:
        pass

    def method_html(self, snap: Snapshot) -> str:
        return (
            "<h2>Method</h2><p>This is the plain field view: sensors, shot and the "
            "located position. Choose a mode from the toolbar to see the math behind "
            "one way of solving the problem, step by step.</p>"
        )

    # ---- optional extras -------------------------------------------------

    def controls(self) -> QWidget | None:
        return None

    def handles(self, snap: Snapshot) -> list[Handle]:
        return []

    def move_handle(self, name: str, x: float, y: float) -> None:
        pass

    def step_count(self, snap: Snapshot) -> int:
        """Number of Method steps; 0 hides the Prev/Next buttons."""
        return 0

    def set_step(self, step: int) -> None:
        self.step = step
        self.changed.emit()
