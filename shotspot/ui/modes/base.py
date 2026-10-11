"""The mode interface, and the plain "Field" view that every mode builds on.

A mode is consulted on every redraw, in this order:

    update(snap)        compute whatever it needs from the snapshot; set self.fix
    draw(ax, snap)      add overlays to the field (sensors and shot are drawn for it)
    method_html(snap)   the Method tab's content for the current step

Modes can also contribute widgets to the control strip (`controls`), a panel
under the field (`view`, e.g. the sweep's residual plot), extra legend entries
(`legend_handles`), draggable points on the field (`handles` / `move_handle`),
and a sequence of steps for the Method tab's Prev/Next buttons (`step_count`).
When a mode's own state changes (a slider moves, say) it emits `changed` and
the window redraws.

Every mode draws with the snapshot's theme (`snap.theme`, `snap.colors`), so
it works on the light and the dark field alike.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
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
    t0_is_given = False  # True for a mode that is handed t₀ instead of estimating it
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
        """A small widget for the control strip under the field; created once, shown while active."""
        return None

    def view(self) -> QWidget | None:
        """A panel between the field and the control strip; created once, shown while active."""
        return None

    def legend_handles(self, snap: Snapshot) -> list:
        """Extra matplotlib handles (with labels) for the field's legend."""
        return []

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


class Throttle:
    """For work too slow to redo on every mouse move (a sweep, a heatmap): while the scenario keeps
    changing, `allowed()` says yes at most once per `interval` seconds, and otherwise (re)starts a timer
    that calls `catch_up` once the changes pause. Call `done()` after doing the work."""

    def __init__(self, parent: QObject, interval: float, catch_up: Callable[[], None]) -> None:
        self.interval = interval
        self._last = -np.inf
        self._timer = QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.setInterval(round(interval * 1000))
        self._timer.timeout.connect(catch_up)

    def allowed(self) -> bool:
        if time.monotonic() - self._last >= self.interval:
            return True
        self._timer.start()
        return False

    def done(self) -> None:
        self._last = time.monotonic()


def roadmap(titles: list[str], step: int, numbered: bool = True, shown: set[int] | None = None) -> str:
    """A mode's steps on one line: the current one in bold, the others links ("step:N") that jump to them.

    `shown` limits a long list to some of its steps (the current one is always shown); each run of
    hidden steps becomes "…".
    """
    items, gap = [], False
    for n, title in enumerate(titles):
        if shown is not None and n not in shown and n != step:
            if not gap:
                items.append('<span class="muted">…</span>')
            gap = True
            continue
        gap = False
        text = (f"{n + 1}&nbsp;" if numbered else "") + title.replace(" ", "&nbsp;")
        items.append(f"<b>{text}</b>" if n == step else f'<a href="step:{n}"><span class="muted">{text}</span></a>')
    return "<p>" + ' <span class="muted">·</span> '.join(items) + "</p>"
