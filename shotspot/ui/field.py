"""The field: a matplotlib canvas showing sensors and the shot, with mouse editing.

Mouse map:
    left-click empty space         add a sensor (keep holding to drag it into place)
    left-drag a sensor/shot/handle move it
    left-click a sensor            select it (click again to deselect)
    right-click a sensor           delete it
    shift-click, or Fire mode      place the shot (on empty space)
    scroll wheel                   zoom around the cursor
    middle-drag                    pan

The canvas never edits the scenario itself. It emits signals, the main
window updates the scenario, and then calls `render` with a fresh snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, Signal  # import the Qt binding before matplotlib picks one
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .fmt import usub
from .modes.base import Handle, Mode
from .snapshot import Snapshot
from .style import AXIS, GRID, INK, INK_SECONDARY, MUTED, SURFACE

PICK_RADIUS_PX = 12
DRAG_THRESHOLD_PX = 4
ZOOM_STEP = 1.2
FIELD_MARGIN = 0.04  # fraction of the field shown beyond its edges


@dataclass
class _Drag:
    kind: str  # "sensor", "shot", "handle" or "pan"
    index: int = -1  # sensor index
    name: str = ""  # handle name
    press_px: tuple[float, float] = (0.0, 0.0)
    moved: bool = False
    new: bool = False  # a sensor this very press created, so releasing it isn't a "click"
    start_view: tuple | None = None  # for panning


class FieldCanvas(FigureCanvasQTAgg):
    sensorAdded = Signal(float, float)
    sensorMoved = Signal(int, float, float)
    sensorRemoved = Signal(int)
    sensorClicked = Signal(int)
    shotMoved = Signal(float, float)
    handleMoved = Signal(str, float, float)
    cursorMoved = Signal(float, float)

    def __init__(self) -> None:
        fig = Figure(facecolor=SURFACE)
        super().__init__(fig)
        self.ax = fig.add_subplot()
        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.06, top=0.95)
        self.fire_mode = False
        self._field_size = (1000.0, 1000.0)
        self._view: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._sensors = np.empty((0, 2))
        self._shot: np.ndarray | None = None
        self._handles: list[Handle] = []
        self._drag: _Drag | None = None

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("resize_event", lambda event: self._apply_view())

    # ---- view ------------------------------------------------------------

    def reset_view(self, width: float | None = None, height: float | None = None) -> None:
        if width is not None:
            self._field_size = (width, height)
        w, h = self._field_size
        mx, my = FIELD_MARGIN * w, FIELD_MARGIN * h
        self._view = ((-mx, w + mx), (-my, h + my))
        self._apply_view()
        self.draw_idle()

    def _apply_view(self) -> None:
        """Show the requested view, widened along one axis so 1 m is the same
        number of pixels horizontally and vertically (circles stay circles)."""
        if self._view is None:
            self.reset_view()
            return
        (x0, x1), (y0, y1) = self._view
        box = self.ax.get_window_extent()
        if box.width > 0 and box.height > 0:
            per_px_x, per_px_y = (x1 - x0) / box.width, (y1 - y0) / box.height
            if per_px_x > per_px_y:
                cy, half = (y0 + y1) / 2, per_px_x * box.height / 2
                y0, y1 = cy - half, cy + half
            else:
                cx, half = (x0 + x1) / 2, per_px_y * box.width / 2
                x0, x1 = cx - half, cx + half
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)

    # ---- drawing ---------------------------------------------------------

    def render(self, snap: Snapshot, mode: Mode) -> None:
        ax = self.ax
        ax.clear()
        self._style_axes(snap)
        mode.draw(ax, snap)
        self._draw_selected_geometry(snap)
        self._draw_sensors(snap)
        self._draw_shot_and_estimate(snap, mode)
        self._handles = mode.handles(snap)
        for h in self._handles:
            ax.scatter(*h.position, marker=h.marker, s=90, color=h.color, edgecolors=INK, linewidths=1.2, zorder=14)

        self._sensors = snap.positions
        self._shot = snap.shot
        self._apply_view()
        self.draw_idle()

    def _style_axes(self, snap: Snapshot) -> None:
        ax = self.ax
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("x (m)", color=MUTED, fontsize=9)
        ax.set_ylabel("y (m)", color=MUTED, fontsize=9)
        w, h = snap.scenario.width, snap.scenario.height
        ax.add_patch(Rectangle((0, 0), w, h, fill=False, edgecolor=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1))

    def _draw_sensors(self, snap: Snapshot) -> None:
        if snap.n == 0:
            return
        ax = self.ax
        pos = snap.positions
        ax.scatter(pos[:, 0], pos[:, 1], marker="^", s=120, c=snap.colors, edgecolors=SURFACE, linewidths=1.5, zorder=10)
        for i, (label, (x, y)) in enumerate(zip(snap.labels, pos)):
            text = f"{label} (ref)" if i == snap.reference else label
            ax.annotate(
                text, (x, y), xytext=(8, 7), textcoords="offset points", fontsize=10, color=INK,
                fontweight="bold" if i == snap.selected else "normal", zorder=11,
            )
        if snap.selected is not None:
            x, y = pos[snap.selected]
            ax.scatter([x], [y], s=480, facecolors="none", edgecolors=snap.colors[snap.selected], linewidths=2.0, zorder=9)

    def _draw_selected_geometry(self, snap: Snapshot) -> None:
        """The selected sensor's distance to the shot, the quantity its equation is about."""
        if snap.selected is None or snap.shot is None:
            return
        i = snap.selected
        s, p = snap.positions[i], snap.shot
        self.ax.plot([s[0], p[0]], [s[1], p[1]], color=snap.colors[i], linewidth=1.6, linestyle="--", zorder=5)
        mid = (s + p) / 2
        self.ax.annotate(
            f"|p − {usub('s', snap.labels[i])}| = {snap.distances[i]:.2f} m", mid, xytext=(6, -14),
            textcoords="offset points", fontsize=9, color=INK_SECONDARY, zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85),
        )

    def _draw_shot_and_estimate(self, snap: Snapshot, mode: Mode) -> None:
        ax = self.ax
        legend = []
        if snap.shot is not None:
            ax.scatter(*snap.shot, marker="*", s=300, color=INK, edgecolors=SURFACE, linewidths=1.0, zorder=12)
            legend.append(Line2D([], [], marker="*", color=INK, linestyle="none", markersize=13, label="shot (true)"))
        fix = mode.fix
        if fix is not None:
            ex, ey = fix.position
            if snap.shot is not None:
                ax.plot([ex, snap.shot[0]], [ey, snap.shot[1]], color=INK_SECONDARY, linewidth=1.0, linestyle=":", zorder=11)
            ax.scatter([ex], [ey], marker="o", s=170, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=13)
            ax.scatter([ex], [ey], marker="+", s=170, color=INK, linewidths=1.3, zorder=13)
            legend.append(
                Line2D([], [], marker="o", markerfacecolor="none", color=INK, linestyle="none", markersize=10,
                       label=f"estimate ({mode.estimate_source})")
            )
        if legend:
            ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(legend),
                      frameon=False, fontsize=9, borderaxespad=0.2, handletextpad=0.3)

    # ---- mouse -----------------------------------------------------------

    def _hit(self, event) -> tuple[str, int, str] | None:
        """The nearest draggable thing within PICK_RADIUS_PX of the cursor."""
        candidates = []  # (distance_px, kind, index, name)
        to_px = self.ax.transData.transform
        click = np.array([event.x, event.y])
        if len(self._sensors):
            d = np.linalg.norm(to_px(self._sensors) - click, axis=1)
            i = int(np.argmin(d))
            candidates.append((d[i], "sensor", i, ""))
        if self._shot is not None:
            candidates.append((np.linalg.norm(to_px(self._shot) - click), "shot", -1, ""))
        for h in self._handles:
            # Handles win ties: they sit on top and are the reason the mode exists.
            candidates.append((np.linalg.norm(to_px(h.position) - click) - 0.5, "handle", -1, h.name))
        if not candidates:
            return None
        dist, kind, index, name = min(candidates, key=lambda cand: cand[0])
        return (kind, index, name) if dist <= PICK_RADIUS_PX else None

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        press = (event.x, event.y)
        if event.button == 2:  # middle: pan
            self._drag = _Drag("pan", press_px=press, start_view=(self.ax.get_xlim(), self.ax.get_ylim()))
            return
        hit = self._hit(event)
        if event.button == 3:
            if hit is not None and hit[0] == "sensor":
                self.sensorRemoved.emit(hit[1])
            return
        if event.button != 1:
            return
        if hit is not None:
            kind, index, name = hit
            self._drag = _Drag(kind, index=index, name=name, press_px=press)
        elif self.fire_mode or "shift" in getattr(event, "modifiers", ()):
            self._drag = _Drag("shot", press_px=press, moved=True)
            self.shotMoved.emit(event.xdata, event.ydata)
        else:
            # The window appends the new sensor, so its index will be the current count.
            self._drag = _Drag("sensor", index=len(self._sensors), press_px=press, new=True)
            self.sensorAdded.emit(event.xdata, event.ydata)

    def _on_motion(self, event) -> None:
        if event.inaxes is self.ax and event.xdata is not None:
            self.cursorMoved.emit(event.xdata, event.ydata)
        drag = self._drag
        if drag is None:
            self._update_cursor(event)
            return
        if drag.kind == "pan":
            self._pan(drag, event)
            return
        if not drag.moved:
            if np.hypot(event.x - drag.press_px[0], event.y - drag.press_px[1]) < DRAG_THRESHOLD_PX:
                return
            drag.moved = True
        if event.inaxes is not self.ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        if drag.kind == "sensor":
            self.sensorMoved.emit(drag.index, x, y)
        elif drag.kind == "shot":
            self.shotMoved.emit(x, y)
        elif drag.kind == "handle":
            self.handleMoved.emit(drag.name, x, y)

    def _on_release(self, event) -> None:
        drag, self._drag = self._drag, None
        if drag is not None and drag.kind == "sensor" and not drag.moved and not drag.new:
            self.sensorClicked.emit(drag.index)

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 1 / ZOOM_STEP if event.step > 0 else ZOOM_STEP
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        self._view = (
            (cx + (x0 - cx) * factor, cx + (x1 - cx) * factor),
            (cy + (y0 - cy) * factor, cy + (y1 - cy) * factor),
        )
        self._apply_view()
        self.draw_idle()

    def _pan(self, drag: _Drag, event) -> None:
        (x0, x1), (y0, y1) = drag.start_view
        bbox = self.ax.get_window_extent()
        dx = (event.x - drag.press_px[0]) * (x1 - x0) / bbox.width
        dy = (event.y - drag.press_px[1]) * (y1 - y0) / bbox.height
        self._view = ((x0 - dx, x1 - dx), (y0 - dy, y1 - dy))
        self._apply_view()
        self.draw_idle()

    def _update_cursor(self, event) -> None:
        if event.inaxes is self.ax and self._hit(event) is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.fire_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
