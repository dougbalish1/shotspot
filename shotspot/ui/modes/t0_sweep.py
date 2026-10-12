"""Mode 2, "t₀ sweep": guess t₀ and watch how badly the circles fail to meet.

The guess is set with the slider (or by clicking the plot) under the field.
For each guess the circles have radius c·(tᵢ − t₀_guess); the mode shows the
best compromise point for that guess (it is this mode's estimate) and, in the
plot, the RMS mismatch for every guess in the slider's range. That curve
drops to zero only where the circles meet: at the true t₀, or at two guesses
in an ambiguous layout. The math is in core/sweep.py.

Steps and their geometry on the field:

    1 Guess t₀           the circles for the current guess
    2 Best point         + each circle's miss, a short segment from the best point to the circle
    3 Sweep              + the track the best point follows as the guess changes
    4 What it tells us   (as step 3)
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from ...core.accuracy import Fix
from ...core.sweep import SweepCurve, SweepPoint, best_fit_for_t0, default_t0_range, sweep
from ..fmt import big, length, meters, millis, num, point, seconds, sub, swatch, wash
from ..snapshot import Snapshot
from .base import Mode, Throttle, roadmap

STEPS = ["Guess t₀", "Best point", "Sweep", "What it tells us"]
GUESS, BEST, SWEEP, MEANING = range(len(STEPS))
SAMPLES = 400  # guesses per sweep
RANGE_FACTOR = 1.0  # the slider reaches back far enough for a shot one field diagonal from the nearest sensor
SLIDER_TICK = 1e-4  # s per slider tick (0.1 ms)
START = 0.05  # a new scenario's first guess: this fraction of the range before the first arrival
FLANK = 0.010  # s: how far either side of the best dip the panel reports the mismatch
SHARP = 0.2  # a dip is "sharp" if moving FLANK away raises the miss by this fraction of c·FLANK
RMS_FLOOR = 1e-3  # m: the plot's log scale bottoms out at 1 mm
MEETS = 1e-3  # m: an RMS mismatch below this counts as "the circles meet"
SWEEP_INTERVAL = 0.2  # s: while a sensor is being dragged, redo the sweep at most this often


class SweepMode(Mode):
    title = "t₀ sweep"
    estimate_source = "best point for the guessed t₀"
    summary = (
        "We don't know t₀, so guess it: each guess gives circles of radius c·(tᵢ − t₀_guess). "
        "A wrong guess makes every radius wrong by the same amount and the circles miss each other; "
        "the right guess makes them meet. Sweep the guess with the slider under the field and watch "
        "the plot: the mismatch drops to zero only at the true t₀ (or at two guesses, in an ambiguous layout)."
    )

    def __init__(self) -> None:
        super().__init__()
        self.t0_guess: float | None = None
        self.t0_range = (0.0, 1.0)
        self.curve: SweepCurve | None = None
        self.point: SweepPoint | None = None  # the best point for the current guess
        self.flanks: tuple[SweepPoint, SweepPoint] | None = None  # FLANK either side of the best dip
        self._curve_key = None
        self._flanks_cache: tuple[SweepPoint, SweepPoint] | None = None
        self._scenario = None
        self._view: SweepView | None = None
        # A sweep takes tens of milliseconds, too slow to redo on every mouse move. While the scenario is
        # being edited, the plot keeps the last curve and is brought up to date when the edits pause.
        self._throttle = Throttle(self, SWEEP_INTERVAL, self.changed.emit)

    # ---- state -------------------------------------------------------------

    def view(self) -> QWidget:
        if self._view is None:
            self._view = SweepView(self)
        return self._view

    def set_guess(self, t0: float) -> None:
        self.t0_guess = float(t0)
        self.changed.emit()

    def jump_to_best(self) -> None:
        if self.curve is not None and self.curve.dips:
            self.set_guess(self.curve.dips[0].t0_guess)

    def update(self, snap: Snapshot) -> None:
        self.fix = self.point = self.flanks = None
        if not snap.ready:
            self.curve = None
            self._refresh_view(snap)
            return
        sc = snap.scenario
        lo, hi = default_t0_range(snap.times, snap.c, RANGE_FACTOR * np.hypot(sc.width, sc.height))
        self.t0_range = (lo, hi)

        # The curve depends only on the sensors and their times, so moving the guess doesn't redo it.
        key = (snap.positions.tobytes(), snap.times.tobytes(), snap.c, snap.reference, lo, hi)
        if key != self._curve_key:
            if sc is not self._scenario or self.curve is None or self._throttle.allowed():
                self._redo_sweep(snap, lo, hi)
                self._curve_key = key
        self.flanks = self._flanks_cache

        if sc is not self._scenario or self.t0_guess is None:
            # A new scenario: start just before the first arrival, the latest t₀ could possibly be.
            # That is only right if the shot is practically on top of a sensor.
            self._scenario = sc
            self.t0_guess = hi - START * (hi - lo)
        self.t0_guess = float(np.clip(self.t0_guess, lo, hi))

        nearest = self.curve.positions[np.argmin(np.abs(self.curve.t0_values - self.t0_guess))]
        self.point = best_fit_for_t0(snap.positions, snap.times, snap.c, self.t0_guess, snap.reference, nearest)
        self.fix = Fix(self.point.position, self.t0_guess, "sweep")
        self._refresh_view(snap)

    def _redo_sweep(self, snap: Snapshot, lo: float, hi: float) -> None:
        self.curve = sweep(snap.positions, snap.times, snap.c, np.linspace(lo, hi, SAMPLES), snap.reference)
        best = self.curve.dips[0] if self.curve.dips else None
        self._flanks_cache = None if best is None else tuple(
            best_fit_for_t0(snap.positions, snap.times, snap.c, best.t0_guess + dt, snap.reference, best.position)
            for dt in (-FLANK, FLANK)
        )
        self._throttle.done()

    def _refresh_view(self, snap: Snapshot) -> None:
        if self._view is not None:
            self._view.show_state(self, snap)

    def step_count(self, snap: Snapshot) -> int:
        return len(STEPS) if snap.ready else 0

    def legend_handles(self, snap: Snapshot) -> list:
        if self.curve is None or self.step < SWEEP:
            return []
        t = snap.theme
        return [Line2D([], [], color=t.ink_secondary, linestyle=":", linewidth=1.4, label="best point as the guess changes")]

    # ---- field -----------------------------------------------------------

    def draw(self, ax, snap: Snapshot) -> None:
        if self.point is None:
            return
        t, pt = snap.theme, self.point
        for i, (sensor, r) in enumerate(zip(snap.positions, pt.radii)):
            if r > 0:
                width = 2.6 if i == snap.selected else 1.5
                ax.add_patch(Circle(sensor, r, fill=False, edgecolor=snap.colors[i], linewidth=width, zorder=3))

        if self.step >= BEST:  # each circle's miss: from the best point straight to the circle
            for i, (sensor, r) in enumerate(zip(snap.positions, pt.radii)):
                away = pt.position - sensor
                d = np.linalg.norm(away)
                if r > 0 and d > 0:
                    on_circle = sensor + r * away / d
                    ax.plot([pt.position[0], on_circle[0]], [pt.position[1], on_circle[1]],
                            color=snap.colors[i], linewidth=3.0, solid_capstyle="butt", zorder=8)

        if self.step >= SWEEP:
            track = self.curve.positions
            ax.plot(track[:, 0], track[:, 1], color=t.ink_secondary, linestyle=":", linewidth=1.4, zorder=6)

    # ---- Method tab --------------------------------------------------------

    def method_html(self, snap: Snapshot) -> str:
        out = [f"<h2>{self.title}</h2>"]
        if self.point is None:
            out.append('<p class="note">Place at least 3 sensors and fire a shot (shift-click) to sweep t<sub>0</sub>.</p>')
            return "\n".join(out)
        out.append(roadmap(STEPS, self.step))
        builders = {GUESS: self._guess, BEST: self._best, SWEEP: self._sweep, MEANING: self._meaning}
        out.append(builders[self.step](snap))
        return "\n".join(out)

    def _sensor_block(self, snap: Snapshot, i: int, line: str) -> str:
        label = snap.labels[i]
        html = f'<p class="eq"><a name="sensor-{label}"></a>{swatch(snap.colors[i])}&nbsp;{line}</p>'
        return wash(html, snap.theme.selected_wash) if i == snap.selected else html

    def _guess(self, snap: Snapshot) -> str:
        pt, sc, g = self.point, snap.scenario, self.t0_guess
        out = [
            "<h3>Step 1. Guess t<sub>0</sub></h3>",
            f"<p>We don't know when the shot was fired, so guess: t<sub>0,guess</sub> = {seconds(g)} (the slider "
            "under the field). Each sensor's circle then has radius</p>",
            '<p class="eq">r<sub>i</sub> = c·(t<sub>i</sub> − t<sub>0,guess</sub>)</p>',
        ]
        for i, label in enumerate(snap.labels):
            out.append(self._sensor_block(
                snap, i, f"{sub('r', label)} = {num(snap.c, 1)} × ({num(snap.times[i], 6)} − {num(g, 6)}) = {meters(pt.radii[i])}"
            ))
        off = snap.c * (sc.t0 - g)
        out.append(
            "<p>If the guess is right, these are the true ranges and the circles meet at the shot, as in Known "
            "t<sub>0</sub>. If it's wrong, <i>every</i> radius is wrong by the same amount, "
            f"c·(t<sub>0</sub> − t<sub>0,guess</sub>) = {num(snap.c, 1)} × {millis(sc.t0 - g)} = {meters(off)} here "
            "(only the simulator knows that number), and the circles miss each other.</p>"
        )
        first = int(np.argmin(snap.times))
        out.append(
            f'<p class="note">The guess can\'t be later than the first arrival ({seconds(snap.times[first])}, at '
            f"{snap.labels[first]}), or that sensor would have heard the shot before it was fired; the slider ends there. "
            "Drag it, or click the plot under the field: the circles grow and shrink together.</p>"
        )
        return "\n".join(out)

    def _best(self, snap: Snapshot) -> str:
        pt = self.point
        out = [
            "<h3>Step 2. The best point for this guess</h3>",
            "<p>For a wrong guess no point is on every circle, so take the best compromise: the point p* that "
            "makes the sum of squared misses as small as possible,</p>",
            '<p class="eq">p* = the p minimizing Σ (|p − s<sub>i</sub>| − r<sub>i</sub>)²</p>',
        ]
        start = "" if pt.linear_position is None else (
            f"from the point the subtraction trick of Known t<sub>0</sub> gives for these radii, {point(pt.linear_position)}, "
            "and from the middle of the array, "
        )
        out.append(
            f"<p>It is found numerically: start {start}take small downhill steps (Gauss-Newton, which has a mode of "
            "its own) until the sum stops shrinking, and keep the best result:</p>"
        )
        out.append(f'<p class="eq"><b>p* = {point(pt.position)}</b></p>')
        out.append("<p>How far each circle misses it:</p>")
        for i, label in enumerate(snap.labels):
            d = pt.radii[i] + pt.mismatch[i]
            out.append(self._sensor_block(
                snap, i, f"|p* − {sub('s', label)}| − {sub('r', label)} = {num(d)} − {num(pt.radii[i])} = {meters(pt.mismatch[i])}"
            ))
        out.append(
            f'<p class="eq">Σ misses² = {big(pt.cost)} m²,&nbsp; RMS = √(Σ / {snap.n}) = {length(pt.rms)}</p>'
            '<p class="note">The RMS is the size of a typical miss. On the field, each short colored segment is one '
            "circle's miss, from p* to the circle; p* is drawn as the estimate (circle and plus).</p>"
        )
        return "\n".join(out)

    def _sweep(self, snap: Snapshot) -> str:
        curve, sc = self.curve, snap.scenario
        out = [
            "<h3>Step 3. Sweep the guess</h3>",
            f"<p>Do that for {SAMPLES} guesses across the slider's range and plot each one's RMS miss: that is the "
            "curve under the field (log scale, so each grid line is ten times the one below). The dot is the "
            "current guess, and the curve drops sharply wherever the circles meet.</p>",
        ]
        meeting = sorted((d for d in curve.dips if d.rms < MEETS), key=lambda d: d.t0_guess)
        if not curve.dips:
            out.append(
                "<p>The curve has no dip inside the slider's range: the miss keeps shrinking toward the edge. The shot "
                "is probably farther from the sensors than the range reaches.</p>"
            )
            return "\n".join(out)
        out.append("<p>The bottoms of the dips:</p>" if len(curve.dips) > 1 else "<p>The bottom of the dip:</p>")
        for d in sorted(curve.dips, key=lambda d: d.t0_guess):
            verdict = "the circles meet" if d.rms < MEETS else f"RMS miss {length(d.rms)}"
            out.append(f'<p class="eq">t<sub>0,guess</sub> = {seconds(d.t0_guess)}: p* = {point(d.position)}, {verdict}</p>')

        if len(meeting) == 1:
            out.append(
                "<p>Only one guess makes the circles meet, so it must be the true t<sub>0</sub>, and the point "
                "where they meet is the shot.</p>"
            )
        elif len(meeting) >= 2:
            out.append(
                "<p><b>Two guesses</b> make the circles meet exactly: two different shots, fired at two different "
                "times, would produce exactly the same arrival times. With 3 sensors this happens when the shot is "
                "in the right place outside the array; nothing in the data can tell them apart, and only a 4th "
                "sensor can.</p>"
            )
        elif not meeting:
            out.append(
                "<p>No guess makes all the circles meet exactly: with 4 or more sensors and timing noise there are "
                "more equations than unknowns, and they can't all be satisfied. The bottom of the dip is the best "
                "compromise, the least-squares answer.</p>"
            )
        if sc.sigma > 0 and snap.n == 3 and meeting:
            out.append(
                '<p class="note">With exactly 3 sensors there is almost always a guess where the circles meet, even with '
                "noise: three equations, three unknowns. The noise moves <i>where</i> they meet, not whether. (Only "
                "large noise can leave no exact meeting at all: the closed form's quadratic then has no real root.)</p>"
            )

        if self.flanks is not None:
            left, right = self.flanks
            change = snap.c * FLANK
            out.append(
                f"<p>How sharp is the dip? Move the guess {num(FLANK * 1e3, 0)} ms either way from the best one and every "
                f"radius changes by c × {num(FLANK * 1e3, 0)} ms = {meters(change)}. The circles then miss by "
                f"{length(left.rms)} (earlier guess) and {length(right.rms)} (later guess).</p>"
            )
            rise = min(left.rms, right.rms) - curve.dips[0].rms
            if rise > SHARP * change:
                out.append("<p>That is a large share of the change: a <b>sharp</b> dip, so the arrival times pin "
                           "t<sub>0</sub> down tightly.</p>")
            else:
                out.append(
                    "<p>That is far less than the change: the circles nearly meet anyway, at a point that slides along "
                    "the dotted track. A <b>shallow</b> dip like this pins t<sub>0</sub> down only loosely, and with it "
                    "the distance to the shot, so a little timing noise moves the answer a long way along that track.</p>"
                )
        out.append(
            '<p class="note">On the field, the dotted line is the track p* follows as the guess changes: it passes '
            "through the shot at the right guess. Press <i>Jump to best dip</i> to set the guess there.</p>"
        )
        return "\n".join(out)

    def _meaning(self, snap: Snapshot) -> str:
        return "\n".join([
            "<h3>Step 4. What it tells us</h3>",
            "<p>t<sub>0</sub> is just another unknown. The problem has three, x, y and t<sub>0</sub>, and the sweep "
            "split them up: for each guess of t<sub>0</sub>, least squares found the best (x, y), and scanning the "
            "guesses found the t<sub>0</sub> whose best point fits perfectly. That is why three sensors are "
            "enough: three arrival times for three unknowns.</p>",
            "<p>Scanning is slow and only as precise as its steps, but the picture is the point: a dip means the data "
            "pin down a time and a place. The remaining modes find all three unknowns at once:</p>",
            '<p class="eq"><b>TDOA hyperbolas</b> remove t<sub>0</sub> by subtracting arrival times.</p>',
            '<p class="eq"><b>Closed form</b> solves for t<sub>0</sub> with a quadratic, whose two roots are the two '
            "dips of an ambiguous layout.</p>",
            '<p class="eq"><b>Gauss-Newton</b> walks downhill in x, y and t<sub>0</sub> together.</p>',
        ])


class SweepView(QWidget):
    """The guess controls and the RMS-miss-vs-guess plot, shown under the field."""

    def __init__(self, mode: SweepMode) -> None:
        super().__init__()
        self.mode = mode
        self._lo = 0.0
        self._xlim: tuple[float, float] | None = None  # zoomed x range, or None for the full range
        self._scenario = None

        row = QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 0)
        guess_label = QLabel("Guessed t₀")
        row.addWidget(guess_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setSingleStep(1)  # 0.1 ms
        self.slider.setPageStep(100)  # 10 ms
        row.addWidget(self.slider, stretch=1)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(6)
        self.spin.setSingleStep(0.001)
        self.spin.setSuffix(" s")
        self.spin.setKeyboardTracking(False)
        row.addWidget(self.spin)
        guess_tip = (
            "The guessed emission time t₀. Every circle's radius is c·(tᵢ − t₀,guess), so moving the guess grows or\n"
            "shrinks all the circles together. The range ends at the first arrival time (a later t₀ is impossible)\n"
            "and reaches back far enough for a shot one field diagonal away. Clicking or dragging on the plot also sets it."
        )
        for w in (guess_label, self.slider, self.spin):
            w.setToolTip(guess_tip)
        self.jump = QPushButton("Jump to best dip")
        self.jump.setToolTip("Set the guess to the t₀ where the circles come closest to meeting (the lowest dip of the plot)")
        row.addWidget(self.jump)

        self.figure = Figure(figsize=(6, 1.6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedHeight(170)
        self.ax = self.figure.add_subplot()
        self.figure.subplots_adjust(left=0.06, right=0.985, bottom=0.27, top=0.95)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addLayout(row)
        column.addWidget(self.canvas)

        self.slider.valueChanged.connect(lambda v: mode.set_guess(self._lo + v * SLIDER_TICK))
        self.spin.valueChanged.connect(mode.set_guess)
        self.jump.clicked.connect(mode.jump_to_best)
        self._pressed = False
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", lambda e: setattr(self, "_pressed", False))
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    # ---- showing the mode's state ----------------------------------------

    def show_state(self, mode: SweepMode, snap: Snapshot) -> None:
        ready = mode.point is not None
        for w in (self.slider, self.spin, self.jump):
            w.setEnabled(ready)
        if snap.scenario is not self._scenario:
            self._scenario, self._xlim = snap.scenario, None
        if ready:
            lo, hi = mode.t0_range
            self._lo = lo
            for w in (self.slider, self.spin):
                w.blockSignals(True)
            self.slider.setRange(0, round((hi - lo) / SLIDER_TICK))
            self.slider.setValue(round((mode.t0_guess - lo) / SLIDER_TICK))
            self.spin.setRange(lo, hi)
            self.spin.setValue(mode.t0_guess)
            for w in (self.slider, self.spin):
                w.blockSignals(False)
            self.jump.setEnabled(bool(mode.curve.dips))
        self._plot(mode, snap)

    def _plot(self, mode: SweepMode, snap: Snapshot) -> None:
        ax, t = self.ax, snap.theme
        ax.clear()
        self.figure.set_facecolor(t.surface)
        ax.set_facecolor(t.surface)
        for spine in ax.spines.values():
            spine.set_color(t.axis)
        ax.tick_params(colors=t.muted, labelsize=8, which="both")
        ax.set_xlabel("guessed t₀ (s)", color=t.muted, fontsize=9, labelpad=2)
        ax.set_ylabel("RMS miss", color=t.muted, fontsize=9)
        if mode.point is None:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.text(0.5, 0.5, "Place at least 3 sensors and a shot", transform=ax.transAxes, ha="center",
                    va="center", color=t.muted, fontsize=9)
            self.canvas.draw_idle()
            return

        curve, lo, hi = mode.curve, *mode.t0_range
        rms = np.maximum(curve.rms, RMS_FLOOR)
        ax.set_yscale("log")
        ax.grid(True, which="major", axis="y", color=t.grid, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.plot(curve.t0_values, rms, color=t.ink, linewidth=1.5, zorder=3)
        for d in curve.dips:
            ax.scatter([d.t0_guess], [max(d.rms, RMS_FLOOR)], s=36, facecolors="none", edgecolors=t.ink, linewidths=1.2, zorder=4)
        ax.scatter([snap.scenario.t0], [RMS_FLOOR], marker="*", s=110, color=t.ink, edgecolors=t.surface,
                   linewidths=0.8, zorder=5, clip_on=False)  # the true t₀, like the star on the field
        guess_rms = max(mode.point.rms, RMS_FLOOR)
        ax.axvline(mode.t0_guess, color=t.muted, linestyle="--", linewidth=1.0, zorder=2)
        ax.scatter([mode.t0_guess], [guess_rms], s=60, color=t.ink, edgecolors=t.surface, linewidths=1.5, zorder=6)

        top = max(float(rms.max()), RMS_FLOOR * 10)
        ax.set_ylim(RMS_FLOOR / 1.5, top * 2.5)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(_length_label))
        x0, x1 = self._xlim if self._xlim is not None else (lo, hi)
        ax.set_xlim(max(x0, lo), min(x1, hi))
        self.canvas.draw_idle()

    # ---- mouse: click or drag to set the guess, scroll to zoom -----------

    def _guess_from(self, event) -> None:
        if event.inaxes is self.ax and event.xdata is not None and self.mode.point is not None:
            lo, hi = self.mode.t0_range
            self.mode.set_guess(float(np.clip(event.xdata, lo, hi)))

    def _on_press(self, event) -> None:
        if event.button == 1:
            if event.dblclick:  # double-click: back to the full range
                self._xlim = None
                self.mode.changed.emit()
                return
            self._pressed = True
            self._guess_from(event)

    def _on_motion(self, event) -> None:
        if self._pressed:
            self._guess_from(event)

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or self.mode.point is None:
            return
        lo, hi = self.mode.t0_range
        x0, x1 = self.ax.get_xlim()
        factor = 1 / 1.25 if event.step > 0 else 1.25
        x0, x1 = event.xdata + (x0 - event.xdata) * factor, event.xdata + (x1 - event.xdata) * factor
        self._xlim = None if (x0 <= lo and x1 >= hi) else (max(x0, lo), min(x1, hi))
        self.mode.changed.emit()


def _length_label(value: float, _pos) -> str:
    """Tick labels for the log axis: 1 mm, 1 cm, 10 cm, 1 m, 10 m, 100 m, 1 km."""
    if value < 0.01:
        return f"{value * 1000:g} mm"
    if value < 1:
        return f"{value * 100:g} cm"
    if value < 1000:
        return f"{value:g} m"
    return f"{value / 1000:g} km"
