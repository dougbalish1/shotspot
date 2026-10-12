"""Mode 6, "Accuracy / GDOP": how big an error should we expect, and where?

The field is colored by the 1σ position error (DRMS) the current layout would
give a shot at each point, for the current timing noise σ. That error is a
geometry factor (core/accuracy.py: `dilution`) times c·σ. The error ellipse at
the shot comes from the covariance (JᵀJ)⁻¹·(cσ)², and the Monte Carlo button
fires many noisy shots so the prediction can be compared with real scatter.

With the noise slider at 0 every predicted error is 0, so the mode then uses
FALLBACK_SIGMA instead, and says so.

The panel under the field holds the Monte Carlo controls, the heatmap's
colorbar, and a magnified view around the shot (inside the array the
ellipses are far smaller than a pixel of the field).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from ...core.accuracy import (
    SIGMA_95, MonteCarloResult, covariance, dilution, drms, ellipse_probability, fraction_inside, jacobian_at,
    monte_carlo,
)
from ...core.geometry import convex_hull, ellipse_points, error_ellipse
from ..fmt import length, matrix, millis, num, point, sub, swatch, wash
from ..snapshot import Snapshot
from ..style import Theme
from .base import Mode, Throttle, draw_solution_square, roadmap, solution_square_handle

STEPS = ["Noise to error", "The map", "Error ellipse", "Monte Carlo"]
NOISE, MAP, ELLIPSE, TRIALS = range(len(STEPS))
FALLBACK_SIGMA = 1e-3  # s: used when the noise slider is at 0
GRID = 220  # heatmap points along the longer side of the view
COLOR_RANGE = (0.5, 100.0)  # geometry factors at the two ends of the color scale
MAP_INTERVAL = 0.2  # s: while a sensor is being dragged, recompute the map at most this often


def _error_scale(c_sigma: float) -> LogNorm:
    return LogNorm(vmin=COLOR_RANGE[0] * c_sigma, vmax=COLOR_RANGE[1] * c_sigma)


def _colormap(theme: Theme) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(f"error-{theme.name}", list(theme.sequential))


def _contour_levels(norm: LogNorm) -> list[float]:
    """1-2-5 steps (…, 0.5, 1, 2, 5, 10, … meters) inside the color range."""
    levels = []
    for decade in range(int(np.floor(np.log10(norm.vmin))), int(np.ceil(np.log10(norm.vmax))) + 1):
        levels += [m * 10.0**decade for m in (1, 2, 5)]
    return [v for v in levels if norm.vmin * 1.5 < v < norm.vmax]


@dataclass
class Cluster:
    """Where the Monte Carlo fixes gather, and the predicted covariance there. Normally the shot; in an ambiguous
    layout the solvers keep landing on the other solution, and the prediction has to be made there to be comparable."""

    center: np.ndarray
    cov: np.ndarray  # 2×2 position covariance predicted at the center
    name: str  # "the shot" or "the other solution"
    share_other: float  # fraction of the fixes nearer the other solution than the shot (0 when there is none)


class AccuracyMode(Mode):
    title = "Accuracy / GDOP"
    summary = (
        "How far off should we expect a fix to be? Timing noise σ becomes position error through the geometry: "
        "error = (geometry factor) × c·σ. The colors show that error for a shot anywhere on the field: small "
        "inside the sensor array, growing outside it. The ellipse at the shot shows its shape, stretched along "
        "the line of sight for distant shots, and Monte Carlo fires hundreds of noisy shots to check the prediction."
    )

    def __init__(self) -> None:
        super().__init__()
        self.sigma = FALLBACK_SIGMA
        self.cov: np.ndarray | None = None  # 3×3 covariance of (x, y, t₀) at the shot
        self.other: np.ndarray | None = None  # the second solution of an ambiguous layout, where the solvers may land
        self.other_cov: np.ndarray | None = None  # the covariance there
        self.cluster: Cluster | None = None
        self.mc: MonteCarloResult | None = None
        self._mc_key = None
        self._key = None
        self._snap: Snapshot | None = None
        self._map_cache: tuple | None = None  # (key, X, Y, error)
        self._map_throttle = Throttle(self, MAP_INTERVAL, self.changed.emit)
        self._view: AccuracyView | None = None

    @property
    def c_sigma(self) -> float:
        return self._snap.c * self.sigma if self._snap is not None else 0.0

    @property
    def position_cov(self) -> np.ndarray | None:
        return None if self.cov is None else self.cov[:2, :2]

    def view(self) -> QWidget:
        if self._view is None:
            self._view = AccuracyView(self)
        return self._view

    def update(self, snap: Snapshot) -> None:
        super().update(snap)  # the estimate: closed form polished by Gauss-Newton
        self._snap = snap
        self.sigma = snap.scenario.sigma if snap.scenario.sigma > 0 else FALLBACK_SIGMA
        self.cov = self.other = self.other_cov = self.cluster = None
        if snap.ready:
            self.cov = covariance(snap.shot, snap.positions, snap.c, self.sigma)
            self._key = (snap.positions.tobytes(), snap.shot.tobytes(), snap.c, self.sigma, snap.reference)
            if self._key != self._mc_key:
                self.mc = None  # the trials belonged to a different scenario
            if snap.ambiguous:
                self.other = snap.other_solution(snap.shot)
                self.other_cov = covariance(self.other, snap.positions, snap.c, self.sigma)
            self.cluster = self._cluster(snap)
        if self._view is not None:
            self._view.show_state(self, snap)

    def _cluster(self, snap: Snapshot) -> Cluster | None:
        if self.cov is None:
            return None
        share = 0.0
        if self.mc is not None and self.other_cov is not None and len(self.mc.estimates):
            d_other = np.linalg.norm(self.mc.estimates - self.other, axis=1)
            d_shot = np.linalg.norm(self.mc.estimates - snap.shot, axis=1)
            share = float(np.mean(d_other < d_shot))
        if share > 0.5:
            return Cluster(self.other, self.other_cov[:2, :2], "the other solution", share)
        return Cluster(snap.shot, self.position_cov, "the shot", share)

    def run_monte_carlo(self, trials: int) -> None:
        snap = self._snap
        if snap is None or not snap.ready:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.mc = monte_carlo(snap.positions, snap.shot, snap.scenario.t0, snap.c, self.sigma, trials,
                                  np.random.default_rng(), snap.reference)
        finally:
            QApplication.restoreOverrideCursor()
        self._mc_key = self._key
        self.changed.emit()

    def step_count(self, snap: Snapshot) -> int:
        return len(STEPS) if snap.ready else 0

    def legend_handles(self, snap: Snapshot) -> list:
        if self.cov is None:
            return []
        t = snap.theme
        handles = [Line2D([], [], color=t.ink, linewidth=1.6, label="1σ ellipse"),
                   Line2D([], [], color=t.ink, linewidth=1.2, linestyle=(0, (4, 2)), label="95% ellipse")]
        if self.mc is not None:
            handles.append(Line2D([], [], marker="o", color=t.ink_secondary, linestyle="none", markersize=3,
                                  alpha=0.6, label="Monte Carlo fixes"))
        if self.other is not None:
            handles.append(solution_square_handle(snap))
        return handles

    # ---- field -----------------------------------------------------------

    def error_map(self, snap: Snapshot, view: tuple[float, float, float, float]):
        """(X, Y, error) over the view, cached until the view or the layout changes. While a sensor is being
        dragged the old map is reused between recomputations (it is drawn where it was computed)."""
        key = (view, snap.positions.tobytes(), self.c_sigma)
        same_scale = self._map_cache is not None and self._map_cache[0][2] == key[2] and len(self._map_cache[0][1]) == len(key[1])
        if self._map_cache is None or (self._map_cache[0] != key and (not same_scale or self._map_throttle.allowed())):
            self._map_throttle.done()
            x0, x1, y0, y1 = view
            scale = GRID / max(x1 - x0, y1 - y0)
            X, Y = np.meshgrid(np.linspace(x0, x1, max(round((x1 - x0) * scale), 2)),
                               np.linspace(y0, y1, max(round((y1 - y0) * scale), 2)))
            error = dilution(np.stack([X, Y], axis=-1), snap.positions) * self.c_sigma
            self._map_cache = (key, X, Y, error)
        return self._map_cache[1:]

    def draw(self, ax, snap: Snapshot) -> None:
        if snap.n < 3:
            return
        t = snap.theme
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        X, Y, error = self.error_map(snap, (x0, x1, y0, y1))
        norm = _error_scale(self.c_sigma)
        shown = np.where(np.isfinite(error), error, norm.vmax)
        extent = (X[0, 0], X[0, -1], Y[0, 0], Y[-1, 0])  # where the map was computed (it may lag a drag)
        ax.imshow(np.clip(shown, norm.vmin, norm.vmax), extent=extent, origin="lower", cmap=_colormap(t),
                  norm=norm, interpolation="bilinear", aspect="auto", zorder=0.6)  # over the grid (0.5)
        contours = ax.contour(X, Y, shown, levels=_contour_levels(norm), colors=[t.ink_secondary], linewidths=0.7,
                              alpha=0.8, zorder=2)
        ax.clabel(contours, fmt=lambda v: _plain_length(v), inline=True, fontsize=7, colors=[t.ink_secondary])

        hull = convex_hull(snap.positions)
        if len(hull) >= 3:
            loop = np.vstack([hull, hull[:1]])
            ax.plot(loop[:, 0], loop[:, 1], color=t.ink_secondary, linewidth=1.0, linestyle=(0, (5, 3)), zorder=2.5)

        halo = [patheffects.withStroke(linewidth=3.5, foreground=t.surface)]  # readable over any map color
        if self.cov is not None:
            self._draw_ellipses(ax, snap, snap.shot, self.position_cov, halo)
        if self.other is not None:  # the solvers may land here instead, so predict the error here too
            draw_solution_square(ax, snap, self.other)
            if self.other_cov is not None:
                self._draw_ellipses(ax, snap, self.other, self.other_cov[:2, :2], halo)
        if self.mc is not None and len(self.mc.estimates):
            ax.scatter(self.mc.estimates[:, 0], self.mc.estimates[:, 1], s=5, color=t.ink_secondary, alpha=0.5,
                       linewidths=0, zorder=9)

    @staticmethod
    def _draw_ellipses(ax, snap: Snapshot, center: np.ndarray, cov: np.ndarray, halo) -> None:
        for n_sigma, style, width in ((1.0, "solid", 1.6), (SIGMA_95, (0, (4, 2)), 1.2)):
            pts = ellipse_points(center, cov, n_sigma)
            ax.plot(pts[:, 0], pts[:, 1], color=snap.theme.ink, linewidth=width, linestyle=style, zorder=12, path_effects=halo)

    # ---- Method tab --------------------------------------------------------

    def method_html(self, snap: Snapshot) -> str:
        out = [f"<h2>{self.title}</h2>"]
        if not snap.ready:
            out.append('<p class="note">Place at least 3 sensors and fire a shot (shift-click) to see its expected error. '
                       "The map already shows the error for a shot anywhere.</p>" if snap.n >= 3 else
                       '<p class="note">Place at least 3 sensors to see the error map.</p>')
            return "\n".join(out)
        out.append(roadmap(STEPS, self.step))
        out.append({NOISE: self._noise, MAP: self._map, ELLIPSE: self._ellipse, TRIALS: self._trials}[self.step](snap))
        return "\n".join(out)

    def _sigma_note(self, snap: Snapshot) -> str:
        if snap.scenario.sigma > 0:
            return f"<p>Timing noise σ = {millis(self.sigma, 2)} (the slider), so c·σ = {length(self.c_sigma)}.</p>"
        return (f'<p class="note">The noise slider is at 0, which would make every error 0, so this mode uses '
                f"σ = {millis(self.sigma, 2)} (c·σ = {length(self.c_sigma)}). Move the slider to use your own.</p>")

    def _noise(self, snap: Snapshot) -> str:
        out = [
            "<h3>Step 1. From timing noise to position error</h3>",
            self._sigma_note(snap),
            "<p>Each arrival time is off by a small random δt<sub>i</sub> with standard deviation σ. That shifts each "
            "residual by c·δt<sub>i</sub>, and one Gauss-Newton step at the true answer turns that into an error in the "
            "unknowns:</p>",
            '<p class="eq">δx = −(JᵀJ)⁻¹·Jᵀ·(c·δt)&nbsp;&nbsp; ⇒ &nbsp;&nbsp;Cov[x, y, t<sub>0</sub>] = (JᵀJ)⁻¹·(cσ)²</p>',
            "<p>J is Gauss-Newton's Jacobian, evaluated at the shot. Its rows [u<sub>ix</sub>, u<sub>iy</sub>, c]:</p>",
        ]
        J = jacobian_at(snap.shot, snap.positions, snap.c)
        for i, label in enumerate(snap.labels):
            html = (f'<p class="eq"><a name="sensor-{label}"></a>{swatch(snap.colors[i])}&nbsp;{sub("J", label)} = '
                    f"[{num(J[i, 0], 4)}, {num(J[i, 1], 4)}, {num(J[i, 2], 1)}]</p>")
            out.append(wash(html, snap.theme.selected_wash) if i == snap.selected else html)
        if self.cov is None:
            out.append("<p><b>JᵀJ is singular here</b>: the geometry can't determine a position at this shot (for "
                       "example, collinear sensors with the shot on their line).</p>")
            return "\n".join(out)
        P = self.position_cov
        sx, sy = np.sqrt(P[0, 0]), np.sqrt(P[1, 1])
        out.append(f"<p>The position part of the covariance (m²):</p>{matrix(P)}")
        out.append(
            f'<p class="eq">σ<sub>x</sub> = {length(sx)},&nbsp; σ<sub>y</sub> = {length(sy)},&nbsp; '
            f"<b>DRMS = √(σ<sub>x</sub>² + σ<sub>y</sub>²) = {length(drms(P))}</b></p>"
            '<p class="note">DRMS is the typical distance between a fix and the true shot.</p>'
        )
        return "\n".join(out)

    def _map(self, snap: Snapshot) -> str:
        out = ["<h3>Step 2. The map: geometric dilution of precision</h3>"]
        out.append(
            "<p>Measure t<sub>0</sub> in meters (b = c·t<sub>0</sub>) and J's rows become [u<sub>ix</sub>, u<sub>iy</sub>, 1], "
            "with no c or σ in them. So</p>"
            '<p class="eq"><b>position error = (geometry factor) × c·σ</b></p>'
            "<p>The geometry factor is a <i>dilution of precision</i> (DOP): how many meters of position error each meter "
            "of range error becomes. It depends only on where the shot is relative to the sensors. The layout decides the "
            "multiplier; the timing noise sets the scale. (This factor leaves the t<sub>0</sub> term out, so in GPS "
            "vocabulary it is the horizontal DOP, HDOP; the GDOP proper also counts the clock error.)</p>"
        )
        if self.cov is not None:
            factor = drms(self.position_cov) / self.c_sigma
            out.append(f'<p class="eq">at the shot: {num(factor, 2)} × {length(self.c_sigma)} = {length(drms(self.position_cov))}</p>')
        out.append(
            "<p>The colors show the error a shot would have at each point of the field, with labeled contour lines. "
            "Inside the sensors' outline (dashed) the unit vectors u<sub>i</sub> point in many different directions and "
            "the error stays near c·σ. Outside, they point more and more the same way and the error grows. Along the "
            "line through collinear sensors it is infinite.</p>"
            "<p>Look for the streaks fanning out beyond each sensor. On the line through two sensors, past one of them, "
            "both unit vectors point the same way, so that pair's time difference says nothing about position, and with "
            "only three sensors the error blows up there.</p>"
            '<p class="note">Drag a sensor and watch the map change; the colorbar is under the field.</p>'
        )
        return "\n".join(out)

    def _ellipse(self, snap: Snapshot) -> str:
        out = ["<h3>Step 3. The error ellipse</h3>"]
        if self.cov is None:
            out.append("<p>No ellipse: the covariance is undefined at this shot.</p>")
            return "\n".join(out)
        major, minor, angle = error_ellipse(self.position_cov)
        center = snap.positions.mean(axis=0)
        sight = snap.shot - center
        bearing = np.degrees(np.arctan2(sight[1], sight[0]))
        off = abs((np.degrees(angle) - bearing + 90) % 180 - 90)  # angle between two undirected lines, 0 to 90°
        out.append(
            "<p>The error isn't the same in every direction. The covariance's eigenvectors point along the largest and "
            "smallest error, and the square roots of its eigenvalues are their sizes:</p>"
            f'<p class="eq">semi-axes {length(major)} and {length(minor)}, the long one at {num(np.degrees(angle) % 180, 1)}° '
            "from the x axis</p>"
            f'<p class="eq">line of sight from the array\'s center to the shot: {num(bearing % 180, 1)}°, '
            f"{num(off, 1)}° from the long axis</p>"
        )
        ratio = major / minor if minor > 0 else np.inf
        if ratio > 2 and off < 25:
            out.append(
                f"<p>The ellipse is {num(ratio, 1)} times longer than it is wide, and stretched <b>along the line of "
                "sight</b>: the distance to the shot is much less certain than its direction. That is radial stretching, "
                "the far-field \"good bearing, poor range\" seen as an error shape.</p>"
            )
        elif ratio < 1.5:
            out.append("<p>Nearly round: the sensors surround the shot, so no direction is much weaker than another.</p>")
        else:
            out.append(f"<p>{num(ratio, 1)} times longer than wide: one direction is noticeably weaker.</p>")
        out.append(
            f'<p class="note">Solid: the 1σ ellipse, which holds only {num(100 * ellipse_probability(1), 1)}% of fixes in '
            f"two dimensions (not 68% as in one). Dashed: {num(SIGMA_95, 2)}σ, which holds 95%. Both are drawn at true size "
            "on the field (zoom in with the scroll wheel) and magnified under it.</p>"
        )
        if self.other_cov is not None:
            o_major, o_minor, _ = error_ellipse(self.other_cov[:2, :2])
            out.append(
                f'<p class="note">This layout has a second solution at {point(self.other)} (the square on the field), and '
                "the solvers land there rather than on the shot. Its ellipse is drawn too, from the covariance at that "
                f"point: semi-axes {length(o_major)} and {length(o_minor)}. The geometry differs, so the predicted error "
                "differs.</p>"
            )
        return "\n".join(out)

    def _trials(self, snap: Snapshot) -> str:
        out = ["<h3>Step 4. Check it: Monte Carlo</h3>"]
        mc = self.mc
        if mc is None:
            out.append("<p>Press <i>Run trials</i> under the field. It fires that many shots from the same spot, each "
                       "with fresh random timing noise, locates every one, and scatters the fixes so the predicted "
                       "ellipse can be compared with what actually happens.</p>")
            return "\n".join(out)
        n = len(mc.estimates)
        cluster = self.cluster
        out.append(f"<p>{n} fixes" + (f" ({mc.failures} trials found no fix)" if mc.failures else "") + ":</p>")
        if cluster is not None and self.other is not None and n:
            pct = num(100 * cluster.share_other, 0)
            if cluster.share_other > 0.5:
                out.append(
                    f"<p><b>{pct}% of the fixes landed at the other solution</b>, {point(self.other)}, not at the shot. "
                    "The solvers keep the root with the later t<sub>0</sub> (the nearer shot), and timing noise doesn't "
                    "change which root that is. The fixes still scatter the way the covariance predicts, but around "
                    f"{cluster.name}, so the prediction below is made there:</p>"
                )
            else:
                out.append(f"<p>{pct}% of the fixes landed at the other solution, {point(self.other)}.</p>")
        if cluster is not None and n > 2:
            P, center = cluster.cov, cluster.center
            out.append('<table class="data" cellspacing="0"><tr><th></th><th>predicted</th><th>measured</th></tr>')
            rows = [("σ<sub>x</sub>", np.sqrt(P[0, 0]), np.sqrt(mc.covariance[0, 0])),
                    ("σ<sub>y</sub>", np.sqrt(P[1, 1]), np.sqrt(mc.covariance[1, 1])),
                    ("DRMS", drms(P), drms(mc.covariance))]
            for name, predicted, measured in rows:
                out.append(f'<tr><td nowrap class="note" style="text-align:left">{name}</td><td nowrap>{length(predicted)}</td>'
                           f"<td nowrap>{length(measured)}</td></tr>")
            for n_sigma, name in ((1.0, "inside 1σ ellipse"), (SIGMA_95, "inside 95% ellipse")):
                share = fraction_inside(mc.estimates, center, P, n_sigma)
                out.append(f'<tr><td nowrap class="note" style="text-align:left">{name}</td>'
                           f"<td nowrap>{num(100 * ellipse_probability(n_sigma), 1)}%</td><td nowrap>{num(100 * share, 1)}%</td></tr>")
            out.append("</table>")
            bias = float(np.linalg.norm(mc.mean - center))
            out.append(f"<p>The average fix is {length(bias)} from {cluster.name}.</p>")
        out.append(
            '<p class="note">The prediction is a straight-line approximation around the answer. It holds while the errors '
            "are small compared with the distances involved. For distant shots or big noise the scatter bends into a "
            "banana and spreads further than the ellipse.</p>"
        )
        return "\n".join(out)


class AccuracyView(QWidget):
    """Under the field: Monte Carlo controls, the heatmap's colorbar, and a magnified view around the shot."""

    def __init__(self, mode: AccuracyMode) -> None:
        super().__init__()
        self.mode = mode
        row = QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 0)
        trials_label = QLabel("Monte Carlo trials")
        row.addWidget(trials_label)
        self.trials = QSpinBox()
        self.trials.setRange(50, 5000)
        self.trials.setSingleStep(50)
        self.trials.setValue(500)
        for w in (trials_label, self.trials):
            w.setToolTip("How many shots to fire from the same spot, each with fresh random timing noise")
        row.addWidget(self.trials)
        self.run = QPushButton("Run trials")
        self.run.setToolTip("Fire that many noisy shots, locate each one, and scatter the fixes on the field to compare "
                            "with the predicted ellipse")
        row.addWidget(self.run)
        self.status = QLabel()
        row.addWidget(self.status, stretch=1)

        self.bar_figure = Figure(figsize=(4, 0.8))
        self.bar_canvas = FigureCanvasQTAgg(self.bar_figure)
        self.bar_canvas.setFixedHeight(70)
        self.bar_ax = self.bar_figure.add_axes((0.04, 0.55, 0.92, 0.25))

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.addLayout(row)
        left.addWidget(self.bar_canvas)
        left.addStretch()

        self.zoom_figure = Figure(figsize=(3, 1.9))
        self.zoom_canvas = FigureCanvasQTAgg(self.zoom_figure)
        self.zoom_canvas.setFixedSize(320, 190)
        self.zoom_ax = self.zoom_figure.add_axes((0.2, 0.17, 0.75, 0.75))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.addLayout(left, stretch=1)
        layout.addWidget(self.zoom_canvas)

        self.run.clicked.connect(lambda: mode.run_monte_carlo(self.trials.value()))
        # Redrawing a canvas costs tens of milliseconds, so each one is redrawn only when what it shows changes.
        self._bar_key = None
        self._zoom_key = None

    def show_state(self, mode: AccuracyMode, snap: Snapshot) -> None:
        t = snap.theme
        self.run.setEnabled(snap.ready and mode.cov is not None)
        if mode.mc is not None:
            self.status.setText(f"{len(mode.mc.estimates)} fixes shown")
        elif snap.ready:
            self.status.setText("")
        else:
            self.status.setText("needs a shot")
        for fig in (self.bar_figure, self.zoom_figure):
            fig.set_facecolor(t.surface)
        bar_key = (t.name, snap.n >= 3, mode.c_sigma, snap.scenario.sigma > 0)
        if bar_key != self._bar_key:
            self._bar_key = bar_key
            self._draw_colorbar(mode, snap)
        cluster = mode.cluster
        zoom_key = (t.name, id(mode.mc)) if cluster is None else (
            t.name, id(mode.mc), cluster.cov.tobytes(), cluster.center.tobytes(), cluster.name)
        if zoom_key != self._zoom_key:
            self._zoom_key = zoom_key
            self._draw_zoom(mode, snap)

    def _draw_colorbar(self, mode: AccuracyMode, snap: Snapshot) -> None:
        t, ax = snap.theme, self.bar_ax
        ax.clear()
        if snap.n < 3:
            ax.set_axis_off()
            self.bar_canvas.draw_idle()
            return
        ax.set_axis_on()
        norm = _error_scale(mode.c_sigma)
        self.bar_figure.colorbar(ScalarMappable(norm=norm, cmap=_colormap(t)), cax=ax, orientation="horizontal")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _plain_length(v)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(colors=t.muted, labelsize=8, which="both")
        for spine in ax.spines.values():
            spine.set_color(t.axis)
        given = "" if snap.scenario.sigma > 0 else " (slider at 0: using 1 ms)"
        ax.set_title(f"expected 1σ position error for a shot at each point, σ = {millis(mode.sigma, 1).replace('&nbsp;', ' ')}{given}",
                     color=t.muted, fontsize=8, loc="left", pad=4)
        self.bar_canvas.draw_idle()

    def _draw_zoom(self, mode: AccuracyMode, snap: Snapshot) -> None:
        t, ax = snap.theme, self.zoom_ax
        ax.clear()
        ax.set_facecolor(t.surface)
        for spine in ax.spines.values():
            spine.set_color(t.axis)
        ax.tick_params(colors=t.muted, labelsize=7)
        cluster = mode.cluster
        if cluster is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "the ellipse around the shot\nappears here", transform=ax.transAxes, ha="center",
                    va="center", color=t.muted, fontsize=8)
            self.zoom_canvas.draw_idle()
            return
        P, center = cluster.cov, cluster.center
        reach = error_ellipse(P, SIGMA_95)[0]
        for n_sigma, style, width in ((1.0, "solid", 1.6), (SIGMA_95, (0, (4, 2)), 1.2)):
            pts = ellipse_points(np.zeros(2), P, n_sigma)
            ax.plot(pts[:, 0], pts[:, 1], color=t.ink, linewidth=width, linestyle=style, zorder=3)
        if mode.mc is not None and len(mode.mc.estimates):
            d = mode.mc.estimates - center
            ax.scatter(d[:, 0], d[:, 1], s=4, color=t.ink_secondary, alpha=0.5, linewidths=0, zorder=2)
            ax.scatter(*(mode.mc.mean - center), marker="+", s=60, color=t.ink, linewidths=1.2, zorder=4)
            reach = max(reach, float(np.percentile(np.abs(d), 99)))
        if cluster.name == "the shot":
            ax.scatter([0], [0], marker="*", s=90, color=t.ink, edgecolors=t.surface, linewidths=0.8, zorder=5)
        else:
            ax.scatter([0], [0], marker="s", s=60, facecolors="none", edgecolors=t.ink, linewidths=1.2, zorder=5)
        lim = 1.25 * reach
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        signed = FuncFormatter(lambda v, _pos: _plain_length(v))
        ax.xaxis.set_major_formatter(signed)
        ax.yaxis.set_major_formatter(signed)
        ax.set_title(f"around {cluster.name}", color=t.muted, fontsize=8, loc="left", pad=3)
        self.zoom_canvas.draw_idle()


def _plain_length(value: float) -> str:
    """Tick and contour labels: 50 cm, 2 m, −1.5 m (plain text for matplotlib, no HTML)."""
    a = abs(value)
    if a == 0:
        return "0"
    sign = "−" if value < 0 else ""
    if a < 1:
        return f"{sign}{a * 100:.3g} cm"
    if a < 1000:
        return f"{sign}{a:.3g} m"
    return f"{sign}{a / 1000:.3g} km"
