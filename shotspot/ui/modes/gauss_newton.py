"""Mode 5, "Gauss-Newton": walk downhill in x, y and t₀ at once.

The start is a draggable diamond on the field (the array's center to begin
with). Every iteration is recorded by core/gauss_newton.py, and the Method tab
steps through them: Start, one step per iteration, then Result.

On the field: the whole path (faint), the part walked so far (solid), and, for
the iterate the current step starts from, the circles c·(tᵢ − t₀) for its t₀
estimate with each circle's miss. A miss is that sensor's residual rᵢ, and it
lies along uᵢ, the unit vector in that sensor's Jacobian row. An iteration
step also draws its update Δ as an arrow.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from PySide6.QtWidgets import QPushButton

from ...core.accuracy import Fix
from ...core.gauss_newton import GNIteration, GNResult, gauss_newton
from ..fmt import big, matrix, meters, millis, num, point, seconds, sub, swatch, vector, wash
from ..snapshot import Snapshot
from .base import Handle, Mode, roadmap

T0 = "t<sub>0</sub>"
STATUS = {
    "converged": "converged",
    "max_iterations": "stopped after the maximum number of iterations without settling",
    "singular": "stopped: JᵀJ is singular here, so the step is undefined",
    "hit_sensor": "stopped: the estimate landed on a sensor, where the direction uᵢ is undefined",
    "diverged": "diverged: the estimate ran off toward infinity",
    "too_few_sensors": "needs at least 3 sensors",
}


class GaussNewtonMode(Mode):
    title = "Gauss-Newton"
    estimate_source = "Gauss-Newton from the diamond"
    summary = (
        "Start from a guess for all three unknowns (x, y, t₀) and improve it: approximate each sensor's "
        "residual by a straight line around the guess, solve for the step that best zeroes them all, take "
        "it, and repeat. Near the answer it converges in a few steps. From a poor start it can run away, or "
        "settle on the other solution of an ambiguous layout: drag the diamond to try."
    )

    def __init__(self) -> None:
        super().__init__()
        self.start: np.ndarray | None = None
        self.result: GNResult | None = None
        self._scenario = None
        self._button: QPushButton | None = None

    def controls(self) -> QPushButton:
        if self._button is None:
            self._button = QPushButton("Reset start")
            self._button.setToolTip("Move the starting guess back to the middle of the sensor array")
            self._button.clicked.connect(self.reset_start)
        return self._button

    def reset_start(self) -> None:
        self.start = None
        self.changed.emit()

    def handles(self, snap: Snapshot) -> list[Handle]:
        if self.start is None:
            return []
        return [Handle("start", self.start, snap.theme.surface, marker="D")]

    def move_handle(self, name: str, x: float, y: float) -> None:
        self.start = np.array([x, y], float)
        self.changed.emit()

    def update(self, snap: Snapshot) -> None:
        self.result = self.fix = None
        if snap.scenario is not self._scenario:
            self._scenario, self.start = snap.scenario, None
        if snap.n and self.start is None:
            self.start = snap.positions.mean(axis=0)
        if not snap.ready:
            return
        self.result = gauss_newton(snap.positions, snap.times, snap.c, self.start)
        if self.result.converged:
            self.fix = Fix(self.result.position, self.result.t0, "gauss_newton")

    @property
    def n_iterations(self) -> int:
        return len(self.result.iterations) - 1 if self.result is not None else 0

    def step_count(self, snap: Snapshot) -> int:
        return self.n_iterations + 2 if self.result is not None else 0  # Start, each iteration, Result

    def _current(self) -> tuple[int, bool]:
        """Which recorded iterate the current step starts from, and whether it is an iteration step."""
        k = self.n_iterations
        if self.step == 0:
            return 0, False
        if self.step <= k:
            return self.step - 1, True
        return k, False

    def legend_handles(self, snap: Snapshot) -> list:
        if self.result is None:
            return []
        t = snap.theme
        return [
            Line2D([], [], marker="D", markerfacecolor=t.surface, markeredgecolor=t.ink, color=t.ink, linestyle="none",
                   markersize=8, label="start (drag it)"),
            Line2D([], [], marker="o", color=t.ink, linewidth=1.6, markersize=4, label="Gauss-Newton path"),
        ]

    # ---- field -----------------------------------------------------------

    def draw(self, ax, snap: Snapshot) -> None:
        gn, t = self.result, snap.theme
        if gn is None:
            return
        k, iterating = self._current()
        path = gn.path
        ax.plot(path[:, 0], path[:, 1], color=t.muted, linewidth=1.0, linestyle=(0, (1, 2)), zorder=7)
        walked = path[: k + 1]
        ax.plot(walked[:, 0], walked[:, 1], color=t.ink, linewidth=1.6, marker="o", markersize=4, zorder=8)

        it = gn.iterations[k]
        p, t0 = it.estimate[:2], it.estimate[2]
        for i, (sensor, ti) in enumerate(zip(snap.positions, snap.times)):
            r = snap.c * (ti - t0)
            if r <= 0:
                continue
            ax.add_patch(Circle(sensor, r, fill=False, edgecolor=snap.colors[i], linewidth=2.4 if i == snap.selected else 1.2,
                                alpha=0.8, zorder=3))
            away = p - sensor
            d = np.linalg.norm(away)
            if d > 0:
                on_circle = sensor + r * away / d
                ax.plot([p[0], on_circle[0]], [p[1], on_circle[1]], color=snap.colors[i], linewidth=3.0,
                        solid_capstyle="butt", zorder=8)
        if iterating:
            nxt = gn.iterations[k + 1].estimate[:2]
            ax.annotate("", xy=nxt, xytext=p, zorder=12,
                        arrowprops=dict(arrowstyle="-|>", color=t.ink, linewidth=1.6, shrinkA=0, shrinkB=0))

    # ---- Method tab --------------------------------------------------------

    def method_html(self, snap: Snapshot) -> str:
        out = [f"<h2>{self.title}</h2>"]
        gn = self.result
        if gn is None:
            out.append('<p class="note">Place at least 3 sensors and fire a shot (shift-click) to run Gauss-Newton.</p>')
            return "\n".join(out)
        k = self.n_iterations
        titles = ["Start", *[str(n) for n in range(1, k + 1)], "Result"]
        shown = {0, 1, 2, 3, k - 1, k, k + 1} | set(range(self.step - 1, self.step + 2))
        out.append(roadmap(titles, self.step, numbered=False, shown=shown if len(titles) > 10 else None))
        if self.step == 0:
            out.append(self._start(snap))
        elif self.step <= k:
            out.append(self._iteration(snap, gn.iterations[self.step - 1], gn.iterations[self.step]))
        else:
            out.append(self._result(snap))
        return "\n".join(out)

    def _block(self, snap: Snapshot, i: int, text: str) -> str:
        label = snap.labels[i]
        html = f'<p class="eq"><a name="sensor-{label}"></a>{swatch(snap.colors[i])}&nbsp;{text}</p>'
        return wash(html, snap.theme.selected_wash) if i == snap.selected else html

    def _residual_rows(self, snap: Snapshot, it: GNIteration, with_jacobian: bool) -> list[str]:
        t0 = it.estimate[2]
        rows = []
        for i, label in enumerate(snap.labels):
            text = (f"{sub('r', label)} = {num(it.distances[i])} − {num(snap.c, 1)} × ({num(snap.times[i], 6)} − "
                    f"{num(t0, 6)}) = {meters(it.residuals[i])}")
            if with_jacobian:
                ux, uy, ct = it.jacobian[i]
                text += f";&nbsp; {sub('J', label)} = [{num(ux, 4)}, {num(uy, 4)}, {num(ct, 1)}]"
            rows.append(self._block(snap, i, text))
        return rows

    def _start(self, snap: Snapshot) -> str:
        gn = self.result
        first = gn.iterations[0]
        x, y, t0 = first.estimate
        j = int(np.argmin(snap.times))
        out = [
            "<h3>Start</h3>",
            "<p>Three unknowns, x, y and t<sub>0</sub>. For any guess, each sensor's <i>residual</i> says how far "
            "the guess is from satisfying that sensor's range equation:</p>",
            '<p class="eq">r<sub>i</sub> = |p − s<sub>i</sub>| − c·(t<sub>i</sub> − t<sub>0</sub>)&nbsp;&nbsp;(meters)</p>',
            "<p>The goal is to make them all zero, or, with noise, Σr<sub>i</sub>² as small as possible.</p>",
            f"<p>The starting position is the diamond on the field (drag it): p = {point(first.estimate[:2])}. For t<sub>0</sub>, "
            f"assume the first sensor to hear the shot, {snap.labels[j]}, is exactly |p − s| away from it:</p>",
            f'<p class="eq">{T0} = {sub("t", snap.labels[j])} − |p − {sub("s", snap.labels[j])}| / c = {num(snap.times[j], 6)} − '
            f"{num(first.distances[j])} / {num(snap.c, 1)} = {seconds(t0)}</p>",
            '<p class="note">That choice hardly matters: every residual is linear in t<sub>0</sub>, so the first step fixes any '
            "error in it exactly. Only the starting <i>position</i> decides where Gauss-Newton ends up.</p>",
            "<p>The residuals at the start:</p>",
        ]
        out += self._residual_rows(snap, first, with_jacobian=False)
        out.append(f'<p class="eq">Σr<sub>i</sub>² = {big(first.cost)} m²</p>')
        out.append('<p class="note">On the field: each sensor\'s circle for this t<sub>0</sub>, and its residual as the '
                   "short colored segment from the guess to the circle.</p>")
        return "\n".join(out)

    def _iteration(self, snap: Snapshot, it: GNIteration, nxt: GNIteration) -> str:
        c = snap.c
        x, y, t0 = it.estimate
        out = [
            f"<h3>Iteration {nxt.k}</h3>",
            f"<p>From (x, y, t<sub>0</sub>) = ({num(x)}, {num(y)}, {seconds(t0)}). Near the guess, each residual is "
            "approximately a straight-line function of a small change Δ = (Δx, Δy, Δt<sub>0</sub>):</p>",
            '<p class="eq">r<sub>i</sub>(guess + Δ) ≈ r<sub>i</sub> + J<sub>i</sub>·Δ,&nbsp;&nbsp; J<sub>i</sub> = '
            "[u<sub>ix</sub>, u<sub>iy</sub>, c]</p>",
            "<p>u<sub>i</sub> is the unit vector from sensor i toward the guess (moving one meter along it adds one meter of "
            "distance), and making t<sub>0</sub> later by one second adds c meters to r<sub>i</sub>. Residuals and Jacobian rows "
            "here:</p>",
        ]
        out += self._residual_rows(snap, it, with_jacobian=True)
        out.append(
            "<p>The Δ that makes Σ(r<sub>i</sub> + J<sub>i</sub>·Δ)² smallest solves the <i>normal equations</i> "
            "JᵀJ·Δ = −Jᵀr:</p>"
        )
        out.append(f"<p>JᵀJ =</p>{matrix(it.normal_matrix)}")
        out.append(
            f'<p class="note">The big entry is Σc² = {snap.n}·c² = {big(snap.n * c * c)}, because every row ends in c: '
            "the t<sub>0</sub> column is in seconds while x and y are in meters.</p>"
        )
        out.append(f'<p class="eq">−Jᵀr = {vector(-it.gradient)}</p>')
        dx, dy, dt = it.step
        out.append(f'<p class="eq"><b>Δ = ({num(dx)} m, {num(dy)} m, {millis(dt)})</b></p>')
        nx, ny, nt = nxt.estimate
        out.append(f'<p class="eq">new guess: ({num(nx)}, {num(ny)}, {seconds(nt)})</p>')
        out.append(f'<p class="eq">Σr<sub>i</sub>²: {big(it.cost)} → {big(nxt.cost)} m²</p>')
        if nxt.k == 1:
            out.append(
                '<p class="note">Numerically the app doesn\'t form JᵀJ: it solves the equivalent least-squares problem '
                "J·Δ ≈ −r directly, with t<sub>0</sub> measured as c·t<sub>0</sub> meters so all three columns are the "
                "same size. Forming JᵀJ squares J's condition number, and far outside the array (nearly parallel "
                "u<sub>i</sub>) that turns a nearly singular problem into an exactly singular one.</p>"
            )
        out.append('<p class="note">On the field, the arrow is this step.</p>')
        return "\n".join(out)

    def _result(self, snap: Snapshot) -> str:
        gn = self.result
        final = gn.iterations[-1]
        out = ["<h3>Result</h3>", f"<p>Gauss-Newton <b>{STATUS[gn.status]}</b> after {self.n_iterations} iteration"
               f"{'s' if self.n_iterations != 1 else ''}.</p>"]
        out.append('<table class="data" cellspacing="0"><tr><th>k</th><th>x (m)</th><th>y (m)</th>'
                   "<th>t<sub>0</sub> (s)</th><th>Σr² (m²)</th></tr>")
        for it in gn.iterations:
            x, y, t0 = it.estimate
            out.append(f"<tr><td nowrap>{it.k}</td><td nowrap>{big(x)}</td><td nowrap>{big(y)}</td>"
                       f"<td nowrap>{num(t0, 6)}</td><td nowrap>{big(it.cost, 4) if it.cost >= 1e-4 else f'{it.cost:.1e}'}</td></tr>")
        out.append("</table>")

        if gn.converged:
            error = float(np.linalg.norm(final.estimate[:2] - snap.shot))
            out.append(f'<p class="eq"><b>p<sub>est</sub> = {point(final.estimate[:2])}, t<sub>0,est</sub> = '
                       f"{seconds(final.estimate[2])}</b>, {meters(error)} from the true shot</p>")
            out.append(
                "<p>Near the answer, Σr² roughly squares at every step (each digit of accuracy doubles): that fast "
                "finish is Gauss-Newton's hallmark when the residuals at the answer are zero or small.</p>"
            )
            if snap.scenario.sigma == 0 and error > 1.0 and final.cost < 1e-6:
                out.append(
                    "<p><b>Every residual is zero, yet this isn't the shot.</b> It is a second point that fits the arrival "
                    "times exactly: the other solution of an ambiguous layout. Gauss-Newton finds whichever solution is "
                    "downhill from where it starts. Drag the diamond to find the other one.</p>"
                )
        elif gn.status == "diverged":
            out.append(
                "<p>Far from the sensors the straight-line approximation is poor: the unit vectors u<sub>i</sub> from all "
                "the sensors point almost the same way, JᵀJ is nearly singular, and the step overshoots wildly. Plain "
                "Gauss-Newton has no safeguard against that (damped versions, such as Levenberg-Marquardt, shrink such "
                "steps). Drag the diamond closer to the sensors.</p>"
            )
        elif gn.status in ("singular", "hit_sensor"):
            out.append('<p class="note">Drag the diamond somewhere else.</p>')
        return "\n".join(out)
