"""Mode 1, "Known t₀": pretend we know when the shot was fired.

With t₀ handed to us, each arrival time becomes a distance, rᵢ = c·(tᵢ − t₀),
and the shot is where the circles of those radii meet. The Method tab walks
through the algebra in core/known_t0.py one step at a time, and each step
adds its own geometry to the field:

    1 Ranges      circles centered on the *shot*, one per sensor: how far the sound
                  had spread when that sensor heard it, so each passes through its sensor
    2 Circles     the same radii, now centered on the sensors (as in every later step)
    3 Subtract    each differenced equation is a straight line, drawn dashed
    4 Solve       the lines again: the estimate is where they cross
    5 Check       the catch: rerunning with a t₀ that is 10 ms off
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle

from ...core.accuracy import Fix
from ...core.known_t0 import CircleFit, solve_known_t0
from ..fmt import big, index_of, linear, meters, millis, num, plus, point, seconds, sub, swatch, term, wash
from ..snapshot import Snapshot
from .base import Mode, roadmap

STEPS = ["Ranges", "Circles", "Subtract", "Solve", "Check, and the catch"]
RANGES, CIRCLES, SUBTRACT, SOLVE, CHECK = range(len(STEPS))
T0_ERROR = 0.010  # seconds: how far off t₀ is in the last step's "what if" example


class KnownT0Mode(Mode):
    title = "Known t₀"
    estimate_source = "circles, with t₀ known"
    t0_is_given = True
    summary = (
        "Cheating mode: pretend we know the moment t₀ the shot was fired. Then each arrival "
        "time gives that sensor's exact distance to the shot, rᵢ = c·(tᵢ − t₀), and the shot "
        "is where the circles with those radii meet. Easy, and impossible in practice: nobody "
        "knows t₀. The Method tab works through it step by step."
    )

    def __init__(self) -> None:
        super().__init__()
        self.result: CircleFit | None = None
        self.wrong_t0: CircleFit | None = None  # the same method, run with t₀ + T0_ERROR

    def update(self, snap: Snapshot) -> None:
        self.result = self.wrong_t0 = self.fix = None
        if snap.shot is None:
            return
        t0 = snap.scenario.t0
        # Circles make sense with any number of sensors, so fit even when there are too few to solve.
        self.result = solve_known_t0(snap.positions, snap.times, snap.c, t0, snap.reference)
        if snap.ready:
            self.wrong_t0 = solve_known_t0(snap.positions, snap.times, snap.c, t0 + T0_ERROR, snap.reference)
        if self.result.position is not None:
            self.fix = Fix(self.result.position, t0, "known_t0")

    def step_count(self, snap: Snapshot) -> int:
        return len(STEPS) if snap.ready else 0

    # ---- field -----------------------------------------------------------

    def draw(self, ax, snap: Snapshot) -> None:
        fit = self.result
        if fit is None:
            return
        step = self.step if snap.ready else CIRCLES

        for i, (sensor, r) in enumerate(zip(snap.positions, fit.radii)):
            if r <= 0:  # a negative range ("heard before it was fired") has no circle
                continue
            width = 2.6 if i == snap.selected else 1.5
            if step == RANGES:
                # Centered on the shot, with the radius drawn out toward the sensor it reaches.
                ax.add_patch(Circle(snap.shot, r, fill=False, edgecolor=snap.colors[i], linewidth=width, zorder=3))
                toward = sensor - snap.shot
                if np.linalg.norm(toward) > 0:
                    end = snap.shot + r * toward / np.linalg.norm(toward)
                    ax.plot([snap.shot[0], end[0]], [snap.shot[1], end[1]], color=snap.colors[i], linewidth=1.4, zorder=4)
            else:
                ax.add_patch(Circle(sensor, r, fill=False, edgecolor=snap.colors[i], linewidth=width, zorder=3))

        if step in (SUBTRACT, SOLVE):
            for eq in fit.equations:
                a = np.array([eq.a_x, eq.a_y])
                if not a.any():  # two sensors in the same spot: no line
                    continue
                on_line = eq.rhs * a / (a @ a)  # the line's closest point to the origin
                along = np.array([-a[1], a[0]])
                width = 2.4 if eq.sensor == snap.selected else 1.3
                ax.axline(on_line, on_line + along, color=snap.colors[eq.sensor], linewidth=width,
                          linestyle=(0, (6, 4)), zorder=2.5)

    # ---- Method tab --------------------------------------------------------

    def method_html(self, snap: Snapshot) -> str:
        out = [f"<h2>{self.title}</h2>"]
        if snap.shot is None or snap.n < 3:
            out.append(
                '<p class="note">Place at least 3 sensors and fire a shot (shift-click) to step through '
                "this method. With only 2, the field shows why: two circles cross at two points, "
                "and nothing says which one is the shot.</p>"
            )
            return "\n".join(out)
        out.append(roadmap(STEPS, self.step))
        builders = {RANGES: self._ranges, CIRCLES: self._circles, SUBTRACT: self._subtract,
                    SOLVE: self._solve, CHECK: self._check}
        out.append(builders[self.step](snap))
        return "\n".join(out)

    def _sensor_block(self, snap: Snapshot, i: int, lines: list[str]) -> str:
        """One sensor's lines in a step, anchored so selecting the sensor scrolls here, and washed if selected."""
        label = snap.labels[i]
        first = f'<a name="sensor-{label}"></a>{swatch(snap.colors[i])}&nbsp;{lines[0]}'
        html = "\n".join(f'<p class="eq">{line}</p>' for line in [first, *lines[1:]])
        return wash(html, snap.theme.selected_wash) if i == snap.selected else html

    def _ranges(self, snap: Snapshot) -> str:
        fit, sc = self.result, snap.scenario
        out = [
            "<h3>Step 1. Ranges from the known t<sub>0</sub></h3>",
            f"<p>Pretend we know the shot was fired at t<sub>0</sub> = {seconds(sc.t0)}. The sound then "
            f"travelled for t<sub>i</sub> − t<sub>0</sub> seconds to reach sensor i, at c = {num(snap.c, 1)} m/s, "
            "so every sensor knows its distance to the shot:</p>",
            '<p class="eq">r<sub>i</sub> = c·(t<sub>i</sub> − t<sub>0</sub>)</p>',
        ]
        for i, label in enumerate(snap.labels):
            out.append(self._sensor_block(snap, i, [
                f"{sub('r', label)} = {num(snap.c, 1)} × ({num(snap.times[i], 6)} − {num(sc.t0, 6)}) = {meters(fit.radii[i])}"
            ]))
        out.append(
            '<p class="note">On the field, each circle is centered on the shot, in its sensor\'s color, with '
            "radius r<sub>i</sub>: it is how far the sound had spread when sensor i heard it, so it passes "
            "through sensor i (exactly, when there is no timing noise).</p>"
        )
        out += self._negative_range_notes(snap)
        return "\n".join(out)

    def _negative_range_notes(self, snap: Snapshot) -> list[str]:
        return [
            f'<p class="note">{label}\'s range is negative: with this much timing noise it "heard" the shot '
            "before it was fired. A negative range has no circle, so none is drawn for it.</p>"
            for label, r in zip(snap.labels, self.result.radii) if r <= 0
        ]

    def _circles(self, snap: Snapshot) -> str:
        fit = self.result
        out = [
            "<h3>Step 2. Each range is a circle</h3>",
            "<p>Now turn step 1's picture around. Sensor i doesn't know where the shot is, only that it is "
            "r<sub>i</sub> away: a range says how far, not which way. So the shot p = (x, y) is somewhere on "
            "the circle of radius r<sub>i</sub> around s<sub>i</sub> = (x<sub>i</sub>, y<sub>i</sub>), and it "
            "is on all of these circles at once:</p>",
            '<p class="eq">(x − x<sub>i</sub>)² + (y − y<sub>i</sub>)² = r<sub>i</sub>²</p>',
        ]
        for i, (x, y) in enumerate(snap.positions):
            out.append(self._sensor_block(snap, i, [
                f"(x − {term(x)})² + (y − {term(y)})² = {term(fit.radii[i])}² = {big(fit.radii_sq[i])}"
            ]))
        out.append(
            '<p class="note">Two circles cross in at most two points, and a third circle picks one of them. '
            "On the field, every circle passes through the shot.</p>"
        )
        out.append(
            "<p>Solving circle equations together means solving quadratics. Multiplying out the squares "
            "shows a way around that:</p>"
            '<p class="eq">x² + y² − 2x<sub>i</sub>·x − 2y<sub>i</sub>·y + (x<sub>i</sub>² + y<sub>i</sub>²) = r<sub>i</sub>²</p>'
        )
        for i, (x, y) in enumerate(snap.positions):
            label = snap.labels[i]
            out.append(self._sensor_block(snap, i, [
                f"x² + y²{plus(-2 * x)}·x{plus(-2 * y)}·y{plus(fit.norms_sq[i])} = {big(fit.radii_sq[i])}"
                f'&nbsp;&nbsp;<span class="muted">({label})</span>'
            ]))
        out.append('<p class="note">Every one of them starts with the same x² + y².</p>')
        return "\n".join(out)

    def _subtract(self, snap: Snapshot) -> str:
        fit = self.result
        k = snap.reference
        ref = snap.labels[k]
        kk = index_of(ref)
        xk, yk = snap.positions[k]
        out = [
            "<h3>Step 3. Subtract to cancel x² + y²</h3>",
            f"<p>Subtract the reference {ref}'s equation from each of the others. The x² + y² cancels, "
            "and what's left is linear in x and y:</p>",
            f'<p class="eq">2(x<sub>i</sub> − x<sub>{kk}</sub>)·x + 2(y<sub>i</sub> − y<sub>{kk}</sub>)·y = '
            f"(|s<sub>i</sub>|² − |s<sub>{kk}</sub>|²) − (r<sub>i</sub>² − r<sub>{kk}</sub>²)</p>",
        ]
        for eq in fit.equations:
            xi, yi = snap.positions[eq.sensor]
            out.append(self._sensor_block(snap, eq.sensor, [
                f"2({term(xi)} − {term(xk)})·x + 2({term(yi)} − {term(yk)})·y = "
                f"({big(fit.norms_sq[eq.sensor])} − {big(fit.norms_sq[k])}) − "
                f"({big(fit.radii_sq[eq.sensor])} − {big(fit.radii_sq[k])})",
                linear(eq.a_x, eq.a_y, eq.rhs),
            ]))
        out.append(self._sensor_block(snap, k, [f"{ref} is the reference: its equation was subtracted from the others."]))
        out.append(
            "<p class=\"note\">On the field, each of these equations is a dashed line in its sensor's color: "
            f"the line through the two points where that sensor's circle crosses {ref}'s. Anything on both "
            "circles satisfies both equations, so it satisfies their difference too. The lines cross at the shot.</p>"
        )
        return "\n".join(out)

    def _solve(self, snap: Snapshot) -> str:
        fit = self.result
        M, v, det = fit.system_matrix, fit.system_rhs, fit.determinant
        out = ["<h3>Step 4. Solve the linear equations</h3>"]
        if not fit.normal_equations:
            out.append("<p>Two lines, two unknowns. In general form</p>")
        else:
            n = len(fit.equations)
            out.append(
                f"<p>{n} lines but only 2 unknowns. Without noise they all pass through one point; with noise "
                "they don't, and least squares picks the p that makes the sum of squared equation errors, "
                "Σ(a<sub>i</sub>·p − b<sub>i</sub>)², as small as possible. Setting that sum's derivative to zero "
                "gives the 2×2 <i>normal equations</i> AᵀA·p = Aᵀb, where each row of A holds one line's "
                "x and y coefficients and b the right-hand sides. Multiplied out, in general form</p>"
            )
        out.append('<p class="eq">m<sub>11</sub>·x + m<sub>12</sub>·y = v<sub>1</sub><br>'
                   "m<sub>21</sub>·x + m<sub>22</sub>·y = v<sub>2</sub></p>")
        if fit.normal_equations:
            out.append(
                '<p class="note">with m<sub>11</sub> = Σa<sub>ix</sub>², m<sub>12</sub> = m<sub>21</sub> = '
                "Σa<sub>ix</sub>a<sub>iy</sub>, m<sub>22</sub> = Σa<sub>iy</sub>², v<sub>1</sub> = "
                "Σa<sub>ix</sub>b<sub>i</sub>, v<sub>2</sub> = Σa<sub>iy</sub>b<sub>i</sub>. Here:</p>"
            )
        else:
            out.append("<p>Here:</p>")
        out.append(f'<p class="eq">{linear(M[0, 0], M[0, 1], v[0])}<br>{linear(M[1, 0], M[1, 1], v[1])}</p>')

        out.append("<p>Cramer's rule solves it:</p>")
        out.append(
            '<p class="eq">det = m<sub>11</sub>m<sub>22</sub> − m<sub>12</sub>m<sub>21</sub> = '
            f"{term(M[0, 0])} × {term(M[1, 1])} − {term(M[0, 1])} × {term(M[1, 0])} = {big(det)}</p>"
        )
        if fit.position is None:
            out.append(
                '<p class="note">det = 0 (to rounding): the lines are parallel and never cross at one point. '
                "All the sensors are on one straight line, and a shot on one side of it produces exactly the "
                "same arrival times as its mirror image on the other side, so no method can tell them apart. "
                "Move a sensor off the line.</p>"
            )
            return "\n".join(out)
        x, y = fit.position
        out.append(
            '<p class="eq">x = (v<sub>1</sub>m<sub>22</sub> − m<sub>12</sub>v<sub>2</sub>) / det = '
            f"({term(v[0])} × {term(M[1, 1])} − {term(M[0, 1])} × {term(v[1])}) / {term(det)} = {num(x)}</p>"
        )
        out.append(
            '<p class="eq">y = (m<sub>11</sub>v<sub>2</sub> − v<sub>1</sub>m<sub>21</sub>) / det = '
            f"({term(M[0, 0])} × {term(v[1])} − {term(v[0])} × {term(M[1, 0])}) / {term(det)} = {num(y)}</p>"
        )
        out.append(f'<p class="eq"><b>p<sub>est</sub> = {point(fit.position)}</b></p>')
        out.append('<p class="note">On the field, the estimate (circle and plus) is where the dashed lines cross.</p>')
        return "\n".join(out)

    def _check(self, snap: Snapshot) -> str:
        fit, sc = self.result, snap.scenario
        out = ["<h3>Step 5. Check, and the catch</h3>"]
        if fit.position is None:
            out.append('<p class="note">There is no estimate to check: see step 4.</p>')
        else:
            out.append("<p>Put p<sub>est</sub> back into each circle: is it at distance r<sub>i</sub> from s<sub>i</sub>?</p>")
            for i, label in enumerate(snap.labels):
                d = fit.radii[i] + fit.mismatch[i]
                out.append(self._sensor_block(snap, i, [
                    f"|p<sub>est</sub> − {sub('s', label)}| = {meters(d)},&nbsp; {sub('r', label)} = {meters(fit.radii[i])},"
                    f"&nbsp; off by {meters(fit.mismatch[i], 3)}"
                ]))
            if sc.sigma > 0:
                error = float(np.linalg.norm(fit.position - snap.shot))
                out.append(
                    f'<p class="note">With timing noise σ = {millis(sc.sigma, 2)}, each radius is off by '
                    "c × (that sensor's noise), so the circles don't quite meet at one point. p<sub>est</sub> is the "
                    f"least-squares compromise between the linear equations of step 4, {meters(error)} from the true shot.</p>"
                )
            else:
                out.append(
                    '<p class="note">Every mismatch is zero: with t<sub>0</sub> known and perfect clocks, the '
                    "circles meet exactly at the shot.</p>"
                )

        out.append("<h3>The catch: nobody knows t<sub>0</sub></h3>")
        out.append(
            "<p>All of that was easy because we were handed t<sub>0</sub>. In reality a gunshot carries no "
            "timestamp: the sensors only learn when the sound <i>arrived</i>. Their clocks are synchronized "
            "with each other (by GPS, in real systems), so the differences between their arrival times can be "
            "trusted, but the gun isn't on that clock.</p>"
        )
        catch = (
            f"<p>And t<sub>0</sub> matters. Every millisecond of error in t<sub>0</sub> changes every radius by "
            f"c × 1 ms = {meters(snap.c * 1e-3)}, all in the same direction. Run the same method with "
            f"t<sub>0</sub> {num(T0_ERROR * 1e3, 0)} ms late ({seconds(sc.t0 + T0_ERROR)}): every radius shrinks "
            f"by {meters(snap.c * T0_ERROR)}, so the circles no longer meet at one point"
        )
        wrong = self.wrong_t0
        if wrong is not None and wrong.position is not None:
            miss = float(np.linalg.norm(wrong.position - snap.shot))
            worst = float(np.max(np.abs(wrong.mismatch)))
            catch += (
                f". The method's best point misses the circles by up to {meters(worst)}, and lands "
                f"{meters(miss)} from the shot."
            )
        out.append(catch + "</p>")
        out.append(
            "<p>How far the point moves depends on the layout. What always happens is that the circles stop "
            "meeting, and that is the clue the next mode uses.</p>"
            '<p class="note">t<sub>0</sub> sweep turns the problem around: guess t<sub>0</sub>, measure how badly '
            "the circles fail to meet, and keep the guess where they meet.</p>"
        )
        return "\n".join(out)
