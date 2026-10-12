"""Mode 4, "Closed form": square, subtract, and solve, with no searching at all.

The derivation is in core/closed_form.py. Which steps there are depends on the
route the solver took:

    3 sensors (quadratic)      Square → Subtract → p = u + v·t₀ → Quadratic → Which root?
    4+ sensors (least squares) Square → Subtract → Solve
    4+, t₀ drops out           Square → Subtract → Rank → p = u + v·t₀ → Quadratic → Which root?
    collinear sensors          Square → Subtract → Collinear

Geometry on the field: in Subtract (and Solve), each linear equation at the
solved t₀ (the straight lines of Known t₀); from p = u + v·t₀ on, the line of
candidate points the equations allow, with the quadratic's roots marked on it.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from ...core.accuracy import Fix
from ...core.closed_form import ClosedFormResult, solve_closed_form
from ...core.gauss_newton import GNResult, gauss_newton
from ..fmt import big, index_of, length, linear, linear_terms, meters, num, plus, point, seconds, sub, swatch, term, wash
from ..snapshot import Snapshot
from .base import Mode, roadmap

T0 = "t<sub>0</sub>"
ROUTES = {
    "quadratic": ["Square", "Subtract", "p = u + v·t₀", "Quadratic", "Which root?"],
    "least_squares": ["Square", "Subtract", "Solve"],
    "constrained": ["Square", "Subtract", "Rank", "p = u + v·t₀", "Quadratic", "Which root?"],
    "degenerate": ["Square", "Subtract", "Collinear"],
}


class ClosedFormMode(Mode):
    title = "Closed form"
    estimate_source = "closed form"
    summary = (
        "No searching: square each range equation, subtract the reference sensor's to cancel the squared "
        "unknowns, and what's left is linear in x, y and t₀. With 4+ sensors that linear system is solved "
        "directly. With exactly 3 it is one equation short, so the shot is pinned to a line, p = u + v·t₀, "
        "and putting that back into one original equation gives a quadratic in t₀: two roots, which is "
        "where the two-solution ambiguity comes from."
    )

    def __init__(self) -> None:
        super().__init__()
        self.result: ClosedFormResult | None = None
        # Gauss-Newton run from the 4+-sensor linear answer, to show what that answer leaves on the table.
        self.polished: GNResult | None = None

    @property
    def steps(self) -> list[str]:
        return ROUTES.get(self.result.method, []) if self.result is not None else []

    @property
    def step_name(self) -> str:
        steps = self.steps
        return steps[self.step] if self.step < len(steps) else ""

    def update(self, snap: Snapshot) -> None:
        self.result, self.fix, self.polished = None, None, None
        if not snap.ready:
            return
        self.result = solve_closed_form(snap.positions, snap.times, snap.c, snap.reference)
        if self.result.position is not None:
            self.fix = Fix(self.result.position, self.result.t0, "closed_form")
            if self.result.method == "least_squares":
                self.polished = gauss_newton(snap.positions, snap.times, snap.c, self.result.position, self.result.t0)

    def step_count(self, snap: Snapshot) -> int:
        return len(self.steps)

    def legend_handles(self, snap: Snapshot) -> list:
        r = self.result
        if r is None or r.quadratic is None or self.step_name not in ("p = u + v·t₀", "Quadratic", "Which root?"):
            return []
        t = snap.theme
        handles = [Line2D([], [], color=t.ink_secondary, linestyle=(0, (1, 2)), linewidth=1.6, label="p = u + v·t₀")]
        if self.step_name != "p = u + v·t₀":
            handles.append(Line2D([], [], marker="s", markerfacecolor="none", color=t.ink, linestyle="none",
                                  markersize=9, label="root kept"))
            if any(not cand.valid for cand in r.candidates):
                handles.append(Line2D([], [], marker="x", color=t.muted, linestyle="none", markersize=9,
                                      label="root rejected"))
        return handles

    # ---- field -----------------------------------------------------------

    def draw(self, ax, snap: Snapshot) -> None:
        r, name, t = self.result, self.step_name, snap.theme
        if r is None:
            return
        if name in ("Subtract", "Solve") and r.t0 is not None:
            # At the solved t₀, each linear equation is a line through the shot (Known t₀'s lines).
            for eq in r.equations:
                a = np.array([eq.a_x, eq.a_y])
                if not a.any():
                    continue
                on_line = (eq.rhs - eq.a_t * r.t0) * a / (a @ a)
                width = 2.4 if eq.sensor == snap.selected else 1.3
                ax.axline(on_line, on_line + np.array([-a[1], a[0]]), color=snap.colors[eq.sensor], linewidth=width,
                          linestyle=(0, (6, 4)), zorder=2.5)
        q = r.quadratic
        if q is not None and name in ("p = u + v·t₀", "Quadratic", "Which root?") and np.any(q.v):
            ax.axline(q.u, q.u + q.v, color=t.ink_secondary, linestyle=(0, (1, 2)), linewidth=1.6, zorder=6)
            if name != "p = u + v·t₀":
                for cand in r.candidates:
                    if cand.valid:
                        ax.scatter(*cand.position, marker="s", s=110, facecolors="none", edgecolors=t.ink,
                                   linewidths=1.5, zorder=13)
                    else:
                        ax.scatter(*cand.position, marker="x", s=90, color=t.muted, linewidths=2.0, zorder=13)

    # ---- Method tab --------------------------------------------------------

    def method_html(self, snap: Snapshot) -> str:
        out = [f"<h2>{self.title}</h2>"]
        r = self.result
        if r is None:
            out.append('<p class="note">Place at least 3 sensors and fire a shot (shift-click) to step through this mode.</p>')
            return "\n".join(out)
        out.append(roadmap(self.steps, self.step))
        builders = {
            "Square": self._square, "Subtract": self._subtract, "p = u + v·t₀": self._line, "Quadratic": self._quadratic,
            "Which root?": self._roots, "Solve": self._solve, "Rank": self._rank, "Collinear": self._collinear,
        }
        out.append(f"<h3>Step {self.step + 1}. {self._heading()}</h3>")
        out.append(builders[self.step_name](snap))
        return "\n".join(out)

    def _heading(self) -> str:
        return {
            "Square": "Square each range equation", "Subtract": "Subtract to cancel the squares",
            "p = u + v·t₀": f"One equation short: p = u + v·{T0}", "Quadratic": f"A quadratic in {T0}",
            "Which root?": "Which root?", "Solve": "Solve the linear system", "Rank": f"{T0} drops out",
            "Collinear": "Collinear sensors",
        }[self.step_name]

    def _block(self, snap: Snapshot, i: int, lines: list[str], anchor: bool = True) -> str:
        label = snap.labels[i]
        first = (f'<a name="sensor-{label}"></a>' if anchor else "") + f"{swatch(snap.colors[i])}&nbsp;{lines[0]}"
        html = "\n".join(f'<p class="eq">{line}</p>' for line in [first, *lines[1:]])
        return wash(html, snap.theme.selected_wash) if i == snap.selected else html

    def _ref_block(self, snap: Snapshot, text: str) -> str:
        return self._block(snap, snap.reference, [text])

    def _square(self, snap: Snapshot) -> str:
        r, c = self.result, snap.c
        eq0 = r.equations[0]
        out = [
            "<p>Each sensor's range equation, |p − s<sub>i</sub>| = c·(t<sub>i</sub> − t<sub>0</sub>), has three "
            "unknowns: x, y and t<sub>0</sub>. Square both sides and multiply out:</p>",
            '<p class="eq">x² + y² − 2x<sub>i</sub>·x − 2y<sub>i</sub>·y + |s<sub>i</sub>|² = '
            "c²t<sub>i</sub>² − 2c²t<sub>i</sub>·t<sub>0</sub> + c²t<sub>0</sub>²&nbsp;&nbsp;(E<sub>i</sub>)</p>",
            f"<p>With c = {num(c, 1)} m/s, so c² = {big(c * c)}:</p>",
        ]
        for i, (x, y) in enumerate(snap.positions):
            if i == snap.reference:
                s_sq, ct_sq = eq0.ref_s_sq, eq0.ref_ct_sq
            else:
                eq = next(e for e in r.equations if e.sensor == i)
                s_sq, ct_sq = eq.s_sq, eq.ct_sq
            ti = snap.times[i]
            out.append(self._block(snap, i, [
                f"x² + y²{plus(-2 * x)}·x{plus(-2 * y)}·y{plus(s_sq)} = {big(ct_sq)}{plus(-2 * c * c * ti)}·{T0}"
                f"{plus(c * c)}·{T0}²"
            ]))
        out.append(
            "<p>Still not linear: x² + y² on the left and c²t<sub>0</sub>² on the right. But both are the same in "
            "every sensor's equation.</p>"
        )
        return "\n".join(out)

    def _subtract(self, snap: Snapshot) -> str:
        r, c = self.result, snap.c
        k = snap.reference
        ref, kk = snap.labels[k], index_of(snap.labels[k])
        xk, yk = snap.positions[k]
        tk = snap.times[k]
        out = [
            f"<p>Subtract the reference {ref}'s equation E<sub>{kk}</sub> from each of the others. x² + y² and "
            "c²t<sub>0</sub>² cancel, and what's left is <i>linear</i> in x, y and t<sub>0</sub>:</p>",
            f'<p class="eq">2(x<sub>i</sub> − x<sub>{kk}</sub>)·x + 2(y<sub>i</sub> − y<sub>{kk}</sub>)·y '
            f"− 2c²(t<sub>i</sub> − t<sub>{kk}</sub>)·{T0} = (|s<sub>i</sub>|² − |s<sub>{kk}</sub>|²) − "
            f"(c²t<sub>i</sub>² − c²t<sub>{kk}</sub>²)&nbsp;&nbsp;(L<sub>i</sub>)</p>",
        ]
        for eq in r.equations:
            xi, yi = snap.positions[eq.sensor]
            ti = snap.times[eq.sensor]
            out.append(self._block(snap, eq.sensor, [
                f"2({term(xi)} − {term(xk)})·x + 2({term(yi)} − {term(yk)})·y − 2 × {big(c * c)} × "
                f"({num(ti, 6)} − {num(tk, 6)})·{T0} = ({big(eq.s_sq)} − {big(eq.ref_s_sq)}) − "
                f"({big(eq.ct_sq)} − {big(eq.ref_ct_sq)})",
                linear_terms([eq.a_x, eq.a_y, eq.a_t], ["x", "y", T0], eq.rhs),
            ]))
        out.append(self._ref_block(snap, f"{ref} is the reference: its equation was subtracted from the others."))
        n_eq = len(r.equations)
        if r.method == "quadratic":
            more = "That is 2 equations for 3 unknowns: one short."
        elif r.method == "degenerate":
            more = f"That is {n_eq} equations, but they can't separate x from y."
        else:
            more = f"That is {n_eq} equations for 3 unknowns."
        out.append(f"<p>{more}</p>")
        if r.t0 is not None:
            out.append(
                f'<p class="note">For any one value of t<sub>0</sub> each L<sub>i</sub> is a straight line. On the field '
                f"they're drawn for the solved t<sub>0</sub> = {seconds(r.t0)}: then they are Known t<sub>0</sub>'s lines "
                "through the shot.</p>"
            )
        return "\n".join(out)

    def _line(self, snap: Snapshot) -> str:
        r = self.result
        q = r.quadratic
        out = []
        if r.method == "quadratic":
            out.append("<p>Two equations can't fix three unknowns, but they can fix x and y <i>for any given</i> "
                       f"t<sub>0</sub>. Move the t<sub>0</sub> terms to the right:</p>")
        else:
            out.append("<p>As with 3 sensors, solve the linear equations for x and y in terms of t<sub>0</sub> "
                       "(by least squares, since there are more equations than the two unknowns). With the "
                       "t<sub>0</sub> terms moved to the right:</p>")
        for eq in r.equations:
            out.append(self._block(snap, eq.sensor, [
                f"{linear(eq.a_x, eq.a_y, eq.rhs)}{plus(-eq.a_t)}·{T0}"
            ]))
        out.append(
            "<p>Solving for x and y gives a point that moves in a straight line as t<sub>0</sub> changes:</p>"
            f'<p class="eq"><b>p = u + v·{T0}</b>,&nbsp; u = {point(q.u)} m,&nbsp; v = {point(q.v)} m/s</p>'
        )
        speed = float(np.linalg.norm(q.v))
        out.append(
            f"<p>Whatever t<sub>0</sub> is, the shot is on this line (dotted on the field). Each millisecond of "
            f"t<sub>0</sub> slides the point {meters(speed * 1e-3)} along it (|v| = {num(speed, 1)} m/s).</p>"
            '<p class="note">One equation hasn\'t been used yet: the reference sensor\'s own E<sub>k</sub>. That is '
            "what picks the point on the line.</p>"
        )
        return "\n".join(out)

    def _quadratic(self, snap: Snapshot) -> str:
        r, c = self.result, snap.c
        q = r.quadratic
        k = snap.reference
        kk = index_of(snap.labels[k])
        tk = snap.times[k]
        out = [
            f"<p>Put p = u + v·t<sub>0</sub> into the reference sensor's range equation, |p − s<sub>{kk}</sub>|² = "
            f"c²(t<sub>{kk}</sub> − t<sub>0</sub>)². With w = u − s<sub>{kk}</sub> = {point(q.w)}:</p>",
            f'<p class="eq">|w + v·{T0}|² = c²(t<sub>{kk}</sub> − {T0})²</p>',
            f'<p class="eq">A·{T0}² + B·{T0} + C = 0,&nbsp; with A = |v|² − c²,&nbsp; B = 2(w·v + c²t<sub>{kk}</sub>),'
            f"&nbsp; C = |w|² − c²t<sub>{kk}</sub>²</p>",
            "<p>With the numbers:</p>",
            f'<p class="eq">A = {big(float(q.v @ q.v))} − {big(c * c)} = {big(q.qa)}</p>',
            f'<p class="eq">B = 2 × ({big(float(q.w @ q.v))} + {big(c * c)} × {num(tk, 6)}) = {big(q.qb)}</p>',
            f'<p class="eq">C = {big(float(q.w @ q.w))} − {big((c * tk) ** 2)} = {big(q.qc)}</p>',
            f'<p class="eq"><b>{big(q.qa)}·{T0}²{plus(q.qb)}·{T0}{plus(q.qc)} = 0</b></p>',
            f'<p class="eq">D = B² − 4AC = {big(q.discriminant)}</p>',
            '<p class="note">D is the discriminant: it decides how many real roots there are, two if positive, one if '
            "zero, none if negative.</p>",
        ]
        roots = [cand.t0 for cand in r.candidates]
        if len(roots) == 2:
            out.append(f'<p class="eq">{T0} = (−B ± √D) / 2A = {seconds(roots[0])} or {seconds(roots[1])}</p>')
        elif len(roots) == 1:
            out.append(f'<p class="eq">D ≈ 0: a single (double) root, {T0} = {seconds(roots[0])}</p>'
                       '<p class="note">With noise D can come out slightly negative: there is then no exact solution, '
                       "and the vertex −B/2A is the t<sub>0</sub> that comes closest.</p>")
        out.append(
            '<p class="note">Each coefficient is a difference of numbers around a million (look at C), and subtracting '
            f"nearly equal large numbers loses digits. So the app computes the roots from the same quadratic written "
            f"in τ = t<sub>0</sub> − t<sub>{kk}</sub>, where the numbers stay small; the roots are identical.</p>"
        )
        out.append('<p class="note">On the field, each root is a point on the dotted line: squares are kept, crosses rejected '
                   "(next step).</p>")
        return "\n".join(out)

    def _roots(self, snap: Snapshot) -> str:
        r = self.result
        out = [
            "<p>Squaring threw away the sign of every range: c·(t<sub>i</sub> − t<sub>0</sub>) and its negative square "
            "to the same thing. So check each root: t<sub>0</sub> can't be later than any arrival time, or that "
            "sensor would have heard the shot before it was fired.</p>"
        ]
        for n, cand in enumerate(r.candidates, start=1):
            worst = snap.labels[cand.worst_sensor]
            smallest = cand.ranges[cand.worst_sensor]
            if cand.valid:
                verdict = f"every range is positive (smallest: {worst}, {meters(smallest)}): <b>kept</b>"
            else:
                verdict = (f"{worst} would have heard it {num(-smallest / snap.c * 1e3, 1)} ms before it was fired "
                           f"(range {meters(smallest)}): <b>rejected</b>")
            out.append(f'<p class="eq">Root {n}: {T0} = {seconds(cand.t0)}, p = {point(cand.position)}; {verdict}</p>')
        valid = r.valid_candidates
        if r.ambiguous:
            out.append(
                "<p><b>Both roots survive.</b> Two different shots, fired at two different times, produce exactly "
                "the same arrival times at all three sensors, so no algorithm can choose between them: this is a "
                "real ambiguity, not a flaw in the method. (They are the two dips of the t<sub>0</sub> sweep and the "
                "two crossings of the hyperbolas.) The app reports the later t<sub>0</sub>, the nearer shot, but "
                "only a 4th sensor can settle it.</p>"
            )
        elif len(valid) == 1 and len(r.candidates) == 2:
            out.append("<p>One root survives, so the answer is unique.</p>")
        elif not valid:
            out.append("<p>Neither root survives: timing noise has made the arrival times impossible to satisfy "
                       "exactly with positive ranges.</p>")
        if r.position is not None:
            out.append(f'<p class="eq"><b>p<sub>est</sub> = {point(r.position)}, t<sub>0,est</sub> = {seconds(r.t0)}</b></p>')
        return "\n".join(out)

    def _solve(self, snap: Snapshot) -> str:
        r = self.result
        n_eq = len(r.equations)
        x, y, t0 = r.lstsq_solution
        exact = n_eq == 3
        out = [
            f"<p>{n_eq} linear equations, 3 unknowns. "
            + ("Exactly enough: the system has one solution, found directly." if exact else
               "More equations than unknowns, so (with any noise) they can't all hold exactly; least squares finds the "
               "(x, y, t<sub>0</sub>) that makes the sum of their squared errors smallest.")
            + "</p>",
            "<p>One practical detail: the t<sub>0</sub> column is about c² ≈ 10⁵ times larger than the x and y columns "
            "(seconds against meters), so each column is scaled to unit length before solving, and the answer scaled "
            "back. Without that, rounding in the big column would swamp the small ones.</p>",
        ]
        for eq in r.equations:
            out.append(self._block(snap, eq.sensor, [linear_terms([eq.a_x, eq.a_y, eq.a_t], ["x", "y", T0], eq.rhs)]))
        out.append(f'<p class="eq"><b>x = {num(x)},&nbsp; y = {num(y)},&nbsp; {T0} = {seconds(t0)}</b></p>')
        out.append('<p class="note">On the field, the equations are drawn as lines for the solved t<sub>0</sub>: they '
                   "cross at the answer.</p>")
        cand = r.candidates[0]
        if cand.valid:
            out.append(f"<p>Check: {T0} is earlier than every arrival time, so every range is positive.</p>")
        else:
            out.append(f'<p class="note">{r.message}</p>')
        out.append("<p>No quadratic, so no second root: with 4 or more sensors the ambiguity of three disappears.</p>")
        out.append(self._polish_html(snap))
        return "\n".join(out)

    def _polish_html(self, snap: Snapshot) -> str:
        """What the linear answer leaves out, shown as the range residuals rᵢ there and what Gauss-Newton makes of them."""
        r, gn, c = self.result, self.polished, snap.c
        if gn is None:
            return ""
        start, final = gn.iterations[0], gn.iterations[-1]
        kk = index_of(snap.labels[snap.reference])
        if len(r.equations) == 3:
            out = [
                "<p>One equation went unused. Each L<sub>i</sub> is the difference of two range equations, so these three "
                f"involve all four sensors, but the reference's own equation E<sub>{kk}</sub> was only ever subtracted from "
                "the others, never solved. With exact arrival times that costs nothing: the answer satisfies all four range "
                f"equations. With noise the three differences still hold exactly, but a difference can hold while both "
                "of its equations are off by the same amount, and nothing here pins that amount down. The range residuals "
                "r<sub>i</sub> = |p − s<sub>i</sub>| − c·(t<sub>i</sub> − t<sub>0</sub>) at this answer show it:</p>"
            ]
        else:
            out = [
                "<p>Least squares here minimizes the errors of the squared, differenced equations, measured in m², which "
                "weights the sensors differently from the range residuals r<sub>i</sub> = |p − s<sub>i</sub>| − "
                "c·(t<sub>i</sub> − t<sub>0</sub>). So with noise it is close to, but not quite, the point that makes "
                "Σr<sub>i</sub>² smallest. The residuals at this answer:</p>"
            ]
        for i, label in enumerate(snap.labels):
            out.append(self._block(snap, i, [
                f"{sub('r', label)} = {num(start.distances[i])} − {num(c, 1)} × ({num(snap.times[i], 6)} − "
                f"{num(start.estimate[2], 6)}) = {meters(start.residuals[i], 3)}"
            ], anchor=False))
        if start.cost < 1e-6:
            out.append(f'<p class="eq">Σr<sub>i</sub>² = {start.cost:.1e} m²: with exact arrival times there is nothing to polish.</p>')
            return "\n".join(out)
        out.append(f'<p class="eq">Σr<sub>i</sub>² = {big(start.cost, 3)} m²</p>')
        n_it = len(gn.iterations) - 1
        moved = float(np.linalg.norm(final.estimate[:2] - start.estimate[:2]))
        if gn.converged:
            out.append(
                f"<p>Gauss-Newton started from this answer, using all {snap.n} range equations directly, brings that to "
                f"Σr<sub>i</sub>² = {big(final.cost, 3)} m² in {n_it} iteration{'s' if n_it != 1 else ''}, moving the "
                f"estimate {length(moved)}. That polish is what the Field view's \"closed form, polished by Gauss-Newton\" "
                "means, and it is how a real system would use this answer: as the starting point.</p>"
            )
        else:
            out.append(f'<p class="note">Gauss-Newton from this answer did not converge ({gn.status}).</p>')
        return "\n".join(out)

    def _rank(self, snap: Snapshot) -> str:
        r = self.result
        sv = ", ".join(num(s, 4) for s in r.singular_values)
        return "\n".join([
            "<p>Normally 3+ linear equations fix x, y and t<sub>0</sub>. Here they can't: the t<sub>0</sub> column is a "
            "combination of the x and y columns, so t<sub>0</sub> drops out of the system entirely. This happens even in "
            "good layouts, for example a square of sensors with the shot on one of its lines of symmetry.</p>",
            "<p>The test is the system's <i>singular values</i>: how strongly it responds along each of three "
            "independent directions in (x, y, t<sub>0</sub>), a zero one being a direction the equations can't see "
            f"(computed with each column scaled to unit length): {sv}. The last is zero to rounding, so the matrix has "
            "rank 2, not 3: only two independent equations, which is what the <i>rank</i> counts.</p>",
            "<p>The way out is the 3-sensor route: solve for x and y in terms of t<sub>0</sub>, then use a quadratic "
            "for t<sub>0</sub>.</p>",
        ])

    def _collinear(self, snap: Snapshot) -> str:
        return (
            "<p>The sensors are on one straight line, so the x and y columns of the linear system are proportional: "
            "the equations can't separate x from y. Geometrically, a shot on one side of the line and its mirror image "
            "on the other produce exactly the same arrival times. Move a sensor off the line.</p>"
        )
