"""Mode 3, "TDOA hyperbolas": subtract arrival times and t₀ disappears.

For each pair of sensors, the difference of their arrival times fixes the
difference of their distances to the shot, and the points with that
difference form one branch of a hyperbola. Each branch is drawn as the zero
contour of the signed function in core/tdoa.py, on a grid over the visible
part of the field (the canvas asks for a redraw when the view changes).

Steps and their geometry on the field (the curves are drawn in every step):

    1 Time differences
    2 Hyperbolas
    3 Why the sign     + the mirror-image branch that squaring would let in, dotted
    4 Crossing
    5 Far field        + each branch's asymptotes (also shown in any step by the checkbox)
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from ...core.accuracy import locate
from ...core.closed_form import Candidate, solve_closed_form
from ...core.tdoa import TdoaPair, branch_asymptotes, crossing_angle, hyperbola_function, tdoa_pairs
from ..fmt import index_of, meters, millis, num, point, seconds, sub, swatch, term, wash
from ..snapshot import Snapshot
from .base import Mode, roadmap

STEPS = ["Time differences", "Hyperbolas", "Why the sign", "Crossing", "Far field"]
DIFFERENCES, CURVES, SIGN, CROSSING, FAR = range(len(STEPS))
GRID = 320  # contour grid points along the longer side of the view


class HyperbolaMode(Mode):
    title = "TDOA hyperbolas"
    estimate_source = "where the hyperbolas cross (closed form, polished by Gauss-Newton)"
    summary = (
        "Subtract two sensors' arrival times and t₀ cancels: the time difference Δt says how much farther "
        "the shot is from one sensor than from the other, c·Δt, without either sensor knowing its own "
        "distance. All the points with that difference of distances form one branch of a hyperbola, "
        "and the branches for different pairs cross at the shot."
    )

    def __init__(self) -> None:
        super().__init__()
        self.all_pairs = False
        self.show_asymptotes = False
        self.reference_pairs: list[TdoaPair] = []
        self.extra_pairs: list[TdoaPair] = []  # pairs not involving the reference, shown with "All pairs"
        self.candidates: list[Candidate] = []  # with 3 sensors: every point where the two curves cross
        self._controls: QWidget | None = None

    def controls(self) -> QWidget:
        if self._controls is None:
            self._controls = QWidget()
            row = QHBoxLayout(self._controls)
            row.setContentsMargins(0, 0, 0, 0)
            pairs = QCheckBox("All pairs")
            pairs.setToolTip("Also draw the pairs that don't include the reference sensor")
            asymptotes = QCheckBox("Asymptotes")
            asymptotes.setToolTip("Draw each branch's straight-line asymptotes")
            pairs.toggled.connect(lambda on: self._set("all_pairs", on))
            asymptotes.toggled.connect(lambda on: self._set("show_asymptotes", on))
            row.addWidget(pairs)
            row.addWidget(asymptotes)
        return self._controls

    def _set(self, name: str, value: bool) -> None:
        setattr(self, name, value)
        self.changed.emit()

    @property
    def pairs(self) -> list[TdoaPair]:
        return self.reference_pairs + (self.extra_pairs if self.all_pairs else [])

    def update(self, snap: Snapshot) -> None:
        self.reference_pairs, self.extra_pairs, self.candidates, self.fix = [], [], [], None
        if snap.shot is None or snap.n < 2:
            return
        k = snap.reference
        self.reference_pairs = tdoa_pairs(snap.positions, snap.times, snap.c, k)
        self.extra_pairs = [p for p in tdoa_pairs(snap.positions, snap.times, snap.c, k, all_pairs=True) if k not in (p.i, p.j)]
        if snap.ready:
            self.fix = locate(snap.positions, snap.times, snap.c, k)
            if snap.n == 3:
                self.candidates = solve_closed_form(snap.positions, snap.times, snap.c, k).valid_candidates

    def step_count(self, snap: Snapshot) -> int:
        return len(STEPS) if snap.ready else 0

    # ---- field -----------------------------------------------------------

    def draw(self, ax, snap: Snapshot) -> None:
        if not self.reference_pairs:
            return
        step = self.step if snap.ready else CURVES
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        scale = GRID / max(x1 - x0, y1 - y0)
        X, Y = np.meshgrid(np.linspace(x0, x1, max(round((x1 - x0) * scale), 2)),
                           np.linspace(y0, y1, max(round((y1 - y0) * scale), 2)))
        reach = 4 * np.hypot(x1 - x0, y1 - y0)  # long enough for an asymptote to leave the view
        # Places a label shouldn't go: the sensors, the shot, the estimate, and labels already placed.
        crowded = [*snap.positions, snap.shot] + ([self.fix.position] if self.fix is not None else [])

        for pair in self.pairs:
            if not pair.feasible:
                continue
            s_i, s_j = snap.positions[pair.i], snap.positions[pair.j]
            color = snap.colors[pair.i]
            width = 2.6 if snap.selected in (pair.i, pair.j) else 1.6
            style = "solid" if pair.j == snap.reference else (0, (6, 3))
            F = hyperbola_function(X, Y, s_i, s_j, pair.range_diff)
            curve = ax.contour(X, Y, F, levels=[0.0], colors=[color], linewidths=width, linestyles=[style], zorder=4)
            spot = _label_spot(curve, (x0, x1, y0, y1), crowded)
            if spot is not None:
                crowded.append(spot)
                name = _pair_name(snap, pair)
                ax.clabel(curve, manual=[spot], fmt=lambda _v, name=name: name, inline=True, fontsize=8,
                          colors=[snap.theme.ink_secondary])

            if step == SIGN and pair.range_diff != 0:
                # The mirror branch, |p − sᵢ| − |p − sⱼ| = −c·Δt, is where F = −2c·Δt.
                ax.contour(X, Y, F, levels=[-2 * pair.range_diff], colors=[color], linewidths=1.2,
                           linestyles=[(0, (1, 2.5))], zorder=3)

            if step == FAR or self.show_asymptotes:
                asym = branch_asymptotes(s_i, s_j, pair.range_diff)
                if asym is not None:
                    for direction in asym.directions:
                        end = asym.center + reach * direction
                        ax.plot([asym.center[0], end[0]], [asym.center[1], end[1]], color=color,
                                linewidth=1.0, linestyle=(0, (2, 3)), zorder=3)

    # ---- Method tab --------------------------------------------------------

    def method_html(self, snap: Snapshot) -> str:
        out = [f"<h2>{self.title}</h2>"]
        if snap.shot is None or snap.n < 3:
            out.append('<p class="note">Place at least 3 sensors and fire a shot (shift-click) to step through this mode.</p>')
            return "\n".join(out)
        out.append(roadmap(STEPS, self.step))
        builders = {DIFFERENCES: self._differences, CURVES: self._curves, SIGN: self._sign,
                    CROSSING: self._crossing, FAR: self._far}
        out.append(builders[self.step](snap))
        return "\n".join(out)

    def _rows(self, snap: Snapshot, line) -> list[str]:
        """One block per pair shown, anchored at the first block for each sensor and washed if selected."""
        out, anchored = [], set()
        for pair in self.pairs:
            anchor = ""
            if pair.i not in anchored:
                anchor = f'<a name="sensor-{snap.labels[pair.i]}"></a>'
                anchored.add(pair.i)
            html = f'<p class="eq">{anchor}{swatch(snap.colors[pair.i])}&nbsp;{line(pair)}</p>'
            out.append(wash(html, snap.theme.selected_wash) if snap.selected in (pair.i, pair.j) else html)
        ref = snap.labels[snap.reference]
        out.append(f'<p class="eq"><a name="sensor-{ref}"></a>{swatch(snap.colors[snap.reference])}&nbsp;'
                   f'{ref} is the reference.</p>')
        return out

    def _differences(self, snap: Snapshot) -> str:
        k = snap.reference
        ref, kk = snap.labels[k], index_of(snap.labels[k])
        out = [
            "<h3>Step 1. Time differences: t<sub>0</sub> cancels</h3>",
            f"<p>Write sensor i's range equation and the reference {ref}'s, and subtract:</p>",
            '<p class="eq">|p − s<sub>i</sub>| = c·(t<sub>i</sub> − t<sub>0</sub>)<br>'
            f"|p − s<sub>{kk}</sub>| = c·(t<sub>{kk}</sub> − t<sub>0</sub>)<br>"
            f"⇒ |p − s<sub>i</sub>| − |p − s<sub>{kk}</sub>| = c·(t<sub>i</sub> − t<sub>{kk}</sub>) = c·Δt<sub>i</sub></p>",
            "<p>The unknown t<sub>0</sub> is gone. What's left needs only the <i>difference</i> of two arrival "
            "times, which synchronized sensors measure well:</p>",
        ]

        def line(pair: TdoaPair) -> str:
            li, lj = snap.labels[pair.i], snap.labels[pair.j]
            return (f"{li} − {lj}: Δt = {sub('t', li)} − {sub('t', lj)} = {num(snap.times[pair.i], 6)} − "
                    f"{num(snap.times[pair.j], 6)} = {millis(pair.dt)},&nbsp; c·Δt = {meters(pair.range_diff)}")

        out += self._rows(snap, line)
        first = self.reference_pairs[0]
        li = snap.labels[first.i]
        when, side = ("after", "farther from") if first.dt > 0 else ("before", "closer to")
        out.append(
            f"<p>For example, {li} heard the shot {millis(abs(first.dt))} {when} {ref}, so the shot is "
            f"{meters(abs(first.range_diff))} {side} {li} than {ref} is. Neither sensor needs to know its own "
            "distance to the shot, only how much later one heard it than the other.</p>"
        )
        if self.all_pairs and self.extra_pairs:
            out.append('<p class="note">The pairs without the reference (dashed on the field) are differences of the '
                       "others: Δt for S3 − S2 is (S3 − S1) − (S2 − S1). They add no new information.</p>")
        return "\n".join(out)

    def _curves(self, snap: Snapshot) -> str:
        out = [
            "<h3>Step 2. Each pair is a hyperbola</h3>",
            "<p>All the points whose distances to two sensors differ by a fixed amount form one branch of a "
            "hyperbola, with the two sensors as its foci. Each curve is drawn where the function</p>",
            '<p class="eq">f(p) = |p − s<sub>i</sub>| − |p − s<sub>j</sub>| − c·Δt</p>',
            "<p>is zero: f is computed on a grid over the visible field and traced where it changes sign. "
            "At the true shot p:</p>",
        ]
        d = snap.distances

        def line(pair: TdoaPair) -> str:
            li, lj = snap.labels[pair.i], snap.labels[pair.j]
            value = d[pair.i] - d[pair.j] - pair.range_diff
            return (f"{li} − {lj}: f(p) = {num(d[pair.i])} − {num(d[pair.j])} − {term(pair.range_diff)} = "
                    f"{meters(value)}")

        out += self._rows(snap, line)
        out.append(
            '<p class="note">A small time difference gives a branch close to the straight line halfway between the '
            "two sensors (where Δt = 0 exactly); a large one wraps tightly around the sensor that heard the shot "
            "first.</p>"
        )
        if snap.scenario.sigma > 0:
            out.append('<p class="note">With timing noise the values aren\'t exactly zero: each curve misses the '
                       "true shot by a little, so they don't all cross at exactly one point.</p>")
        else:
            out.append('<p class="note">All zero: the shot is on every curve. On the field, each curve is labeled '
                       "with its pair.</p>")
        return "\n".join(out)

    def _sign(self, snap: Snapshot) -> str:
        out = [
            "<h3>Step 3. Why f keeps its sign</h3>",
            "<p>A hyperbola has two branches, mirror images across the line halfway between its foci. Squaring "
            "to get rid of the square roots, as algebra often does, would accept both: |p − s<sub>i</sub>| − "
            "|p − s<sub>j</sub>| = <i>−</i>c·Δt as well as +c·Δt. The wrong one is dotted on the field. It lies "
            "on the side of the sensor that heard the shot <i>later</i>, which can't be right.</p>",
            "<p>Keeping the sign fixes that: f is negative at s<sub>i</sub> and positive at s<sub>j</sub>, and is "
            "zero only on the branch on the correct side.</p>",
            "<p>There is also a limit. A distance difference can't exceed the distance between the two sensors "
            "(the triangle inequality), so |c·Δt| ≤ |s<sub>i</sub> − s<sub>j</sub>|:</p>",
        ]

        def line(pair: TdoaPair) -> str:
            li, lj = snap.labels[pair.i], snap.labels[pair.j]
            verdict = "a curve exists" if pair.feasible else "<b>impossible: no curve</b>"
            return f"{li} − {lj}: |c·Δt| = {meters(abs(pair.range_diff))} vs |s<sub>{index_of(li)}</sub> − " \
                   f"s<sub>{index_of(lj)}</sub>| = {meters(pair.baseline)}: {verdict}"

        out += self._rows(snap, line)
        if not all(p.feasible for p in self.pairs):
            out.append('<p class="note">Timing noise made a time difference larger than sound could take to cross '
                       "between the two sensors, so that pair has no curve at all.</p>")
        return "\n".join(out)

    def _crossing(self, snap: Snapshot) -> str:
        out = ["<h3>Step 4. Where the curves cross</h3>"]
        n, shot = snap.n, snap.shot
        if n == 3:
            out.append("<p>Three sensors give two independent curves for the two unknowns x and y, and the shot is "
                       "where they cross.</p>")
            if len(self.candidates) >= 2:
                where = " and ".join(point(c.position) for c in self.candidates)
                out.append(
                    f"<p><b>Here they cross twice</b>, at {where}. Both points produce exactly the same time "
                    "differences, so nothing in the data can choose between them: only a 4th sensor can. (These are "
                    "the two dips of the t<sub>0</sub> sweep and the two roots of the closed form.)</p>"
                )
        else:
            out.append(
                f"<p>{n} sensors give {n - 1} curves for 2 unknowns, more than needed. Without noise they all "
                "cross at one point; with noise they don't quite, and least squares picks the best compromise.</p>"
            )
        if self.fix is not None:
            out.append(f'<p class="eq"><b>p<sub>est</sub> = {point(self.fix.position)}</b></p>')

        angles = []
        pairs = self.reference_pairs
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                angles.append((pairs[a], pairs[b], crossing_angle(shot, snap.positions, pairs[a], pairs[b])))
        if angles:
            out.append("<p>The angle between the curves where they cross, at the shot:</p>")
            for pa, pb, angle in angles:
                out.append(f'<p class="eq">{_pair_name(snap, pa)} and {_pair_name(snap, pb)}: {num(angle, 1)}°</p>')
            out.append(
                '<p class="note">Near 90°, a small timing error barely moves the crossing. At a shallow angle the '
                "curves run almost together, and the same error slides the crossing a long way along them.</p>"
            )
        out.append('<p class="note">Turn on <i>All pairs</i>: without noise, the extra curves pass through the same '
                   "crossing, because each one's Δt is a difference of two reference pairs' Δt.</p>")
        return "\n".join(out)

    def _far(self, snap: Snapshot) -> str:
        out = [
            "<h3>Step 5. Far field: good bearing, poor range</h3>",
            "<p>Far from its two sensors, each branch straightens out along two straight lines from the pair's "
            "midpoint, its asymptotes (dotted on the field). They make an angle θ with the line from s<sub>i</sub> "
            "to s<sub>j</sub>, where</p>",
            '<p class="eq">cos θ = c·Δt / |s<sub>i</sub> − s<sub>j</sub>|</p>',
            "<p>So for a distant shot, one time difference across a known baseline gives the <i>direction</i> to "
            "the shot. Seen from each pair's midpoint:</p>",
        ]
        shot = snap.shot

        def line(pair: TdoaPair) -> str:
            li, lj = snap.labels[pair.i], snap.labels[pair.j]
            s_i, s_j = snap.positions[pair.i], snap.positions[pair.j]
            asym = branch_asymptotes(s_i, s_j, pair.range_diff)
            if asym is None:
                return f"{li} − {lj}: no curve"
            to_shot = shot - asym.center
            axis = s_j - s_i
            actual = np.degrees(np.arccos(np.clip(to_shot @ axis / (np.linalg.norm(to_shot) * np.linalg.norm(axis)), -1, 1)))
            return (f"{li} − {lj}: cos θ = {num(pair.range_diff)} / {num(pair.baseline)} = "
                    f"{num(pair.range_diff / pair.baseline, 4)}, θ = {num(np.degrees(asym.angle), 1)}°; "
                    f"the shot is really at {num(actual, 1)}°")

        out += self._rows(snap, line)
        out.append(
            "<p>When the shot is far away compared with the sensor spacing, the two angles agree and every curve "
            "points at the shot: the bearing is good. But nearly straight curves from nearby midpoints cross at "
            "shallow angles, so a small timing error slides the crossing a long way toward or away from the array: "
            "the range is poor. Try the far-field preset, with some timing noise.</p>"
        )
        return "\n".join(out)


def _label_spot(curve, view, crowded) -> np.ndarray | None:
    """Where to label a contour: the point on it, well inside the view, farthest from everything crowded.
    (matplotlib's own placement stacks the labels where curves converge.)"""
    x0, x1, y0, y1 = view
    points = np.vstack([path.vertices for path in curve.get_paths()]) if curve.get_paths() else np.empty((0, 2))
    margin_x, margin_y = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
    inside = ((points[:, 0] > x0 + margin_x) & (points[:, 0] < x1 - margin_x)
              & (points[:, 1] > y0 + margin_y) & (points[:, 1] < y1 - margin_y))
    points = points[inside]
    if len(points) == 0:
        return None
    nearest = np.min(np.linalg.norm(points[:, None, :] - np.asarray(crowded)[None, :, :], axis=2), axis=1)
    return points[np.argmax(nearest)]


def _pair_name(snap: Snapshot, pair: TdoaPair) -> str:
    return f"{snap.labels[pair.i]} − {snap.labels[pair.j]}"
