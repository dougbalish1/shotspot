"""Mode 2: guess t₀, and measure how badly the circles fail to meet.

For a guessed emission time t₀_g, each sensor's circle has radius

    rᵢ = c·(tᵢ − t₀_g)

If the guess is wrong, the circles have no common point. "How badly they
fail" is measured by the best compromise point

    p* = argmin over p of  Σᵢ (|p − sᵢ| − rᵢ)²

and the mismatch left over at that point, Σᵢ (|p* − sᵢ| − rᵢ)² (m²), or its
root-mean-square √(Σ/N), which reads as "how far a typical circle misses".
Plotting that mismatch against t₀_g shows it dropping to zero at the true t₀,
and nowhere else, except in a 3-sensor ambiguous geometry, where there is
a second zero at the false solution. That is why t₀ can be treated as just
another unknown.

Finding p* is a small 2-unknown least-squares problem. We start from the
linear circle fit (the subtraction trick of known_t0.py works for any radii),
refine with Gauss-Newton, and also try the array's centroid as a start so a
local minimum doesn't fool us. A sweep solves this for hundreds of guesses,
so the refinement runs on all of them at once (numpy arrays with one row per
guess); neighbouring guesses then lend each other their answers as fresh
starting points. Finally, the bottom of each dip in the sampled curve is
pinned down by golden-section search, so the plot reaches the true minimum
instead of stopping at the nearest sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import ON_SENSOR, distances
from .known_t0 import linear_positions

GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618…: each golden-section step keeps this fraction of the interval
DIP_ITERATIONS = 40  # golden-section steps per dip: shrinks a 30 ms bracket to ~0.1 ns
DIP_DEPTH = 0.5  # a local minimum counts as a dip if it is at most this fraction of the hills around it


@dataclass
class SweepPoint:
    t0_guess: float
    radii: np.ndarray  # c·(tᵢ − t₀_g); negative means the guess is after sensor i heard it
    position: np.ndarray  # best compromise point p*
    mismatch: np.ndarray  # |p* − sᵢ| − rᵢ for each sensor, meters
    cost: float  # Σ mismatch², m²
    linear_position: np.ndarray | None = None  # where the search started: the linear circle fit

    @property
    def rms(self) -> float:
        """Root-mean-square mismatch in meters; easier to read than the cost."""
        return float(np.sqrt(self.cost / len(self.mismatch)))


@dataclass
class SweepCurve:
    t0_values: np.ndarray  # sorted; includes the refined bottom of every dip
    costs: np.ndarray
    positions: np.ndarray  # (len(t0_values), 2): p* for each guess
    dips: list[SweepPoint] = field(default_factory=list)  # real dips (see _dip_indices), lowest cost first
    n_sensors: int = 1

    @property
    def rms(self) -> np.ndarray:
        return np.sqrt(self.costs / self.n_sensors)


# ---- the best compromise point, for many guesses at once ---------------------


# A row stops iterating once its step is this small (meters), or once a step lowers its cost by less
# than this fraction. Far from the answer the leftover mismatch is large and Gauss-Newton creeps in
# slowly; the second test stops it once the cost is settled far beyond what a plot can show.
CONVERGED = 1e-7
SETTLED = 1e-10


def _costs(p: np.ndarray, sensors: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Σᵢ(|p − sᵢ| − rᵢ)² for each row: p is (T, 2), radii (T, N)."""
    diff = p[:, None, :] - sensors[None, :, :]
    d = np.sqrt(np.einsum("tni,tni->tn", diff, diff))
    return np.sum((d - radii) ** 2, axis=1)


def _gauss_newton_steps(p: np.ndarray, sensors: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """One Gauss-Newton step per row: solve the 2×2 system JᵀJ·step = −Jᵀr.

    J's rows are the unit vectors from each sensor toward p. The 2×2 inverse is
    written out (adjugate over determinant); a tiny multiple of trace² added to
    the determinant keeps it finite when the unit vectors are all parallel.
    """
    diff = p[:, None, :] - sensors[None, :, :]  # (T, N, 2)
    d = np.maximum(np.sqrt(np.einsum("tni,tni->tn", diff, diff)), ON_SENSOR)
    J = diff / d[..., None]
    r = d - radii
    a, b, e = (np.einsum("tn,tn->t", J[..., i], J[..., j]) for i, j in ((0, 0), (0, 1), (1, 1)))
    g = np.einsum("tni,tn->ti", J, r)  # Jᵀr
    det = a * e - b * b + 1e-12 * (a + e) ** 2
    return -np.column_stack([e * g[:, 0] - b * g[:, 1], a * g[:, 1] - b * g[:, 0]]) / det[:, None]


def _refine(p: np.ndarray, sensors: np.ndarray, radii: np.ndarray, max_iter: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton on Σᵢ(|p − sᵢ| − rᵢ)² for every row at once, halving a row's step
    whenever it doesn't lower that row's cost. Rows drop out as they converge.
    Returns (positions, costs)."""
    p = np.array(p, float)
    cost = _costs(p, sensors, radii)
    rows = np.arange(len(p))  # the rows still iterating
    for _ in range(max_iter):
        step = _gauss_newton_steps(p[rows], sensors, radii[rows])
        trial = _costs(p[rows] + step, sensors, radii[rows])
        worse = trial > cost[rows]
        for _ in range(30):
            if not worse.any():
                break
            step[worse] /= 2.0
            trial[worse] = _costs(p[rows[worse]] + step[worse], sensors, radii[rows[worse]])
            worse = trial > cost[rows]
        better = ~worse
        moving = better & (np.linalg.norm(step, axis=1) > CONVERGED) & (cost[rows] - trial > SETTLED * cost[rows])
        p[rows[better]] += step[better]
        cost[rows[better]] = trial[better]
        rows = rows[moving]
        if len(rows) == 0:
            break
    return p, cost


def _best_points(sensors, radii, reference, extra_starts=()):
    """p* and its cost for each row of radii, trying several starting points.
    Returns (positions, costs, linear_starts or None)."""
    T = len(radii)
    linear = linear_positions(sensors, radii, reference)
    starts = [np.tile(sensors.mean(axis=0), (T, 1)), *extra_starts]
    if linear is not None:
        starts.append(linear)
    best_p, best_cost = None, None
    for start in starts:
        p, cost = _refine(start, sensors, radii)
        if best_p is None:
            best_p, best_cost = p, cost
        else:
            better = cost < best_cost
            best_p[better], best_cost[better] = p[better], cost[better]
    return best_p, best_cost, linear


def _points(sensors, times, c, t0_values, reference, extra_starts=()) -> list[SweepPoint]:
    radii = c * (np.asarray(times, float)[None, :] - np.asarray(t0_values, float)[:, None])
    positions, costs, linear = _best_points(sensors, radii, reference, extra_starts)
    return [
        SweepPoint(float(t0), radii[j], positions[j], distances(positions[j], sensors) - radii[j], float(costs[j]),
                   None if linear is None else linear[j])
        for j, t0 in enumerate(t0_values)
    ]


def best_fit_for_t0(
    sensors: np.ndarray,
    times: np.ndarray,
    c: float,
    t0_guess: float,
    reference: int = 0,
    start: np.ndarray | None = None,
) -> SweepPoint:
    """The best compromise point and leftover mismatch for one guessed t₀."""
    sensors = np.asarray(sensors, float)
    extra = () if start is None else (np.asarray(start, float).reshape(1, 2),)
    return _points(sensors, times, c, [t0_guess], reference, extra)[0]


# ---- the sweep ---------------------------------------------------------------


def sweep(sensors: np.ndarray, times: np.ndarray, c: float, t0_values: np.ndarray, reference: int = 0) -> SweepCurve:
    """Mismatch as a function of guessed t₀, for the residual-vs-t₀ plot."""
    sensors = np.asarray(sensors, float)
    t0_values = np.asarray(t0_values, float)
    radii = c * (np.asarray(times, float)[None, :] - t0_values[:, None])
    positions, costs, _ = _best_points(sensors, radii, reference)

    # Neighbouring guesses have nearby answers: offer each guess its neighbours' p* as starts.
    for _ in range(3):
        neighbours = (np.roll(positions, 1, axis=0), np.roll(positions, -1, axis=0))
        improved = False
        for start in neighbours:
            p, cost = _refine(start, sensors, radii)
            better = cost < costs * (1 - 1e-9) - 1e-12
            positions[better], costs[better] = p[better], cost[better]
            improved |= bool(better.any())
        if not improved:
            break

    dips = _refine_dips(sensors, times, c, t0_values, costs, positions, reference)
    if dips:
        t0_values = np.append(t0_values, [d.t0_guess for d in dips])
        costs = np.append(costs, [d.cost for d in dips])
        positions = np.vstack([positions, [d.position for d in dips]])
        order = np.argsort(t0_values, kind="stable")
        t0_values, costs, positions = t0_values[order], costs[order], positions[order]
    return SweepCurve(t0_values, costs, positions, sorted(dips, key=lambda d: d.cost), len(sensors))


def _dip_indices(costs: np.ndarray) -> np.ndarray:
    """Interior local minima that are real dips, not ripples on a plateau.

    The lowest one always counts, however shallow. Any other minimum's "hills"
    are the highest points on each side before the curve drops below it again
    (or ends), and it counts if its RMS miss is at most DIP_DEPTH times the
    lower of the two hills.
    """
    rms = np.sqrt(costs)
    candidates = np.flatnonzero((rms[1:-1] < rms[:-2]) & (rms[1:-1] <= rms[2:])) + 1
    keep = []
    for j in candidates:
        if rms[j] == rms[candidates].min():
            keep.append(j)
            continue
        lower_left = np.flatnonzero(rms[:j] < rms[j])
        lower_right = np.flatnonzero(rms[j + 1:] < rms[j])
        left = rms[lower_left[-1] + 1 if len(lower_left) else 0: j].max()
        right = rms[j + 1: j + 1 + lower_right[0] if len(lower_right) else len(rms)].max()
        if rms[j] <= DIP_DEPTH * min(left, right):
            keep.append(j)
    return np.array(keep, int)


def _refine_dips(sensors, times, c, t0_values, costs, positions, reference) -> list[SweepPoint]:
    """Golden-section search for the bottom of every dip in the sampled curve.

    Each dip j is bracketed by its neighbours [t_{j−1}, t_{j+1}]. Every step
    evaluates two interior points per dip (all dips in one batched call) and
    discards the third of the bracket beyond the worse one. Inside a bracket the
    answer barely moves, so each evaluation starts from the dip's latest p*.
    """
    j = _dip_indices(costs)
    if len(j) == 0:
        return []
    times = np.asarray(times, float)
    a, b = t0_values[j - 1].copy(), t0_values[j + 1].copy()
    starts = positions[j].copy()
    for _ in range(DIP_ITERATIONS):
        lo, hi = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
        t0s = np.concatenate([lo, hi])
        p, f = _refine(np.vstack([starts, starts]), sensors, c * (times[None, :] - t0s[:, None]))
        f_lo, f_hi = f[: len(j)], f[len(j):]
        keep_left = f_lo <= f_hi
        b = np.where(keep_left, hi, b)
        a = np.where(keep_left, a, lo)
        starts = np.where(keep_left[:, None], p[: len(j)], p[len(j):])
    return _points(sensors, times, c, (a + b) / 2, reference, (starts,))


def default_t0_range(times: np.ndarray, c: float, max_distance: float) -> tuple[float, float]:
    """A sensible slider range for t₀_g.

    The shot can't have been fired after the first sensor heard it, and (for a
    shot within `max_distance` of every sensor) not more than max_distance/c
    before that.
    """
    first = float(np.min(times))
    return first - max_distance / c, first
