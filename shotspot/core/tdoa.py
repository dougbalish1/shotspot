"""Mode 3: time differences of arrival (TDOA) and hyperbolas.

Subtract two sensors' range equations, |p − sᵢ| = c(tᵢ − t₀) and
|p − sⱼ| = c(tⱼ − t₀), and the unknown t₀ cancels:

    |p − sᵢ| − |p − sⱼ| = c·(tᵢ − tⱼ) = c·Δtᵢⱼ

Neither sensor needs to know its own distance to the shot, only how much
later one heard it than the other. The points with a fixed *difference* of
distances to two foci form one branch of a hyperbola. We draw it as the zero
contour of the signed function

    f(p) = |p − sᵢ| − |p − sⱼ| − c·Δtᵢⱼ

Keeping the sign (rather than squaring, or using |·|) selects the single
correct branch: the one on the side of the sensor that heard the shot first.

Far from the two sensors a hyperbola straightens out along its asymptotes.
In the far field every branch is nearly a straight ray from its pair's
midpoint: the rays still pin down the *direction* to the shot well, but they
cross at shallow angles, so the *range* is poorly determined.

The asymptotes make an angle θ with the line from sᵢ to sⱼ where

    cos θ = c·Δtᵢⱼ / |sᵢ − sⱼ|

so for a distant shot one time difference across a known baseline gives the
bearing directly. Where two curves cross, the angle between them is the
angle between their gradients, ∇f = uᵢ − uⱼ (uᵢ the unit vector from sᵢ
toward p): near 90° a small timing error barely moves the crossing, while at
a shallow angle it slides the crossing a long way along the curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class TdoaPair:
    """Sensor i relative to sensor j."""

    i: int
    j: int
    dt: float  # Δt = tᵢ − tⱼ, seconds
    range_diff: float  # c·Δt = |p − sᵢ| − |p − sⱼ|, meters
    baseline: float  # |sᵢ − sⱼ|, meters

    @property
    def feasible(self) -> bool:
        """A distance difference can never exceed the distance between the foci
        (triangle inequality). Timing noise can break that, leaving no curve."""
        return abs(self.range_diff) < self.baseline


@dataclass
class Asymptotes:
    center: np.ndarray  # midpoint of the two sensors
    vertex: np.ndarray  # the branch's closest point to the center
    directions: np.ndarray  # (2, 2): unit vectors of the two asymptote rays for this branch
    angle: float  # θ between each ray and the direction from sᵢ to sⱼ: cos θ = range_diff / baseline (radians)


def tdoa_pairs(sensors: np.ndarray, times: np.ndarray, c: float, reference: int = 0, all_pairs: bool = False) -> list[TdoaPair]:
    """Pairs relative to the reference sensor (N−1 of them), or all N(N−1)/2 pairs."""
    sensors = np.asarray(sensors, float)
    times = np.asarray(times, float)
    if all_pairs:
        index_pairs = [(j, i) for i, j in combinations(range(len(sensors)), 2)]
    else:
        index_pairs = [(i, reference) for i in range(len(sensors)) if i != reference]

    pairs = []
    for i, j in index_pairs:
        dt = float(times[i] - times[j])
        pairs.append(TdoaPair(i, j, dt, c * dt, float(np.linalg.norm(sensors[i] - sensors[j]))))
    return pairs


def hyperbola_function(X: np.ndarray, Y: np.ndarray, s_i: np.ndarray, s_j: np.ndarray, range_diff: float) -> np.ndarray:
    """f(p) = |p − sᵢ| − |p − sⱼ| − c·Δt, evaluated on a grid (or at points).

    Zero exactly on the correct hyperbola branch. Positive on the sⱼ side of
    it (p is farther from sᵢ, relative to sⱼ, than the timing says) and
    negative on the sᵢ side.
    """
    d_i = np.hypot(X - s_i[0], Y - s_i[1])
    d_j = np.hypot(X - s_j[0], Y - s_j[1])
    return d_i - d_j - range_diff


def branch_asymptotes(s_i: np.ndarray, s_j: np.ndarray, range_diff: float) -> Asymptotes | None:
    """Asymptote rays of the branch |p − sᵢ| − |p − sⱼ| = range_diff.

    Textbook hyperbola parameters for foci sᵢ, sⱼ:
        h = |sᵢ − sⱼ| / 2       half the distance between the foci
        a = |range_diff| / 2    center-to-vertex distance
        b = √(h² − a²)
    Along the focal axis e (pointing from sᵢ toward sⱼ), the branch's vertex is
    at center + (range_diff/2)·e: toward sⱼ when sᵢ heard the shot later.
    The asymptotes leave the center at angle ±atan(b/a) from that direction.
    Returns None if no hyperbola exists (|range_diff| ≥ baseline).
    """
    s_i, s_j = np.asarray(s_i, float), np.asarray(s_j, float)
    h = np.linalg.norm(s_j - s_i) / 2.0
    a = abs(range_diff) / 2.0
    if h == 0.0 or a >= h:
        return None
    b = np.sqrt(h**2 - a**2)
    e = (s_j - s_i) / (2.0 * h)
    n = np.array([-e[1], e[0]])
    center = (s_i + s_j) / 2.0
    along = (range_diff / 2.0) * e  # signed: picks the branch's side
    directions = np.array([along + b * n, along - b * n]) / h  # |along ± b n| = √(a² + b²) = h
    return Asymptotes(center, center + along, directions, float(np.arccos(range_diff / (2.0 * h))))


def curve_gradient(p: np.ndarray, s_i: np.ndarray, s_j: np.ndarray) -> np.ndarray:
    """∇f at p for f(p) = |p − sᵢ| − |p − sⱼ| − c·Δt: the difference of the two unit vectors toward p."""
    p, s_i, s_j = (np.asarray(v, float) for v in (p, s_i, s_j))
    return (p - s_i) / np.linalg.norm(p - s_i) - (p - s_j) / np.linalg.norm(p - s_j)


def crossing_angle(p: np.ndarray, sensors: np.ndarray, pair_a: TdoaPair, pair_b: TdoaPair) -> float:
    """The angle (degrees, 0 to 90) at which two pairs' curves cross at p."""
    sensors = np.asarray(sensors, float)
    g_a = curve_gradient(p, sensors[pair_a.i], sensors[pair_a.j])
    g_b = curve_gradient(p, sensors[pair_b.i], sensors[pair_b.j])
    cos = abs(g_a @ g_b) / (np.linalg.norm(g_a) * np.linalg.norm(g_b))
    return float(np.degrees(np.arccos(min(cos, 1.0))))
