"""Mode 1: pretend we know the emission time t₀ ("cheating mode").

If t₀ were known, every sensor would know its own range to the shot:

    rᵢ = c·(tᵢ − t₀)

and the shot would simply be where the circles |p − sᵢ| = rᵢ meet. To find
that point, square each circle equation and expand:

    |p|² − 2 sᵢ·p + |sᵢ|² = rᵢ²

Subtracting the reference sensor k's equation cancels the |p|² term and
leaves an equation that is *linear* in p:

    2(sᵢ − s_k)·p = |sᵢ|² − |s_k|² − (rᵢ² − r_k²)

With 3+ non-collinear sensors that gives at least 2 linear equations in the
2 unknowns (x, y), solved by least squares.

In reality nobody knows t₀ (a gunshot doesn't announce when it was fired),
which is the whole point of the other modes. This fit is also reused by the
t₀-sweep mode, where t₀ is a *guess*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import distances


@dataclass
class CircleEquation:
    """One differenced circle equation:  a_x·x + a_y·y = rhs."""

    sensor: int
    a_x: float  # 2(xᵢ − x_k)
    a_y: float  # 2(yᵢ − y_k)
    rhs: float  # |sᵢ|² − |s_k|² − (rᵢ² − r_k²)


@dataclass
class CircleFit:
    reference: int
    radii: np.ndarray  # rᵢ, meters (negative means "heard before it was fired")
    equations: list[CircleEquation]
    position: np.ndarray | None  # None when the sensors can't pin down a point
    rank: int  # rank of the 2-column system; 2 means p is fully determined
    mismatch: np.ndarray | None  # |p − sᵢ| − rᵢ at the estimate, meters


def fit_circles_linear(sensors: np.ndarray, radii: np.ndarray, reference: int = 0) -> CircleFit:
    """Least-squares meeting point of circles with known radii (see module docstring)."""
    sensors = np.asarray(sensors, float)
    radii = np.asarray(radii, float)
    s_k, r_k = sensors[reference], radii[reference]

    equations = []
    for i, (s_i, r_i) in enumerate(zip(sensors, radii)):
        if i == reference:
            continue
        a = 2.0 * (s_i - s_k)
        rhs = s_i @ s_i - s_k @ s_k - (r_i**2 - r_k**2)
        equations.append(CircleEquation(i, float(a[0]), float(a[1]), float(rhs)))

    if len(equations) < 2:
        return CircleFit(reference, radii, equations, None, len(equations), None)

    A = np.array([[e.a_x, e.a_y] for e in equations])
    b = np.array([e.rhs for e in equations])
    solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < 2:  # collinear sensors: every point on a line fits equally well
        return CircleFit(reference, radii, equations, None, int(rank), None)

    mismatch = distances(solution, sensors) - radii
    return CircleFit(reference, radii, equations, solution, int(rank), mismatch)


def solve_known_t0(sensors: np.ndarray, times: np.ndarray, c: float, t0: float, reference: int = 0) -> CircleFit:
    """Locate the shot when the emission time t₀ is (unrealistically) known."""
    radii = c * (np.asarray(times, float) - t0)
    return fit_circles_linear(sensors, radii, reference)
