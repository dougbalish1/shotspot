"""Mode 1: pretend we know the emission time t₀ ("cheating mode").

If t₀ were known, every sensor would know its own range to the shot:

    rᵢ = c·(tᵢ − t₀)

and the shot would simply be where the circles |p − sᵢ| = rᵢ meet. To find
that point, square each circle equation and expand:

    |p|² − 2 sᵢ·p + |sᵢ|² = rᵢ²

Subtracting the reference sensor k's equation cancels the |p|² term and
leaves an equation that is *linear* in p:

    2(sᵢ − s_k)·p = |sᵢ|² − |s_k|² − (rᵢ² − r_k²)

Geometrically, each such equation is a straight line: the line through the
two points where circle i crosses circle k (any point on both circles
satisfies both equations, so it satisfies their difference).

With 3+ non-collinear sensors that gives at least 2 linear equations
A·p = b in the 2 unknowns (x, y). Exactly 2 lines cross at one point, found
with Cramer's rule. With more lines than unknowns (and any noise) they don't
all cross at one point, and least squares picks the p that minimizes
Σ(aᵢ·p − bᵢ)². Setting that sum's derivative to zero gives the 2×2 "normal
equations" AᵀA·p = Aᵀb, solved the same way. (The solution itself is computed
with a numerically safer least-squares routine; the 2×2 system is kept so it
can be shown.)

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
    norm_sq_diff: float  # |sᵢ|² − |s_k|²
    radius_sq_diff: float  # rᵢ² − r_k²


@dataclass
class CircleFit:
    reference: int
    radii: np.ndarray  # rᵢ, meters (negative means "heard before it was fired")
    norms_sq: np.ndarray  # |sᵢ|² = xᵢ² + yᵢ²
    radii_sq: np.ndarray  # rᵢ²
    equations: list[CircleEquation]
    position: np.ndarray | None  # None when the sensors can't pin down a point
    rank: int  # rank of the 2-column system; 2 means p is fully determined
    mismatch: np.ndarray | None  # |p − sᵢ| − rᵢ at the estimate, meters
    # The 2×2 system M·p = v that Cramer's rule is applied to: A and b themselves
    # with exactly 2 equations, otherwise the normal equations AᵀA·p = Aᵀb.
    system_matrix: np.ndarray | None = None
    system_rhs: np.ndarray | None = None
    determinant: float | None = None  # det M; 0 means the lines are parallel

    @property
    def normal_equations(self) -> bool:
        return len(self.equations) > 2


def fit_circles_linear(sensors: np.ndarray, radii: np.ndarray, reference: int = 0) -> CircleFit:
    """Least-squares meeting point of circles with known radii (see module docstring)."""
    sensors = np.asarray(sensors, float)
    radii = np.asarray(radii, float)
    norms_sq = np.sum(sensors**2, axis=1)
    radii_sq = radii**2
    k = reference

    equations = []
    for i in range(len(sensors)):
        if i == k:
            continue
        a = 2.0 * (sensors[i] - sensors[k])
        norm_sq_diff = norms_sq[i] - norms_sq[k]
        radius_sq_diff = radii_sq[i] - radii_sq[k]
        equations.append(
            CircleEquation(i, float(a[0]), float(a[1]), float(norm_sq_diff - radius_sq_diff), float(norm_sq_diff), float(radius_sq_diff))
        )

    fit = CircleFit(reference, radii, norms_sq, radii_sq, equations, None, len(equations), None)
    if len(equations) < 2:
        return fit

    A = np.array([[e.a_x, e.a_y] for e in equations])
    b = np.array([e.rhs for e in equations])
    if len(equations) == 2:
        fit.system_matrix, fit.system_rhs = A, b
    else:
        fit.system_matrix, fit.system_rhs = A.T @ A, A.T @ b
    fit.determinant = float(np.linalg.det(fit.system_matrix))

    solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    fit.rank = int(rank)
    if rank < 2:  # collinear sensors: every point on a line fits equally well
        return fit

    fit.position = solution
    fit.mismatch = distances(solution, sensors) - radii
    return fit


def solve_known_t0(sensors: np.ndarray, times: np.ndarray, c: float, t0: float, reference: int = 0) -> CircleFit:
    """Locate the shot when the emission time t₀ is (unrealistically) known."""
    radii = c * (np.asarray(times, float) - t0)
    return fit_circles_linear(sensors, radii, reference)


def linear_positions(sensors: np.ndarray, radii: np.ndarray, reference: int = 0) -> np.ndarray | None:
    """`fit_circles_linear`'s answer for many sets of radii at once.

    radii has shape (T, N), one row per set; returns positions of shape (T, 2),
    or None if the sensors are collinear. The matrix A depends only on the
    sensors, so all T right-hand sides are solved in one least-squares call.
    """
    sensors = np.asarray(sensors, float)
    radii = np.atleast_2d(np.asarray(radii, float))
    others = [i for i in range(len(sensors)) if i != reference]
    if len(others) < 2:
        return None
    norms_sq = np.sum(sensors**2, axis=1)
    A = 2.0 * (sensors[others] - sensors[reference])
    B = (norms_sq[others] - norms_sq[reference])[:, None] - (radii[:, others] ** 2 - radii[:, [reference]] ** 2).T
    solution, _, rank, _ = np.linalg.lstsq(A, B, rcond=None)
    return solution.T if rank == 2 else None
