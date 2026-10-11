"""Mode 5: iterative least squares (Gauss-Newton).

Unknowns: x = [x, y, t₀]. Each sensor contributes a residual that says how
far a guess is from satisfying that sensor's range equation:

    rᵢ(x) = |p − sᵢ| − c·(tᵢ − t₀)          (meters)

Gauss-Newton replaces each residual with its straight-line approximation
around the current guess, rᵢ(x + Δ) ≈ rᵢ(x) + Jᵢ·Δ, where the Jacobian row is

    Jᵢ = [ ∂rᵢ/∂x, ∂rᵢ/∂y, ∂rᵢ/∂t₀ ] = [ uᵢₓ, uᵢᵧ, c ]

uᵢ is the unit vector from sensor i toward the guess (moving p one meter
along uᵢ adds one meter of distance), and making t₀ later by one second
shrinks the required range by c meters, so it increases rᵢ by c.

Minimizing Σ(rᵢ + Jᵢ·Δ)² gives the normal equations

    (JᵀJ)·Δ = −Jᵀr

Solve for Δ, step x ← x + Δ, and repeat. Near the answer this converges
very fast. From a bad starting guess it can wander off, or settle on the
*other* solution in a 3-sensor ambiguous geometry. This is plain, undamped
Gauss-Newton, so both behaviors are visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ON_SENSOR, distances, unit_vectors

DIVERGED = 1e6  # m; an estimate 1000 km away has clearly run off


@dataclass
class GNIteration:
    k: int
    estimate: np.ndarray  # [x, y, t₀] at this iteration
    distances: np.ndarray  # |p − sᵢ|
    residuals: np.ndarray  # rᵢ = |p − sᵢ| − c(tᵢ − t₀)
    jacobian: np.ndarray  # (N, 3), rows [uᵢₓ, uᵢᵧ, c]
    normal_matrix: np.ndarray  # JᵀJ
    gradient: np.ndarray  # Jᵀr
    step: np.ndarray | None = None  # Δ from (JᵀJ)Δ = −Jᵀr; None on the last record

    @property
    def cost(self) -> float:
        return float(self.residuals @ self.residuals)


@dataclass
class GNResult:
    iterations: list[GNIteration]  # iterations[0] is the starting guess
    status: str  # "converged", "max_iterations", "singular", "hit_sensor", "diverged" or "too_few_sensors"

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    @property
    def position(self) -> np.ndarray:
        return self.iterations[-1].estimate[:2]

    @property
    def t0(self) -> float:
        return float(self.iterations[-1].estimate[2])

    @property
    def path(self) -> np.ndarray:
        """Estimated positions at each iteration, shape (K, 2), for drawing the trail."""
        return np.array([it.estimate[:2] for it in self.iterations])


def initial_t0(sensors: np.ndarray, times: np.ndarray, c: float, p_init: np.ndarray) -> float:
    """A t₀ consistent with the starting position: assume the first sensor to
    hear the shot is exactly |p_init − s_first| away from it.

    The choice barely matters: rᵢ is linear in t₀, so the first Gauss-Newton
    step corrects any error in t₀ exactly, and every later iterate is the
    same whatever t₀ we start from. Only the starting *position* decides
    whether we converge, and to which solution."""
    first = int(np.argmin(times))
    return float(times[first] - np.linalg.norm(np.asarray(p_init, float) - sensors[first]) / c)


def _evaluate(k: int, x: np.ndarray, sensors: np.ndarray, times: np.ndarray, c: float) -> GNIteration:
    p, t0 = x[:2], x[2]
    d = distances(p, sensors)
    r = d - c * (times - t0)
    J = np.column_stack([unit_vectors(p, sensors), np.full(len(sensors), c)])
    return GNIteration(k, x.copy(), d, r, J, J.T @ J, J.T @ r)


def _is_singular(J: np.ndarray, c: float) -> bool:
    # Rescale the t₀ column to meters (c·t₀) so the condition number reflects geometry, not units.
    return np.linalg.cond(J / np.array([1.0, 1.0, c])) > 1e10


def gauss_newton(
    sensors: np.ndarray,
    times: np.ndarray,
    c: float,
    p_init: np.ndarray,
    t0_init: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> GNResult:
    """Run Gauss-Newton from p_init, recording every iteration.

    Stops when a step moves the estimate less than `tol` meters (with t₀
    measured as c·t₀ meters too).
    """
    sensors = np.asarray(sensors, float)
    times = np.asarray(times, float)
    if t0_init is None:
        t0_init = initial_t0(sensors, times, c, p_init)
    x = np.array([p_init[0], p_init[1], t0_init], dtype=float)

    it = _evaluate(0, x, sensors, times, c)
    iterations = [it]
    if len(sensors) < 3:
        return GNResult(iterations, "too_few_sensors")

    status = "max_iterations"
    for k in range(1, max_iter + 1):
        if it.distances.min() < ON_SENSOR:
            status = "hit_sensor"  # the direction from that sensor is undefined
            break
        if _is_singular(it.jacobian, c):
            status = "singular"
            break
        # This Δ solves the normal equations (JᵀJ)Δ = −Jᵀr. Numerically we solve the
        # equivalent least-squares problem J·Δ ≈ −r instead, because forming JᵀJ squares
        # J's condition number, and far from the sensors (nearly parallel uᵢ) that
        # turns "ill-conditioned" into "exactly singular".
        scale = np.array([1.0, 1.0, c])  # solve for c·t₀ (meters) so all columns are O(1)
        scaled_step, *_ = np.linalg.lstsq(it.jacobian / scale, -it.residuals, rcond=None)
        it.step = scaled_step / scale
        x = it.estimate + it.step
        it = _evaluate(k, x, sensors, times, c)
        iterations.append(it)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x[:2]) > DIVERGED:
            status = "diverged"
            break
        previous_step = iterations[-2].step
        if np.linalg.norm(previous_step[:2]) < tol and abs(c * previous_step[2]) < tol:
            status = "converged"
            break
    return GNResult(iterations, status)
