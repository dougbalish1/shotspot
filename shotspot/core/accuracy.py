"""Mode 6: how accurate can a fix be? (GDOP: geometric dilution of precision)

Suppose each arrival time carries a small independent error δtᵢ with standard
deviation σ. That error shifts each residual by c·δtᵢ, and one linearized
Gauss-Newton step at the true answer turns it into an error in the unknowns:

    δx = −(JᵀJ)⁻¹·Jᵀ·(c·δt)      ⇒      Cov[x, y, t₀] = (JᵀJ)⁻¹·(cσ)²

J is the same Jacobian Gauss-Newton uses (rows [uᵢₓ, uᵢᵧ, c]), evaluated at
the shot. The top-left 2×2 block is the position covariance; its eigenvectors
and eigenvalues give the error ellipse. A single "how big is the error"
number is DRMS = √(σx² + σy²).

Measuring t₀ in meters (b = c·t₀) changes the rows to [uᵢₓ, uᵢᵧ, 1], which
don't involve c at all. So position error = (geometry factor) × (cσ): the
layout decides the multiplier, and the timing noise sets the scale. The
geometry factor is what `dilution` computes. It is small when the unit
vectors uᵢ point in many different directions (shot inside the array) and
large when they are nearly parallel (shot far outside, where the hyperbolas
cross at shallow angles).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .closed_form import solve_closed_form
from .gauss_newton import gauss_newton
from .geometry import ON_SENSOR

SINGULAR_DET = 1e-10  # relative determinant below which JᵀJ is treated as singular


def jacobian_at(p: np.ndarray, sensors: np.ndarray, c: float) -> np.ndarray:
    """Jacobian rows [uᵢₓ, uᵢᵧ, c] at position p."""
    diff = np.asarray(p, float) - np.asarray(sensors, float)
    d = np.maximum(np.linalg.norm(diff, axis=1), ON_SENSOR)
    return np.column_stack([diff / d[:, None], np.full(len(diff), c)])


def covariance(p: np.ndarray, sensors: np.ndarray, c: float, sigma: float) -> np.ndarray | None:
    """Cov[x, y, t₀] = (JᵀJ)⁻¹·(cσ)², or None if the geometry is singular there."""
    J = jacobian_at(p, sensors, c)
    if len(J) < 3:
        return None
    # Invert through the SVD of the column-scaled J (t₀ measured as c·t₀ meters)
    # rather than inverting JᵀJ directly, which would square the conditioning:
    #   JᵀJ = D·Vᵀ·S²·V·D   ⇒   (JᵀJ)⁻¹ = D⁻¹·Vᵀ·S⁻²·V·D⁻¹,   D = diag(1, 1, c)
    scale = np.array([1.0, 1.0, c])
    _, s, Vt = np.linalg.svd(J / scale, full_matrices=False)
    if s[-1] < 1e-10 * s[0]:
        return None
    inv_scaled = Vt.T @ np.diag(1.0 / s**2) @ Vt
    return inv_scaled / np.outer(scale, scale) * (c * sigma) ** 2


def drms(position_cov: np.ndarray) -> float:
    """√(σx² + σy²) from a 2×2 position covariance."""
    return float(np.sqrt(np.trace(position_cov)))


def dilution(points: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """Geometry factor (meters of position error per meter of range error).

    `points` has shape (..., 2). Returns an array of shape (...); inf where
    the geometry can't determine a position. Vectorized so a whole heatmap
    grid is one call. Multiply by c·σ for the 1σ DRMS error in meters.
    """
    points = np.asarray(points, float)
    sensors = np.asarray(sensors, float)
    flat = points.reshape(-1, 2)
    diff = flat[:, None, :] - sensors[None, :, :]  # (G, N, 2)
    d = np.maximum(np.linalg.norm(diff, axis=2), ON_SENSOR)
    J = np.concatenate([diff / d[..., None], np.ones(d.shape + (1,))], axis=2)  # rows [uₓ, uᵧ, 1]
    JTJ = np.einsum("gni,gnj->gij", J, J)

    n = len(sensors)
    singular = np.abs(np.linalg.det(JTJ)) < SINGULAR_DET * max(n, 1) ** 3
    JTJ[singular] = np.eye(3)  # placeholder so the batched inverse doesn't fail
    inv = np.linalg.inv(JTJ)
    factor = np.sqrt(inv[:, 0, 0] + inv[:, 1, 1])
    factor[singular] = np.inf
    return factor.reshape(points.shape[:-1])


@dataclass
class Fix:
    position: np.ndarray
    t0: float
    source: str  # "gauss_newton" or "closed_form"


def locate(sensors: np.ndarray, times: np.ndarray, c: float, reference: int = 0) -> Fix | None:
    """The 'production' pipeline: closed form to get close, Gauss-Newton to polish."""
    sensors = np.asarray(sensors, float)
    closed = solve_closed_form(sensors, times, c, reference)
    if closed.position is not None:
        start, t0_start = closed.position, closed.t0
    else:
        start, t0_start = sensors.mean(axis=0), None
    gn = gauss_newton(sensors, times, c, start, t0_start)
    if gn.converged:
        return Fix(gn.position, gn.t0, "gauss_newton")
    if closed.position is not None:
        return Fix(closed.position, closed.t0, "closed_form")
    return None


@dataclass
class MonteCarloResult:
    estimates: np.ndarray  # (M, 2) successful fixes
    failures: int  # trials where no fix was found
    mean: np.ndarray
    covariance: np.ndarray  # empirical 2×2 position covariance


def monte_carlo(
    sensors: np.ndarray,
    shot: np.ndarray,
    t0: float,
    c: float,
    sigma: float,
    n_trials: int,
    rng: np.random.Generator,
    reference: int = 0,
) -> MonteCarloResult:
    """Fire n_trials shots with fresh timing noise and locate each one."""
    sensors = np.asarray(sensors, float)
    true_times = t0 + np.linalg.norm(sensors - np.asarray(shot, float), axis=1) / c
    estimates = []
    failures = 0
    for _ in range(n_trials):
        noisy = true_times + sigma * rng.standard_normal(len(sensors))
        fix = locate(sensors, noisy, c, reference)
        if fix is None:
            failures += 1
        else:
            estimates.append(fix.position)
    estimates = np.array(estimates).reshape(-1, 2)
    mean = estimates.mean(axis=0) if len(estimates) else np.full(2, np.nan)
    cov = np.cov(estimates, rowvar=False) if len(estimates) > 1 else np.full((2, 2), np.nan)
    return MonteCarloResult(estimates, failures, mean, cov)
