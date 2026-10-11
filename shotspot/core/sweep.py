"""Mode 2: guess t₀, and measure how badly the circles fail to meet.

For a guessed emission time t₀_g, each sensor's circle has radius

    rᵢ = c·(tᵢ − t₀_g)

If the guess is wrong, the circles have no common point. "How badly they
fail" is measured by the best compromise point

    p* = argmin over p of  Σᵢ (|p − sᵢ| − rᵢ)²

and the mismatch left over at that point, Σᵢ (|p* − sᵢ| − rᵢ)² (m²).
Plotting that mismatch against t₀_g shows it dropping to zero at the true t₀,
and nowhere else, except in a 3-sensor ambiguous geometry, where there is
a second zero at the false solution. That is why t₀ can be treated as just
another unknown.

Finding p* is a small 2-unknown least-squares problem. We start from the
linear circle fit (see known_t0.py), refine with Gauss-Newton, and also try a
couple of other starting points so a local minimum doesn't fool us.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import distances, unit_vectors
from .known_t0 import fit_circles_linear


@dataclass
class SweepPoint:
    t0_guess: float
    radii: np.ndarray  # c·(tᵢ − t₀_g); negative means the guess is after sensor i heard it
    position: np.ndarray  # best compromise point p*
    mismatch: np.ndarray  # |p* − sᵢ| − rᵢ for each sensor, meters
    cost: float  # Σ mismatch², m²

    @property
    def rms(self) -> float:
        """Root-mean-square mismatch in meters; easier to read than the cost."""
        return float(np.sqrt(self.cost / len(self.mismatch)))


@dataclass
class SweepCurve:
    t0_values: np.ndarray
    costs: np.ndarray
    positions: np.ndarray  # (len(t0_values), 2)


def _refine(p: np.ndarray, sensors: np.ndarray, radii: np.ndarray, max_iter: int = 60) -> tuple[np.ndarray, float]:
    """Gauss-Newton on Σ(|p − sᵢ| − rᵢ)², halving the step whenever it doesn't help."""
    p = np.asarray(p, float).copy()
    cost = float(np.sum((distances(p, sensors) - radii) ** 2))
    for _ in range(max_iter):
        J = unit_vectors(p, sensors)  # ∂|p − sᵢ|/∂p
        r = distances(p, sensors) - radii
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        for _ in range(30):
            trial = p + step
            trial_cost = float(np.sum((distances(trial, sensors) - radii) ** 2))
            if trial_cost <= cost:
                break
            step = step / 2.0
        else:
            return p, cost  # no step improves things: we're at a minimum
        p, cost = trial, trial_cost
        if np.linalg.norm(step) < 1e-10:
            break
    return p, cost


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
    radii = c * (np.asarray(times, float) - t0_guess)

    starts = [sensors.mean(axis=0)]
    linear = fit_circles_linear(sensors, radii, reference)
    if linear.position is not None:
        starts.append(linear.position)
    if start is not None:
        starts.append(np.asarray(start, float))

    best_p, best_cost = min((_refine(s, sensors, radii) for s in starts), key=lambda pc: pc[1])
    mismatch = distances(best_p, sensors) - radii
    return SweepPoint(float(t0_guess), radii, best_p, mismatch, best_cost)


def sweep(sensors: np.ndarray, times: np.ndarray, c: float, t0_values: np.ndarray, reference: int = 0) -> SweepCurve:
    """Mismatch as a function of guessed t₀, for the residual-vs-t₀ plot."""
    costs, positions = [], []
    previous = None
    for t0 in t0_values:
        point = best_fit_for_t0(sensors, times, c, t0, reference, start=previous)
        costs.append(point.cost)
        positions.append(point.position)
        previous = point.position  # warm start: neighbouring guesses have nearby answers
    return SweepCurve(np.asarray(t0_values, float), np.array(costs), np.array(positions))


def default_t0_range(times: np.ndarray, c: float, max_distance: float) -> tuple[float, float]:
    """A sensible slider range for t₀_g.

    The shot can't have been fired after the first sensor heard it, and (for a
    shot within `max_distance` of every sensor) not more than max_distance/c
    before that.
    """
    first = float(np.min(times))
    return first - max_distance / c, first
