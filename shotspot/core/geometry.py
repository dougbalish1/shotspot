"""Small geometric helpers shared by the solvers."""

from __future__ import annotations

import numpy as np

# Below this distance (meters) a point is treated as sitting on a sensor, where
# the direction from the sensor is undefined.
ON_SENSOR = 1e-9


def distances(p: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """|p − sᵢ| for every sensor. p has shape (2,), sensors (N, 2)."""
    return np.linalg.norm(np.asarray(sensors, float) - np.asarray(p, float), axis=1)


def unit_vectors(p: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """Unit vectors pointing from each sensor toward p, shape (N, 2).

    This is the gradient of |p − sᵢ| with respect to p: moving p along it
    increases the distance to sensor i fastest, one meter per meter.
    """
    diff = np.asarray(p, float) - np.asarray(sensors, float)
    d = np.maximum(np.linalg.norm(diff, axis=1), ON_SENSOR)
    return diff / d[:, None]


def error_ellipse(cov: np.ndarray, n_sigma: float = 1.0) -> tuple[float, float, float]:
    """Axes of the error ellipse for a 2×2 position covariance.

    Returns (semi_major, semi_minor, angle) with the angle in radians,
    measured from +x to the major axis. The axes are the square roots of
    the covariance's eigenvalues, and the directions are its eigenvectors.
    """
    eigvals, eigvecs = np.linalg.eigh(np.asarray(cov, float))  # ascending order
    eigvals = np.maximum(eigvals, 0.0)
    major_dir = eigvecs[:, 1]
    angle = float(np.arctan2(major_dir[1], major_dir[0]))
    return n_sigma * float(np.sqrt(eigvals[1])), n_sigma * float(np.sqrt(eigvals[0])), angle


def ellipse_points(center: np.ndarray, cov: np.ndarray, n_sigma: float = 1.0, n: int = 120) -> np.ndarray:
    """Points around the error ellipse, shape (n, 2), ready for plotting."""
    major, minor, angle = error_ellipse(cov, n_sigma)
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    local = np.column_stack([major * np.cos(theta), minor * np.sin(theta)])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.asarray(center, float) + local @ rot.T


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Corners of the smallest convex polygon around the points, counterclockwise, shape (H, 2).

    Andrew's monotone chain: sort the points by x, then build the lower and
    upper chains, dropping any point that would make a clockwise turn.
    """
    pts = sorted(map(tuple, np.asarray(points, float)))
    if len(pts) < 3:
        return np.array(pts).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])
