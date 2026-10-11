"""Mode 4: the closed-form solution (linearize, then a quadratic in t₀).

Every sensor gives one range equation, |p − sᵢ| = c·(tᵢ − t₀), in three
unknowns: x, y and t₀. Square it and expand both sides:

    |p|² − 2 sᵢ·p + |sᵢ|² = c²tᵢ² − 2c²tᵢ·t₀ + c²t₀²                    (Eᵢ)

The awkward terms |p|² and c²t₀² are the same for every sensor. Subtracting
the reference sensor's equation E_k cancels them and leaves an equation that
is *linear* in (x, y, t₀):

    2(sᵢ − s_k)·p − 2c²(tᵢ − t_k)·t₀ = |sᵢ|² − |s_k|² − c²(tᵢ² − t_k²)   (Lᵢ)

Exactly 3 sensors gives only 2 linear equations for 3 unknowns. Solve them
for p as a function of t₀,

    p = u + v·t₀

and substitute that into E_k, which hasn't been used yet. With w = u − s_k:

    (|v|² − c²)·t₀² + 2(w·v + c²t_k)·t₀ + (|w|² − c²t_k²) = 0

a quadratic with up to two roots. A root with t₀ > min(tᵢ) would mean some
sensor heard the shot before it was fired (a negative range), so it is
rejected. Squaring threw away the sign of each range, and this check puts it
back. If both roots survive, the ambiguity is real: both points produce
exactly the same three arrival times, and only a 4th sensor can tell them
apart.

4+ sensors gives 3+ linear equations: solve for (x, y, t₀) directly by least
squares. That fails when the t₀ column of the system is a combination of the
x and y columns, which happens even in good layouts (e.g. a square array with
the shot on a line of symmetry). Then t₀ drops out of the linear system
entirely, and we fall back to the p = u + v·t₀ → quadratic route.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import distances

TIME_TOLERANCE = 1e-9  # s; slack allowed when checking t₀ ≤ min(tᵢ)
RANK_TOLERANCE = 1e-9  # singular values below this fraction of the largest count as zero


@dataclass
class LinearEquation:
    """One differenced equation Lᵢ:  a_x·x + a_y·y + a_t·t₀ = rhs."""

    sensor: int
    a_x: float  # 2(xᵢ − x_k)
    a_y: float  # 2(yᵢ − y_k)
    a_t: float  # −2c²(tᵢ − t_k)
    rhs: float  # |sᵢ|² − |s_k|² − c²(tᵢ² − t_k²)
    # The ingredients of rhs, kept so the UI can show the substitution:
    s_sq: float  # |sᵢ|²
    ref_s_sq: float  # |s_k|²
    ct_sq: float  # c²tᵢ²
    ref_ct_sq: float  # c²t_k²


@dataclass
class Candidate:
    t0: float
    position: np.ndarray
    ranges: np.ndarray  # c·(tᵢ − t₀) for each sensor; all must be ≥ 0
    valid: bool
    worst_sensor: int  # sensor with the smallest range (the one that rules out an invalid root)


@dataclass
class Quadratic:
    """p = u + v·t₀ substituted into E_k, with w = u − s_k:

    qa·t₀² + qb·t₀ + qc = 0,   qa = |v|² − c²,  qb = 2(w·v + c²t_k),  qc = |w|² − c²t_k²
    """

    u: np.ndarray  # m
    v: np.ndarray  # m/s: how fast the candidate point slides as t₀ changes
    w: np.ndarray  # u − s_k
    qa: float
    qb: float
    qc: float
    discriminant: float  # qb² − 4·qa·qc


@dataclass
class ClosedFormResult:
    method: str  # "quadratic", "least_squares", "constrained", "degenerate" or "too_few_sensors"
    reference: int
    equations: list[LinearEquation]
    message: str
    quadratic: Quadratic | None = None
    lstsq_solution: np.ndarray | None = None  # (x, y, t₀) from direct least squares
    singular_values: np.ndarray | None = None  # of the column-normalized linear system
    rank: int | None = None
    candidates: list[Candidate] = field(default_factory=list)
    position: np.ndarray | None = None
    t0: float | None = None
    ambiguous: bool = False

    @property
    def valid_candidates(self) -> list[Candidate]:
        return [cand for cand in self.candidates if cand.valid]


def linear_equations(sensors: np.ndarray, times: np.ndarray, c: float, reference: int = 0) -> list[LinearEquation]:
    """Build Lᵢ = Eᵢ − E_k for every non-reference sensor."""
    s_k, t_k = sensors[reference], times[reference]
    equations = []
    for i, (s_i, t_i) in enumerate(zip(sensors, times)):
        if i == reference:
            continue
        s_sq, ref_s_sq = float(s_i @ s_i), float(s_k @ s_k)
        ct_sq, ref_ct_sq = float((c * t_i) ** 2), float((c * t_k) ** 2)
        equations.append(
            LinearEquation(
                sensor=i,
                a_x=float(2.0 * (s_i[0] - s_k[0])),
                a_y=float(2.0 * (s_i[1] - s_k[1])),
                a_t=float(-2.0 * c**2 * (t_i - t_k)),
                rhs=s_sq - ref_s_sq - (ct_sq - ref_ct_sq),
                s_sq=s_sq,
                ref_s_sq=ref_s_sq,
                ct_sq=ct_sq,
                ref_ct_sq=ref_ct_sq,
            )
        )
    return equations


def _column_scaled_svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Singular values and rank of A after scaling each column to unit length.

    The t₀ column is about c² ≈ 10⁵ times larger than the x and y columns
    (seconds vs meters), so rank must be judged on a rescaled matrix.
    """
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0.0] = 1.0
    singular_values = np.linalg.svd(A / scale, compute_uv=False)
    rank = int(np.sum(singular_values > RANK_TOLERANCE * singular_values[0]))
    return scale, singular_values, rank


def _quadratic_roots(qa: float, qb: float, qc: float, c: float) -> list[float]:
    """Real roots of qa·τ² + qb·τ + qc, computed without catastrophic cancellation.

    If noise pushes the discriminant below zero there is no exact solution;
    the vertex −qb/(2qa) is then the τ that comes closest. A discriminant
    within roundoff of zero is a double root (the shot sits right on the
    edge of an ambiguity region) and is returned once.
    """
    if abs(qa) < 1e-12 * c**2:  # |v| = c exactly: the quadratic degenerates to a line
        return [] if qb == 0.0 else [-qc / qb]
    disc = qb**2 - 4.0 * qa * qc
    if disc <= 1e-12 * (qb**2 + abs(4.0 * qa * qc)):
        return [-qb / (2.0 * qa)]
    q = -0.5 * (qb + np.copysign(np.sqrt(disc), qb))
    return sorted([q / qa, qc / q])


def _candidate(t0: float, position: np.ndarray, times: np.ndarray, c: float) -> Candidate:
    ranges = c * (times - t0)
    worst = int(np.argmin(ranges))
    return Candidate(float(t0), position, ranges, bool(t0 <= times[worst] + TIME_TOLERANCE), worst)


def _quadratic_route(
    sensors: np.ndarray, times: np.ndarray, c: float, k: int, equations: list[LinearEquation]
) -> tuple[Quadratic, list[Candidate]] | None:
    """Solve the linear equations for p = u + v·t₀, then the quadratic for t₀.

    Returns None if the x, y columns are rank-deficient (collinear sensors).
    """
    M = np.array([[e.a_x, e.a_y] for e in equations])
    b = np.array([e.rhs for e in equations])
    d = np.array([-e.a_t for e in equations])  # M·p = b + d·t₀
    _, _, rank = _column_scaled_svd(M)
    if rank < 2:
        return None
    u, *_ = np.linalg.lstsq(M, b, rcond=None)  # exact solve when M is 2×2
    v, *_ = np.linalg.lstsq(M, d, rcond=None)
    t_k = times[k]
    w = u - sensors[k]
    qa = float(v @ v - c**2)
    qb = float(2.0 * (w @ v + c**2 * t_k))
    qc = float(w @ w - (c * t_k) ** 2)
    quadratic = Quadratic(u, v, w, qa, qb, qc, qb**2 - 4.0 * qa * qc)

    # The roots are computed from the same quadratic rewritten in τ = t₀ − t_k:
    #   |w_k + v·τ|² = c²τ²,  with w_k = u + v·t_k − s_k  (the candidate point at t₀ = t_k).
    # Same roots, but the absolute form above subtracts numbers like c²t_k² ≈ 10⁶
    # from each other, which costs precision exactly where it matters (double roots).
    w_k = w + v * t_k
    taus = _quadratic_roots(qa, float(2.0 * (w_k @ v)), float(w_k @ w_k), c)
    candidates = [_candidate(t_k + tau, u + v * (t_k + tau), times, c) for tau in taus]
    return quadratic, candidates


def _range_cost(candidate: Candidate, sensors: np.ndarray) -> float:
    """Σᵢ (|p − sᵢ| − c(tᵢ − t₀))²: how well a candidate fits *all* the original range equations."""
    return float(np.sum((distances(candidate.position, sensors) - candidate.ranges) ** 2))


def solve_closed_form(sensors: np.ndarray, times: np.ndarray, c: float, reference: int = 0) -> ClosedFormResult:
    sensors = np.asarray(sensors, float)
    times = np.asarray(times, float)
    n = len(sensors)
    k = reference
    if n < 3:
        return ClosedFormResult("too_few_sensors", k, [], "Need at least 3 sensors for 3 unknowns (x, y, t₀).")

    equations = linear_equations(sensors, times, c, k)

    if n == 3:
        route = _quadratic_route(sensors, times, c, k, equations)
        if route is None:
            return ClosedFormResult("degenerate", k, equations, "The sensors are collinear: the linear equations can't separate x from y.")
        quadratic, candidates = route
        result = ClosedFormResult("quadratic", k, equations, "", quadratic=quadratic, candidates=candidates)
        valid = result.valid_candidates
        if not valid:
            result.message = "Neither root is physical: each would need a sensor to hear the shot before it was fired."
            return result
        # When both roots are physical nothing in the data prefers either one.
        # Report the later t₀ (the nearer shot) as "the" estimate, and flag it.
        chosen = max(valid, key=lambda cand: cand.t0)
        result.position, result.t0 = chosen.position, chosen.t0
        result.ambiguous = len(valid) == 2 and np.linalg.norm(valid[0].position - valid[1].position) > 1e-6
        result.message = "Two physical roots: a genuine ambiguity." if result.ambiguous else "One physical root."
        return result

    # 4+ sensors: try direct least squares on the overdetermined linear system.
    A = np.array([[e.a_x, e.a_y, e.a_t] for e in equations])
    b = np.array([e.rhs for e in equations])
    scale, singular_values, rank = _column_scaled_svd(A)
    if rank == 3:
        scaled_solution, *_ = np.linalg.lstsq(A / scale, b, rcond=None)
        solution = scaled_solution / scale
        cand = _candidate(solution[2], solution[:2], times, c)
        message = "Least-squares solution of the linear system."
        if not cand.valid:
            message += " Its t₀ is later than one arrival time (noise), so one range comes out negative."
        return ClosedFormResult(
            "least_squares", k, equations, message,
            lstsq_solution=solution, singular_values=singular_values, rank=rank,
            candidates=[cand], position=cand.position, t0=cand.t0,
        )

    # Rank-deficient: t₀ dropped out of the linear system. Use the quadratic.
    route = _quadratic_route(sensors, times, c, k, equations)
    if route is None:
        return ClosedFormResult(
            "degenerate", k, equations, "The sensors are collinear: the linear equations can't separate x from y.",
            singular_values=singular_values, rank=rank,
        )
    quadratic, candidates = route
    result = ClosedFormResult(
        "constrained", k, equations,
        "The linear system can't determine t₀ here, so p = u + v·t₀ was substituted back into the reference equation.",
        quadratic=quadratic, singular_values=singular_values, rank=rank, candidates=candidates,
    )
    valid = result.valid_candidates
    if valid:
        chosen = min(valid, key=lambda cand: _range_cost(cand, sensors))
        result.position, result.t0 = chosen.position, chosen.t0
    return result
