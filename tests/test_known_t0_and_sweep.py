import numpy as np

from shotspot.core.closed_form import solve_closed_form
from shotspot.core.known_t0 import solve_known_t0
from shotspot.core.presets import far_field, triangle_ambiguous, triangle_inside
from shotspot.core.sweep import best_fit_for_t0, default_t0_range, sweep

from .helpers import C, T0, arrival_times


# ---- mode 1: known t₀ ------------------------------------------------------


def test_known_t0_recovers_every_preset(preset):
    fit = solve_known_t0(preset.positions(), preset.measured_times(), preset.c, preset.t0)
    np.testing.assert_allclose(fit.position, preset.shot_position(), atol=1e-6)
    np.testing.assert_allclose(fit.mismatch, 0.0, atol=1e-6)
    np.testing.assert_allclose(fit.radii, preset.distances(), atol=1e-9)


def test_known_t0_equations_hold_at_the_truth():
    s = triangle_inside()
    fit = solve_known_t0(s.positions(), s.measured_times(), s.c, s.t0)
    x, y = s.shot
    for eq in fit.equations:
        assert np.isclose(eq.a_x * x + eq.a_y * y, eq.rhs, rtol=1e-12)


def test_known_t0_exactly_collinear_has_no_unique_answer():
    sensors = np.array([[0.0, 0.0], [100.0, 0.0], [300.0, 0.0]])
    times = arrival_times(sensors, (150.0, 80.0))
    fit = solve_known_t0(sensors, times, C, T0)
    assert fit.position is None  # the shot and its mirror image fit equally well
    assert fit.rank == 1


# ---- mode 2: t₀ sweep --------------------------------------------------------


def test_sweep_mismatch_is_zero_only_at_true_t0(preset):
    sensors, times = preset.positions(), preset.measured_times()
    at_truth = best_fit_for_t0(sensors, times, preset.c, preset.t0)
    assert at_truth.cost < 1e-12
    np.testing.assert_allclose(at_truth.position, preset.shot_position(), atol=1e-6)
    for wrong in (preset.t0 - 0.05, preset.t0 + 0.05):
        # Clearly nonzero, though in weak geometries (far field) only by centimeters.
        assert best_fit_for_t0(sensors, times, preset.c, wrong).cost > 1e-4


def test_far_field_sweep_valley_is_shallow():
    """Far field: guessing t₀ 50 ms early adds 17 m to every radius, yet the circles
    still nearly meet (a few cm of mismatch) at a point pushed farther away. That
    is "good bearing, poor range" seen from the t₀ side."""
    far, near = far_field(), triangle_inside()
    for s, rms_bound in ((far, (0.0, 0.2)), (near, (5.0, np.inf))):
        point = best_fit_for_t0(s.positions(), s.measured_times(), s.c, s.t0 - 0.05)
        assert rms_bound[0] < point.rms < rms_bound[1]
    pushed = best_fit_for_t0(far.positions(), far.measured_times(), far.c, far.t0 - 0.05).position
    array_center = far.positions().mean(axis=0)
    assert np.linalg.norm(pushed - array_center) > np.linalg.norm(far.shot_position() - array_center) + 10.0


def test_sweep_curve_has_its_minimum_at_true_t0():
    s = triangle_inside()
    times = s.measured_times()
    lo, hi = default_t0_range(times, s.c, 1500.0)
    assert lo < s.t0 < hi
    t0_values = np.append(np.linspace(lo, hi, 200), s.t0)
    curve = sweep(s.positions(), times, s.c, t0_values)
    assert curve.t0_values[np.argmin(curve.costs)] == s.t0


def test_sweep_has_two_zeros_in_the_ambiguous_case():
    s = triangle_ambiguous()
    sensors, times = s.positions(), s.measured_times()
    roots = sorted(cand.t0 for cand in solve_closed_form(sensors, times, s.c).valid_candidates)
    assert len(roots) == 2
    costs = [best_fit_for_t0(sensors, times, s.c, t0).cost for t0 in roots]
    assert max(costs) < 1e-9
    between = best_fit_for_t0(sensors, times, s.c, (roots[0] + roots[1]) / 2)
    assert between.cost > 0.1  # shallow: this shot sits near the fold where the two solutions merge
