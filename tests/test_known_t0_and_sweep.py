import numpy as np
import pytest

from shotspot.core.closed_form import solve_closed_form
from shotspot.core.known_t0 import fit_circles_linear, linear_positions, solve_known_t0
from shotspot.core.presets import far_field, triangle_ambiguous, triangle_inside, triangle_outside
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


def test_known_t0_shown_intermediates_reproduce_the_answer(preset):
    """The panel shows the squared terms and a 2×2 system solved by Cramer's rule; those numbers must agree."""
    sensors = preset.positions()
    fit = solve_known_t0(sensors, preset.measured_times(), preset.c, preset.t0)
    np.testing.assert_allclose(fit.norms_sq, np.sum(sensors**2, axis=1))
    np.testing.assert_allclose(fit.radii_sq, fit.radii**2)
    for eq in fit.equations:
        assert np.isclose(eq.norm_sq_diff - eq.radius_sq_diff, eq.rhs, rtol=1e-12)
    M, v, det = fit.system_matrix, fit.system_rhs, fit.determinant
    assert fit.normal_equations == (len(sensors) > 3)
    assert np.isclose(det, M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0], rtol=1e-9)
    cramer = np.array([v[0] * M[1, 1] - M[0, 1] * v[1], M[0, 0] * v[1] - v[0] * M[1, 0]]) / det
    np.testing.assert_allclose(cramer, fit.position, atol=1e-6)


def test_known_t0_exactly_collinear_has_no_unique_answer():
    sensors = np.array([[0.0, 0.0], [100.0, 0.0], [300.0, 0.0]])
    times = arrival_times(sensors, (150.0, 80.0))
    fit = solve_known_t0(sensors, times, C, T0)
    assert fit.position is None  # the shot and its mirror image fit equally well
    assert fit.rank == 1
    assert abs(fit.determinant) < 1e-6 * np.abs(fit.system_matrix).max() ** 2  # parallel lines


def test_batched_linear_fit_matches_one_at_a_time(preset):
    sensors = preset.positions()
    radii = preset.distances()[None, :] + np.array([[0.0], [25.0], [-40.0]])  # three sets of radii
    batched = linear_positions(sensors, radii)
    for row, r in zip(batched, radii):
        np.testing.assert_allclose(row, fit_circles_linear(sensors, r).position, atol=1e-6)


# ---- mode 2: t₀ sweep --------------------------------------------------------


def test_sweep_dips_are_the_closed_form_roots(preset):
    """At zero noise, the t₀ values where the circles meet are exactly the closed-form solutions:
    one for most layouts, two for the ambiguous and nearly collinear ones."""
    sensors, times = preset.positions(), preset.measured_times()
    lo, hi = default_t0_range(times, preset.c, 2000.0)
    curve = sweep(sensors, times, preset.c, np.linspace(lo, hi, 400))
    meeting = sorted((d for d in curve.dips if d.rms < 1e-4), key=lambda d: d.t0_guess)
    roots = sorted(solve_closed_form(sensors, times, preset.c).valid_candidates, key=lambda cand: cand.t0)
    assert len(meeting) == len(roots) == len(curve.dips)  # and no ripples on the plateau counted as dips
    for dip, root in zip(meeting, roots):
        assert dip.t0_guess == pytest.approx(root.t0, abs=1e-7)
        np.testing.assert_allclose(dip.position, root.position, atol=1e-3)
    assert curve.dips[0].cost == min(curve.costs)  # dips are sorted lowest first, and are in the curve


def test_sweep_agrees_with_one_guess_at_a_time():
    s = triangle_outside()
    sensors, times = s.positions(), s.measured_times()
    guesses = np.array([2.0, 3.0, 3.2, 3.3, 4.0])
    curve = sweep(sensors, times, s.c, guesses)
    for t0, cost in zip(guesses, curve.costs[np.isin(curve.t0_values, guesses)]):
        assert cost == pytest.approx(best_fit_for_t0(sensors, times, s.c, t0).cost, rel=1e-6, abs=1e-9)


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
