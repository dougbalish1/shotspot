import numpy as np
import pytest

from shotspot.core.closed_form import solve_closed_form
from shotspot.core.gauss_newton import gauss_newton, initial_t0
from shotspot.core.presets import square_four, triangle_ambiguous, triangle_inside


def test_converges_from_the_centroid(preset):
    sensors, times = preset.positions(), preset.measured_times()
    result = gauss_newton(sensors, times, preset.c, sensors.mean(axis=0))
    assert result.converged
    # In the two ambiguous presets the centroid may lead to the other valid
    # solution, so compare against every valid closed-form candidate.
    candidates = [cand.position for cand in solve_closed_form(sensors, times, preset.c).valid_candidates]
    assert any(np.allclose(result.position, p, atol=1e-6) for p in candidates)
    assert result.iterations[-1].cost < 1e-12


def test_start_near_truth_finds_truth_even_when_ambiguous(preset):
    sensors, times, shot = preset.positions(), preset.measured_times(), preset.shot_position()
    result = gauss_newton(sensors, times, preset.c, shot + [40.0, -30.0])
    assert result.converged
    np.testing.assert_allclose(result.position, shot, atol=1e-6)
    assert result.t0 == pytest.approx(preset.t0, abs=1e-9)


@pytest.mark.parametrize("start", ["near_phantom", "centroid"])
def test_ambiguous_case_can_converge_to_the_phantom(start):
    """The wrong-minimum demonstration: the data fits the phantom perfectly too.
    Even the natural "start at the middle of the array" lands on it."""
    s = triangle_ambiguous()
    sensors, times = s.positions(), s.measured_times()
    phantom = next(
        cand for cand in solve_closed_form(sensors, times, s.c).valid_candidates
        if not np.allclose(cand.position, s.shot_position(), atol=1e-3)
    )
    p_init = phantom.position + [-20.0, -20.0] if start == "near_phantom" else sensors.mean(axis=0)
    result = gauss_newton(sensors, times, s.c, p_init)
    assert result.converged
    np.testing.assert_allclose(result.position, phantom.position, atol=1e-6)
    assert result.iterations[-1].cost < 1e-12


def test_bad_start_diverges_without_crashing():
    """Far outside the array, undamped Gauss-Newton runs off toward infinity."""
    s = triangle_inside()
    result = gauss_newton(s.positions(), s.measured_times(), s.c, np.array([100.0, 950.0]))
    assert result.status == "diverged"
    assert len(result.iterations) < 10


def test_initial_t0_does_not_change_the_path():
    """rᵢ is linear in t₀, so the first step absorbs any error in the initial t₀ exactly."""
    s = triangle_inside()
    sensors, times = s.positions(), s.measured_times()
    start = np.array([300.0, 700.0])
    a = gauss_newton(sensors, times, s.c, start, t0_init=0.0)
    b = gauss_newton(sensors, times, s.c, start, t0_init=s.t0 - 1.0)
    np.testing.assert_allclose(a.path[1:], b.path[1:len(a.path)], atol=1e-6)


def test_jacobian_matches_finite_differences():
    s = square_four()
    sensors, times = s.positions(), s.measured_times()
    x = np.array([350.0, 520.0, 3.1])
    first = gauss_newton(sensors, times, s.c, x[:2], x[2], max_iter=1).iterations[0]

    def residuals(x):
        return np.linalg.norm(x[:2] - sensors, axis=1) - s.c * (times - x[2])

    for col, h in enumerate([1e-4, 1e-4, 1e-8]):
        dx = np.zeros(3)
        dx[col] = h
        numeric = (residuals(x + dx) - residuals(x - dx)) / (2 * h)
        np.testing.assert_allclose(first.jacobian[:, col], numeric, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(first.jacobian[:, 2], s.c)
    np.testing.assert_allclose(np.linalg.norm(first.jacobian[:, :2], axis=1), 1.0)


def test_iteration_records_are_consistent():
    s = triangle_inside()
    sensors, times = s.positions(), s.measured_times()
    start = np.array([300.0, 700.0])
    result = gauss_newton(sensors, times, s.c, start)
    assert result.converged
    first = result.iterations[0]
    np.testing.assert_allclose(first.estimate, [*start, initial_t0(sensors, times, s.c, start)])
    np.testing.assert_allclose(first.normal_matrix, first.jacobian.T @ first.jacobian)
    np.testing.assert_allclose(first.gradient, first.jacobian.T @ first.residuals)
    for this, following in zip(result.iterations, result.iterations[1:]):
        np.testing.assert_allclose(this.estimate + this.step, following.estimate)
    assert result.iterations[-1].step is None
    assert result.path.shape == (len(result.iterations), 2)


def test_initial_t0_matches_the_first_sensor():
    s = triangle_inside()
    sensors, times = s.positions(), s.measured_times()
    first = int(np.argmin(times))
    t0 = initial_t0(sensors, times, s.c, s.shot_position())
    assert t0 == pytest.approx(s.t0)  # exact when the starting point is the true shot
    assert initial_t0(sensors, times, s.c, sensors[first]) == times[first]


def test_starting_on_a_sensor_stops_cleanly():
    s = triangle_inside()
    sensors = s.positions()
    result = gauss_newton(sensors, s.measured_times(), s.c, sensors[1])
    assert result.status == "hit_sensor"


def test_too_few_sensors():
    sensors = np.array([[0.0, 0.0], [100.0, 0.0]])
    result = gauss_newton(sensors, np.array([3.0, 3.1]), 343.0, np.array([50.0, 50.0]))
    assert result.status == "too_few_sensors"
