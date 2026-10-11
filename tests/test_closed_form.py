import numpy as np
import pytest

from shotspot.core.closed_form import solve_closed_form
from shotspot.core.geometry import distances
from shotspot.core.presets import square_four, triangle_ambiguous, triangle_inside

from .helpers import C, T0, arrival_times


def satisfies_range_equations(candidate, sensors, times, c, atol=1e-6):
    """|p − sᵢ| = c·(tᵢ − t₀) for every sensor: the original, unsquared equations."""
    return np.allclose(distances(candidate.position, sensors), c * (times - candidate.t0), atol=atol)


def test_recovers_every_preset(preset):
    sensors, times, shot = preset.positions(), preset.measured_times(), preset.shot_position()
    result = solve_closed_form(sensors, times, preset.c, preset.reference_index())
    # With 3 sensors and two valid roots the solver can't know which is real,
    # so the guarantee is that the truth is *among* the valid candidates.
    assert any(np.allclose(cand.position, shot, atol=1e-6) for cand in result.valid_candidates)
    assert any(cand.t0 == pytest.approx(preset.t0, abs=1e-9) for cand in result.valid_candidates)
    if not result.ambiguous:
        np.testing.assert_allclose(result.position, shot, atol=1e-6)


def test_linear_equations_hold_at_the_truth(preset):
    x, y = preset.shot
    result = solve_closed_form(preset.positions(), preset.measured_times(), preset.c)
    for eq in result.equations:
        lhs = eq.a_x * x + eq.a_y * y + eq.a_t * preset.t0
        assert lhs == pytest.approx(eq.rhs, rel=1e-9, abs=1e-6)
        assert eq.rhs == pytest.approx(eq.s_sq - eq.ref_s_sq - (eq.ct_sq - eq.ref_ct_sq))


@pytest.mark.parametrize("reference", [0, 1, 2])
def test_three_sensor_intermediate_steps(reference):
    s = triangle_inside()
    sensors, times = s.positions(), s.measured_times()
    result = solve_closed_form(sensors, times, s.c, reference)
    assert result.method == "quadratic"
    q = result.quadratic
    # p = u + v·t₀ passes through the shot at the true t₀ ...
    np.testing.assert_allclose(q.u + q.v * s.t0, s.shot_position(), atol=1e-6)
    np.testing.assert_allclose(q.w, q.u - sensors[reference])
    # ... and the true t₀ is a root of the displayed (absolute-time) quadratic.
    value = q.qa * s.t0**2 + q.qb * s.t0 + q.qc
    assert abs(value) < 1e-9 * max(abs(q.qa * s.t0**2), abs(q.qb * s.t0), abs(q.qc))


def test_inside_shot_rejects_the_unphysical_root():
    s = triangle_inside()
    times = s.measured_times()
    result = solve_closed_form(s.positions(), times, s.c)
    assert len(result.candidates) == 2
    good, bad = sorted(result.candidates, key=lambda cand: not cand.valid)
    assert good.valid and not bad.valid
    assert bad.t0 > times.min()  # "fired" after a sensor already heard it ...
    assert bad.ranges[bad.worst_sensor] < 0  # ... which shows up as a negative range
    assert not result.ambiguous


def test_two_root_case_is_a_genuine_ambiguity():
    s = triangle_ambiguous()
    sensors, times, shot = s.positions(), s.measured_times(), s.shot_position()
    result = solve_closed_form(sensors, times, s.c)
    assert result.ambiguous
    valid = result.valid_candidates
    assert len(valid) == 2
    for cand in valid:
        assert cand.t0 <= times.min()
        assert np.all(cand.ranges >= 0)
        # Both satisfy every original range equation: the data can't tell them apart.
        assert satisfies_range_equations(cand, sensors, times, s.c)
    matches = [np.allclose(cand.position, shot, atol=1e-6) for cand in valid]
    assert sorted(matches) == [False, True]
    phantom = valid[matches.index(False)]
    assert np.linalg.norm(phantom.position - shot) > 100.0


def test_three_noisy_sensors_still_fit_exactly():
    """3 equations, 3 unknowns: noise moves the answer but leaves zero residual."""
    s = triangle_inside()
    s.sigma = 0.002
    times = s.measured_times()
    result = solve_closed_form(s.positions(), times, s.c)
    assert satisfies_range_equations(result.valid_candidates[0], s.positions(), times, s.c)
    assert np.linalg.norm(result.position - s.shot_position()) > 0.01


def test_four_sensors_use_direct_least_squares():
    s = square_four()
    result = solve_closed_form(s.positions(), s.measured_times(), s.c)
    assert result.method == "least_squares"
    assert result.rank == 3
    np.testing.assert_allclose(result.lstsq_solution, [*s.shot, s.t0], atol=1e-6)


def test_symmetric_square_falls_back_to_the_quadratic():
    """On the square's symmetry line the t₀ column depends on the x, y columns."""
    s = square_four()
    s.shot = (500.0, 350.0)
    result = solve_closed_form(s.positions(), s.measured_times(), s.c)
    assert result.method == "constrained"
    assert result.rank == 2
    np.testing.assert_allclose(result.position, s.shot_position(), atol=1e-6)
    assert result.t0 == pytest.approx(s.t0, abs=1e-9)


@pytest.mark.parametrize("n", [3, 4])
def test_exactly_collinear_sensors_are_degenerate(n):
    sensors = np.column_stack([np.linspace(0.0, 900.0, n), np.full(n, 500.0)])
    result = solve_closed_form(sensors, arrival_times(sensors, (400.0, 700.0)), C)
    assert result.method == "degenerate"
    assert result.position is None


def test_too_few_sensors():
    sensors = np.array([[0.0, 0.0], [100.0, 0.0]])
    result = solve_closed_form(sensors, arrival_times(sensors, (50.0, 50.0)), C)
    assert result.method == "too_few_sensors"
    assert result.position is None


def test_noise_can_make_time_differences_impossible():
    """|c·Δt| larger than the baseline has no exact solution: the solver must not crash."""
    sensors = np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 80.0]])
    times = np.array([T0, T0 + 0.5, T0 + 0.1])
    result = solve_closed_form(sensors, times, C)
    for cand in result.candidates:
        assert np.all(np.isfinite(cand.position))
