import numpy as np
import pytest

from shotspot.core.accuracy import SIGMA_95, covariance, dilution, drms, ellipse_probability, fraction_inside, locate, monte_carlo
from shotspot.core.closed_form import solve_closed_form
from shotspot.core.geometry import convex_hull, error_ellipse
from shotspot.core.presets import far_field, square_four


def test_locate_recovers_every_unambiguous_preset(preset):
    if solve_closed_form(preset.positions(), preset.measured_times(), preset.c).ambiguous:
        pytest.skip("ambiguous: either valid answer is acceptable (see test_closed_form)")
    fix = locate(preset.positions(), preset.measured_times(), preset.c)
    np.testing.assert_allclose(fix.position, preset.shot_position(), atol=1e-6)
    assert fix.t0 == pytest.approx(preset.t0, abs=1e-9)
    assert fix.source == "gauss_newton"


def test_dilution_agrees_with_the_covariance(preset):
    sigma = 0.001
    cov = covariance(preset.shot_position(), preset.positions(), preset.c, sigma)
    factor = dilution(preset.shot_position(), preset.positions())
    assert drms(cov[:2, :2]) == pytest.approx(float(factor) * preset.c * sigma, rel=1e-6)


def test_dilution_grid_shape_and_inside_vs_outside():
    sensors = square_four().positions()
    xs, ys = np.linspace(0, 1000, 21), np.linspace(0, 1000, 11)
    grid = dilution(np.stack(np.meshgrid(xs, ys), axis=-1), sensors)
    assert grid.shape == (11, 21)
    inside = dilution(np.array([500.0, 480.0]), sensors)
    outside = dilution(np.array([3000.0, 2500.0]), sensors)
    assert inside < 1.5 < 10 * inside < outside


def test_collinear_sensors_are_singular_on_their_line():
    sensors = np.array([[0.0, 0.0], [300.0, 0.0], [700.0, 0.0]])
    assert np.isinf(dilution(np.array([1000.0, 0.0]), sensors))
    assert covariance(np.array([1000.0, 0.0]), sensors, 343.0, 0.001) is None
    assert np.isfinite(dilution(np.array([300.0, 400.0]), sensors))


def test_far_field_ellipse_stretches_along_the_line_of_sight():
    s = far_field()
    cov = covariance(s.shot_position(), s.positions(), s.c, 0.001)
    major, minor, angle = error_ellipse(cov[:2, :2])
    assert major > 10 * minor
    line_of_sight = s.shot_position() - s.positions().mean(axis=0)
    major_dir = np.array([np.cos(angle), np.sin(angle)])
    cos_between = abs(major_dir @ line_of_sight) / np.linalg.norm(line_of_sight)
    assert cos_between > 0.99


def test_monte_carlo_matches_the_predicted_covariance():
    s = square_four()
    sigma = 0.0005
    predicted = covariance(s.shot_position(), s.positions(), s.c, sigma)[:2, :2]
    mc = monte_carlo(s.positions(), s.shot_position(), s.t0, s.c, sigma, 1000, np.random.default_rng(7))
    assert mc.failures == 0
    np.testing.assert_allclose(mc.mean, s.shot_position(), atol=0.05)
    # Sample variances of 1000 draws are good to roughly ±10 %.
    np.testing.assert_allclose(np.diag(mc.covariance), np.diag(predicted), rtol=0.15)
    assert drms(mc.covariance) == pytest.approx(drms(predicted), rel=0.1)


def test_ellipse_probabilities():
    assert ellipse_probability(1.0) == pytest.approx(0.3935, abs=1e-4)
    assert ellipse_probability(SIGMA_95) == pytest.approx(0.95)
    cov = np.array([[4.0, 1.0], [1.0, 2.0]])
    pts = np.random.default_rng(3).multivariate_normal([10.0, -5.0], cov, size=20000)
    assert fraction_inside(pts, [10.0, -5.0], cov, 1.0) == pytest.approx(0.3935, abs=0.01)
    assert fraction_inside(pts, [10.0, -5.0], cov, SIGMA_95) == pytest.approx(0.95, abs=0.005)


def test_convex_hull():
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    hull = convex_hull(square + [(0.5, 0.5), (0.2, 0.7)])  # interior points are dropped
    assert sorted(map(tuple, hull)) == sorted(square)
    x, y = hull[:, 0], hull[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) == pytest.approx(1.0)  # counterclockwise, area 1
