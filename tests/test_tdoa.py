import numpy as np
import pytest

from shotspot.core.presets import far_field, triangle_inside
from shotspot.core.tdoa import branch_asymptotes, crossing_angle, hyperbola_function, tdoa_pairs


@pytest.mark.parametrize("all_pairs", [False, True])
def test_shot_lies_on_every_hyperbola(preset, all_pairs):
    sensors, times, shot = preset.positions(), preset.measured_times(), preset.shot_position()
    d = preset.distances()
    pairs = tdoa_pairs(sensors, times, preset.c, preset.reference_index(), all_pairs)
    n = len(sensors)
    assert len(pairs) == (n * (n - 1) // 2 if all_pairs else n - 1)
    for pair in pairs:
        assert pair.range_diff == pytest.approx(d[pair.i] - d[pair.j], abs=1e-9)
        assert pair.feasible
        f = hyperbola_function(shot[0], shot[1], sensors[pair.i], sensors[pair.j], pair.range_diff)
        assert abs(f) < 1e-9


def test_signed_function_separates_the_two_sides(preset):
    """f < 0 at sᵢ and f > 0 at sⱼ, so exactly one branch (the zero set) lies between them."""
    sensors = preset.positions()
    for pair in tdoa_pairs(sensors, preset.measured_times(), preset.c, all_pairs=True):
        s_i, s_j = sensors[pair.i], sensors[pair.j]
        assert hyperbola_function(*s_i, s_i, s_j, pair.range_diff) < 0
        assert hyperbola_function(*s_j, s_i, s_j, pair.range_diff) > 0


def test_asymptotes_belong_to_the_branch(preset):
    sensors = preset.positions()
    for pair in tdoa_pairs(sensors, preset.measured_times(), preset.c, all_pairs=True):
        s_i, s_j = sensors[pair.i], sensors[pair.j]
        asym = branch_asymptotes(s_i, s_j, pair.range_diff)
        assert abs(hyperbola_function(*asym.vertex, s_i, s_j, pair.range_diff)) < 1e-9
        np.testing.assert_allclose(np.linalg.norm(asym.directions, axis=1), 1.0)
        axis = (s_j - s_i) / np.linalg.norm(s_j - s_i)
        np.testing.assert_allclose(asym.directions @ axis, np.cos(asym.angle))  # both rays at θ from the axis
        for direction in asym.directions:
            far = asym.center + 1e8 * direction  # the branch hugs its asymptote far away
            assert abs(hyperbola_function(*far, s_i, s_j, pair.range_diff)) < 1e-2


def test_impossible_time_difference_has_no_hyperbola():
    s_i, s_j = np.array([0.0, 0.0]), np.array([100.0, 0.0])
    pairs = tdoa_pairs(np.array([s_i, s_j]), np.array([0.0, 0.5]), 343.0, reference=0)
    assert pairs[0].range_diff == pytest.approx(171.5)  # more than the 100 m baseline
    assert not pairs[0].feasible
    assert branch_asymptotes(s_j, s_i, pairs[0].range_diff) is None


def test_far_field_curves_cross_at_shallow_angles_and_give_the_bearing():
    """Far field: the curves cross at a much shallower angle than inside the array, and each pair's
    far-field bearing, cos θ = c·Δt / baseline, nearly matches the true direction to the shot."""
    angles = {}
    for s in (far_field(), triangle_inside()):
        sensors, shot = s.positions(), s.shot_position()
        pairs = tdoa_pairs(sensors, s.measured_times(), s.c, s.reference_index())
        angles[s.name] = crossing_angle(shot, sensors, pairs[0], pairs[1])
        if s.name == far_field().name:
            for pair in pairs:
                s_i, s_j = sensors[pair.i], sensors[pair.j]
                asym = branch_asymptotes(s_i, s_j, pair.range_diff)
                to_shot = shot - asym.center
                true_angle = np.arccos(to_shot @ (s_j - s_i) / (np.linalg.norm(to_shot) * np.linalg.norm(s_j - s_i)))
                assert np.degrees(abs(asym.angle - true_angle)) < 2.0
    assert angles[far_field().name] < 20 < angles[triangle_inside().name]
