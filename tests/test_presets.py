import numpy as np

from shotspot.core.closed_form import solve_closed_form
from shotspot.core.presets import nearly_collinear, triangle_inside, triangle_outside


def inside_triangle(p, a, b, c):
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    signs = [cross(a, b, p), cross(b, c, p), cross(c, a, p)]
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def test_presets_are_well_formed(preset):
    assert len(preset.sensors) >= 3
    assert preset.reference == "S1"
    assert preset.sigma == 0.0
    x, y = preset.shot
    assert 0 <= x <= preset.width and 0 <= y <= preset.height
    assert preset.description


def test_presets_show_what_they_claim(preset):
    result = solve_closed_form(preset.positions(), preset.measured_times(), preset.c)
    expected_ambiguous = preset.name in ("3 sensors, two solutions", "Nearly collinear sensors")
    assert result.ambiguous == expected_ambiguous


def test_inside_and_outside_presets():
    assert inside_triangle(triangle_inside().shot, *triangle_inside().positions())
    assert not inside_triangle(triangle_outside().shot, *triangle_outside().positions())


def test_collinear_preset_is_nearly_collinear():
    a, b, c = nearly_collinear().positions()
    (ux, uy), (vx, vy) = b - a, c - a
    area = abs(ux * vy - uy * vx) / 2
    longest = max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b))
    assert area / longest**2 < 0.02
