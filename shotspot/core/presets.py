"""Ready-made scenarios, each chosen to show one idea."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .scenario import Scenario, Sensor


def _scenario(name: str, description: str, sensor_xy: list[tuple[float, float]], shot: tuple[float, float], seed: int = 1) -> Scenario:
    rng = np.random.default_rng(seed)
    sensors = [Sensor(f"S{i + 1}", float(x), float(y), float(rng.standard_normal())) for i, (x, y) in enumerate(sensor_xy)]
    return Scenario(sensors=sensors, shot=shot, reference="S1", name=name, description=description)


TRIANGLE = [(200.0, 200.0), (800.0, 250.0), (480.0, 800.0)]


def triangle_inside() -> Scenario:
    return _scenario(
        "3 sensors, shot inside",
        "The easy case: the shot is inside the sensor triangle, so the sensors surround it "
        "and the circles/hyperbolas cross at healthy angles.",
        TRIANGLE,
        (480.0, 420.0),
    )


def triangle_outside() -> Scenario:
    return _scenario(
        "3 sensors, shot outside",
        "The shot is outside the triangle. It can still be located uniquely, but the "
        "hyperbolas cross at shallower angles, so timing noise costs more accuracy.",
        TRIANGLE,
        (850.0, 820.0),
    )


def triangle_ambiguous() -> Scenario:
    # Found by scanning the field with the closed-form solver: from here, a second
    # point near (688, 296) produces exactly the same three arrival times.
    return _scenario(
        "3 sensors, two solutions",
        "The shot is in a region behind one sensor where two different points produce "
        "identical arrival times at all three sensors. No algorithm can tell them apart. "
        "Only a 4th sensor can.",
        [(400.0, 380.0), (620.0, 420.0), (480.0, 600.0)],
        (900.0, 140.0),
    )


def square_four() -> Scenario:
    return _scenario(
        "4 sensors in a square",
        "Four sensors: one more equation than unknowns, so the system is overdetermined, "
        "ambiguity disappears, and least squares can average out noise.",
        [(200.0, 200.0), (800.0, 200.0), (800.0, 800.0), (200.0, 800.0)],
        (430.0, 610.0),
    )


def far_field() -> Scenario:
    return _scenario(
        "Far-field shot",
        "A compact array and a distant shot. The wavefront is nearly flat by the time it "
        "arrives: bearing is well determined, range is not, and the error ellipse "
        "stretches out along the line of sight.",
        [(80.0, 80.0), (230.0, 100.0), (110.0, 230.0), (240.0, 220.0)],
        (880.0, 800.0),
    )


def nearly_collinear() -> Scenario:
    return _scenario(
        "Nearly collinear sensors",
        "Three sensors almost on a line. A shot and its mirror image across that line give "
        "almost identical arrival times, so the geometry can barely distinguish them: "
        "the problem is nearly degenerate.",
        [(150.0, 500.0), (500.0, 512.0), (850.0, 496.0)],
        (560.0, 720.0),
    )


PRESETS: dict[str, Callable[[], Scenario]] = {
    "triangle_inside": triangle_inside,
    "triangle_outside": triangle_outside,
    "triangle_ambiguous": triangle_ambiguous,
    "square_four": square_four,
    "far_field": far_field,
    "nearly_collinear": nearly_collinear,
}
