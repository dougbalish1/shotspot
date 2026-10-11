import numpy as np
import pytest

from shotspot.core.scenario import Scenario, Sensor


def make():
    return Scenario(
        sensors=[Sensor("S1", 0.0, 0.0, 1.0), Sensor("S2", 300.0, 0.0, -2.0), Sensor("S3", 0.0, 400.0, 0.5)],
        shot=(300.0, 400.0),
        t0=1.0,
        c=300.0,
    )


def test_arrival_times_follow_geometry():
    s = make()
    # distances: S1 → 500 (3-4-5 triangle), S2 → 400, S3 → 300
    np.testing.assert_allclose(s.distances(), [500.0, 400.0, 300.0])
    np.testing.assert_allclose(s.true_times(), [1.0 + 500 / 300, 1.0 + 400 / 300, 2.0])


def test_noise_is_sigma_times_fixed_draw():
    s = make()
    s.sigma = 0.002
    np.testing.assert_allclose(s.measured_times() - s.true_times(), [0.002, -0.004, 0.001])
    s.sigma = 0.0
    np.testing.assert_array_equal(s.measured_times(), s.true_times())


def test_labels_and_reference_follow_edits():
    s = make()
    rng = np.random.default_rng(0)
    assert s.add_sensor(10, 10, rng).label == "S4"
    s.reference = "S2"
    s.remove_sensor(1)
    assert s.labels() == ["S1", "S3", "S4"]
    assert s.reference == "S1"  # the reference was deleted, so it falls back to the first sensor
    assert s.add_sensor(20, 20, rng).label == "S2"  # the lowest free label is reused
    assert s.add_sensor(30, 30, rng).label == "S5"


def test_reference_index_falls_back_to_first():
    s = make()
    s.reference = "S3"
    assert s.reference_index() == 2
    s.reference = "nope"
    assert s.reference_index() == 0


def test_reroll_changes_draws():
    s = make()
    before = [x.noise_draw for x in s.sensors]
    s.reroll_noise(np.random.default_rng(42))
    assert [x.noise_draw for x in s.sensors] != before


def test_json_round_trip(tmp_path):
    s = make()
    s.sigma = 0.0015
    s.reference = "S2"
    path = tmp_path / "scenario.json"
    s.save(path)
    loaded = Scenario.load(path)
    assert loaded == s


def test_no_shot_raises():
    s = make()
    s.shot = None
    with pytest.raises(ValueError):
        s.true_times()
