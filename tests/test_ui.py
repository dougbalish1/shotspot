"""Offscreen smoke tests for the window: mouse editing, controls, presets.

Mouse events are synthesized as matplotlib events and fed to the canvas's
own handlers, so hit-testing, drag thresholds and modifiers are exercised
the same way a real click would exercise them.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtWidgets import QApplication

from shotspot.core.presets import PRESETS
from shotspot.core.scenario import Scenario
from shotspot.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app):
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    w.refresh()
    yield w
    w.close()


def mouse(win, kind, xy, button=1, modifiers=(), step=0):
    """Send one mouse event at field coordinates xy (meters), then redraw."""
    canvas = win.field
    x, y = canvas.ax.transData.transform(xy)
    name = {"press": "button_press_event", "move": "motion_notify_event",
            "release": "button_release_event", "scroll": "scroll_event"}[kind]
    event = MouseEvent(name, canvas, x, y, button=button, modifiers=frozenset(modifiers), step=step)
    {"press": canvas._on_press, "move": canvas._on_motion,
     "release": canvas._on_release, "scroll": canvas._on_scroll}[kind](event)
    win.refresh()


def click(win, xy, **kw):
    mouse(win, "press", xy, **kw)
    mouse(win, "release", xy, **kw)


def test_every_preset_renders(win):
    for key in PRESETS:
        win.load_preset(key)
        win.refresh()
        for label in win.scenario.labels():
            assert win.panel.sensors.has_anchor(f"sensor-{label}")
        assert win.mode.fix is not None


def test_click_empty_space_adds_and_selects_a_sensor(win):
    n = len(win.scenario.sensors)
    click(win, (100.0, 900.0))
    assert len(win.scenario.sensors) == n + 1
    np.testing.assert_allclose(win.scenario.sensors[-1].position, [100.0, 900.0], atol=1e-6)
    assert win.selected == n


def test_click_sensor_toggles_selection(win):
    s2 = tuple(win.scenario.sensors[1].position)
    click(win, s2)
    assert win.selected == 1
    click(win, s2)
    assert win.selected is None


def test_drag_moves_a_sensor_without_selecting_it(win):
    start = tuple(win.scenario.sensors[0].position)
    mouse(win, "press", start)
    mouse(win, "move", (start[0] + 60, start[1] + 40))
    mouse(win, "release", (start[0] + 60, start[1] + 40))
    np.testing.assert_allclose(win.scenario.sensors[0].position, [start[0] + 60, start[1] + 40], atol=1e-6)
    assert win.selected is None


def test_right_click_deletes_and_keeps_selection_consistent(win):
    click(win, tuple(win.scenario.sensors[2].position))  # select S3
    mouse(win, "press", tuple(win.scenario.sensors[0].position), button=3)  # delete S1
    assert win.scenario.labels() == ["S2", "S3"]
    assert win.selected == 1 and win.scenario.sensors[win.selected].label == "S3"
    assert win.scenario.reference == "S2"  # S1 was the reference
    assert win.ref_combo.currentText() == "S2"


def test_shift_click_and_fire_mode_place_the_shot(win):
    n = len(win.scenario.sensors)
    click(win, (700.0, 700.0), modifiers=("shift",))
    assert win.scenario.shot == pytest.approx((700.0, 700.0))
    win.fire_action.setChecked(True)
    click(win, (300.0, 650.0))
    assert win.scenario.shot == pytest.approx((300.0, 650.0))
    # In fire mode a sensor is still a sensor: clicking it selects it.
    click(win, tuple(win.scenario.sensors[0].position))
    assert win.selected == 0
    assert len(win.scenario.sensors) == n


def test_drag_the_shot(win):
    shot = win.scenario.shot
    mouse(win, "press", shot)
    mouse(win, "move", (shot[0] - 50, shot[1] + 25))
    mouse(win, "release", (shot[0] - 50, shot[1] + 25))
    assert win.scenario.shot == pytest.approx((shot[0] - 50, shot[1] + 25))


def test_scroll_zooms_around_the_cursor(win):
    before = np.diff(win.field.ax.get_xlim())[0]
    mouse(win, "scroll", (500.0, 500.0), button="up", step=1)
    assert np.diff(win.field.ax.get_xlim())[0] < before


def test_controls_drive_the_scenario(win):
    win.sigma_slider.setValue(25)
    assert win.scenario.sigma == pytest.approx(0.0025)
    assert win.sigma_label.text() == "2.5 ms"
    draws = [s.noise_draw for s in win.scenario.sensors]
    win.reroll_button.click()
    assert [s.noise_draw for s in win.scenario.sensors] != draws
    win.c_spin.setValue(340.0)
    assert win.scenario.c == 340.0
    win.ref_combo.setCurrentText("S3")
    assert win.scenario.reference == "S3"
    win.refresh()
    assert "t<sub>i</sub> − t<sub>3</sub> (ms)" in win.panel.sensors._html  # Δt column is now relative to S3


def test_preset_keeps_the_noise_setting(win):
    win.sigma_slider.setValue(15)
    win.load_preset("square_four")
    assert win.scenario.sigma == pytest.approx(0.0015)
    assert len(win.scenario.sensors) == 4


def test_loading_a_saved_scenario_syncs_the_controls(win, tmp_path):
    s = PRESETS["far_field"]()
    s.sigma, s.c, s.reference = 0.003, 335.0, "S2"
    s.save(tmp_path / "x.json")
    win.set_scenario(Scenario.load(tmp_path / "x.json"))
    win.refresh()
    assert win.sigma_slider.value() == 30
    assert win.c_spin.value() == 335.0
    assert win.ref_combo.currentText() == "S2"


def test_clear_then_build_from_scratch(win):
    win.clear_scenario()
    win.refresh()
    assert win.mode.fix is None
    for xy in [(200, 200), (800, 250), (480, 800)]:
        click(win, xy)
    click(win, (480, 420), modifiers=("shift",))
    np.testing.assert_allclose(win.mode.fix.position, [480, 420], atol=1e-6)


def test_panel_link_selects_a_sensor(win):
    win.panel.sensorLinkClicked.emit("S3")
    win.refresh()
    assert win.selected == 2
