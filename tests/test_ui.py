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
from matplotlib.colors import to_hex
from matplotlib.contour import ContourSet
from matplotlib.patches import Circle
from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QCheckBox

from shotspot.core.presets import PRESETS
from shotspot.core.scenario import Scenario
from shotspot.ui.main_window import MainWindow
from shotspot.ui.snapshot import Snapshot
from shotspot.ui.modes import MODES, AccuracyMode, ClosedFormMode, GaussNewtonMode, HyperbolaMode, KnownT0Mode, SweepMode
from shotspot.core.closed_form import solve_closed_form
from shotspot.ui.modes.known_t0 import CHECK, CIRCLES, STEPS, SUBTRACT, T0_ERROR
from shotspot.ui.modes.accuracy import ELLIPSE, FALLBACK_SIGMA, TRIALS
from shotspot.ui.modes.hyperbolas import CROSSING, SIGN
from shotspot.ui.modes.t0_sweep import SLIDER_TICK, START
from shotspot.ui.style import DARK, LIGHT


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


@pytest.fixture
def set_system_palette(app):
    """Change the app palette the way a system light/dark switch does; restore it afterwards."""
    original = app.palette()

    def set_palette(button_color):
        app.setPalette(QPalette(QColor(button_color)))  # derives window/text colors from the button color
        app.processEvents()

    yield set_palette
    app.setPalette(original)
    app.processEvents()


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


def test_ctrl_drag_pans_without_editing_even_from_a_sensor(win):
    sensors = [tuple(s.position) for s in win.scenario.sensors]
    for start in [(600.0, 600.0), sensors[0]]:  # empty space, then right on S1
        before = win.field.ax.get_xlim()
        mouse(win, "press", start, modifiers=("ctrl",))
        mouse(win, "move", (start[0] - 100, start[1]), modifiers=("ctrl",))
        mouse(win, "release", (start[0] - 100, start[1]), modifiers=("ctrl",))
        assert win.field.ax.get_xlim()[0] == pytest.approx(before[0] + 100, abs=1.0)  # dragged left: view moves right
        assert [tuple(s.position) for s in win.scenario.sensors] == sensors
        assert win.selected is None


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


def test_overview_and_field_show_the_other_solution(win):
    win.load_preset("triangle_ambiguous")
    win.refresh()
    overview = win.panel.overview._html
    assert "Two solutions" in overview and "not</b> the true shot" in overview
    labels = [h.get_label() for h in win.mode.legend_handles(Snapshot.take(win.scenario, None, win.theme))]
    assert labels == ["other solution (fits the data too)"]
    win.load_preset("triangle_inside")
    win.refresh()
    assert "Two solutions" not in win.panel.overview._html
    assert "Arrival times are exact" in win.panel.overview._html
    win.sigma_slider.setValue(10)
    win.refresh()
    assert "keeps its draw" in win.panel.overview._html


def test_controls_have_tooltips(win):
    for widget in (win.sigma_slider, win.reroll_button, win.c_spin, win.ref_combo):
        assert widget.toolTip()


def test_theme_follows_the_system_palette_while_running(win, set_system_palette):
    win.set_selected(0)
    for button, theme in [("#efefef", LIGHT), ("#353535", DARK), ("#efefef", LIGHT)]:
        set_system_palette(button)
        win.refresh()
        assert win.theme is theme
        assert to_hex(win.field.figure.get_facecolor()) == theme.surface
        page = win.panel.sensors
        # Unstyled panel text takes the page's palette color, so it must be the theme's ink, not the system's.
        assert page.palette().color(QPalette.ColorRole.Text).name() == theme.ink
        assert page.palette().color(QPalette.ColorRole.Base).name() == theme.surface
        assert theme.selected_wash in page._html and theme.sensor_colors[0] in page._html


# ---- modes -------------------------------------------------------------------


def use_mode(win, cls):
    win.mode_combo.setCurrentIndex(MODES.index(cls))
    win.refresh()
    assert isinstance(win.mode, cls)


def circles_on_field(win):
    return [p for p in win.field.ax.patches if isinstance(p, Circle)]


def test_every_mode_renders_every_preset_at_every_step(win):
    for index in range(len(MODES)):
        win.mode_combo.setCurrentIndex(index)
        for key in PRESETS:
            win.load_preset(key)
            win.refresh()
            for step in range(max(win.panel._step_count, 1)):
                win.mode.set_step(step)
                win.refresh()
                assert win.mode.step == step
                assert win.panel.method._html


def test_known_t0_recovers_the_shot_and_says_t0_was_given(win):
    use_mode(win, KnownT0Mode)
    np.testing.assert_allclose(win.mode.fix.position, win.scenario.shot, atol=1e-6)
    assert "t<sub>0</sub> given, not estimated" in win.panel.overview._html
    assert len(circles_on_field(win)) == len(win.scenario.sensors)
    assert win.panel._step_count == len(STEPS)


def test_known_t0_catch_shows_the_wrong_t0_run(win):
    use_mode(win, KnownT0Mode)
    win.mode.set_step(CHECK)
    win.refresh()
    html = win.panel.method._html
    assert "shrinks by δ" in html and "slides sideways" in html and "p = u + v·t<sub>0</sub>" in html
    assert len(circles_on_field(win)) == 2 * len(win.scenario.sensors)  # the real circles and the shrunken ones
    assert [h.get_label() for h in win.mode.legend_handles(Snapshot.take(win.scenario, None, win.theme))] == [
        "circles for t₀ 10 ms late", "their best point p′"]
    # p′ moves along the closed form's line p = u + v·t₀, so it is exactly |v|·Δt₀ from p_est.
    moved = np.linalg.norm(win.mode.wrong_t0.position - win.mode.result.position)
    v = solve_closed_form(win.scenario.positions(), win.scenario.measured_times(), win.scenario.c).quadratic.v
    assert moved == pytest.approx(np.linalg.norm(v) * T0_ERROR, rel=1e-6)
    win.mode.set_step(SUBTRACT)
    win.refresh()
    assert len(circles_on_field(win)) == len(win.scenario.sensors)


def test_known_t0_subtract_step_draws_one_line_per_non_reference_sensor(win):
    use_mode(win, KnownT0Mode)
    win.load_preset("square_four")
    win.mode.set_step(CIRCLES)
    win.refresh()
    lines_before = len(win.field.ax.lines)
    win.mode.set_step(SUBTRACT)
    win.refresh()
    assert len(win.field.ax.lines) - lines_before == 3


def test_step_links_jump_and_the_step_is_clamped(win):
    use_mode(win, KnownT0Mode)
    win.panel._on_link(QUrl("step:3"))
    win.refresh()
    assert win.mode.step == 3 and win.panel._step == 3
    win.mode.step = 99
    win.refresh()
    assert win.mode.step == len(STEPS) - 1
    win.clear_scenario()
    win.refresh()
    assert win.mode.step == 0 and win.panel.step_bar.isHidden()


def test_known_t0_with_two_sensors_draws_circles_but_cannot_solve(win):
    use_mode(win, KnownT0Mode)
    win.clear_scenario()
    win.refresh()
    for xy in [(200, 200), (800, 250)]:
        click(win, xy)
    click(win, (480, 420), modifiers=("shift",))
    assert win.mode.fix is None
    assert len(circles_on_field(win)) == 2
    assert "two circles cross at two points" in win.panel.method._html


def test_known_t0_collinear_sensors_explain_the_parallel_lines(win):
    use_mode(win, KnownT0Mode)
    win.clear_scenario()
    win.refresh()
    for xy in [(100, 500), (400, 500), (900, 500)]:
        click(win, xy)
    click(win, (480, 700), modifiers=("shift",))
    win.mode.set_step(STEPS.index("Solve"))
    win.refresh()
    assert win.mode.fix is None
    assert "the lines are parallel" in win.panel.method._html


def test_selecting_a_sensor_on_the_method_tab_stays_there(win):
    use_mode(win, KnownT0Mode)
    win.panel.setCurrentIndex(2)
    win.select_label("S2")
    win.refresh()
    assert win.panel.currentIndex() == 2
    assert win.panel.method.has_anchor("sensor-S2")


def test_sweep_guess_follows_the_slider_without_redoing_the_curve(win):
    use_mode(win, SweepMode)
    view, mode = win.mode.view(), win.mode
    assert view.isVisible() and mode.curve is not None
    curve = mode.curve
    view.slider.setValue(view.slider.value() - 500)  # 50 ms earlier
    win.refresh()
    assert mode.curve is curve  # cached: only the guess changed
    assert mode.t0_guess == pytest.approx(mode.t0_range[0] + view.slider.value() * SLIDER_TICK)
    assert mode.fix.t0 == mode.t0_guess
    assert view.spin.value() == pytest.approx(mode.t0_guess, abs=1e-6)


def test_sweep_jump_to_best_dip_finds_the_shot(win):
    use_mode(win, SweepMode)
    assert np.linalg.norm(win.mode.fix.position - win.scenario.shot) > 1.0  # the starting guess is wrong
    win.mode.view().jump.click()
    win.refresh()
    assert win.mode.t0_guess == pytest.approx(win.scenario.t0, abs=1e-7)
    np.testing.assert_allclose(win.mode.fix.position, win.scenario.shot, atol=1e-3)


def test_sweep_plot_click_sets_the_guess_and_other_modes_hide_the_plot(win):
    use_mode(win, SweepMode)
    view = win.mode.view()
    canvas = view.canvas
    target = 3.5
    x, y = view.ax.transData.transform((target, 1.0))
    view._on_press(MouseEvent("button_press_event", canvas, x, y, button=1))
    view._pressed = False
    win.refresh()
    assert win.mode.t0_guess == pytest.approx(target, abs=0.01)
    use_mode(win, KnownT0Mode)
    assert not view.isVisible()


def test_sweep_new_scenario_resets_the_guess_and_shows_two_dips(win):
    use_mode(win, SweepMode)
    win.mode.view().jump.click()
    win.refresh()
    win.load_preset("triangle_ambiguous")
    win.refresh()
    lo, hi = win.mode.t0_range
    assert win.mode.t0_guess == pytest.approx(hi - START * (hi - lo))
    win.mode.set_step(2)
    win.refresh()
    assert len(win.mode.curve.dips) == 2
    assert "Two guesses" in win.panel.method._html


def contours_on_field(win):
    return [c for c in win.field.ax.collections if isinstance(c, ContourSet)]


def test_hyperbolas_one_curve_per_pair_and_all_pairs_adds_the_rest(win):
    use_mode(win, HyperbolaMode)
    win.load_preset("square_four")
    win.refresh()
    assert len(contours_on_field(win)) == 3  # N − 1 reference pairs
    pairs_box, asymptotes_box = win.mode.controls().findChildren(QCheckBox)
    pairs_box.setChecked(True)
    win.refresh()
    assert len(contours_on_field(win)) == 6  # all N(N − 1)/2 pairs
    lines = len(win.field.ax.lines)
    asymptotes_box.setChecked(True)
    win.refresh()
    assert len(win.field.ax.lines) == lines + 12  # two asymptote rays per pair
    pairs_box.setChecked(False)
    asymptotes_box.setChecked(False)


def test_hyperbolas_sign_step_adds_the_mirror_branches(win):
    use_mode(win, HyperbolaMode)
    win.mode.set_step(SIGN)
    win.refresh()
    assert len(contours_on_field(win)) == 2 * (len(win.scenario.sensors) - 1)


def test_hyperbolas_redraw_when_the_view_changes(win):
    use_mode(win, HyperbolaMode)
    mouse(win, "scroll", (500.0, 500.0), button="down", step=-1)  # zoom out (mouse() refreshes afterwards)
    x0, x1 = win.field.ax.get_xlim()
    reach = max(path.vertices[:, 0].max() for c in contours_on_field(win) for path in c.get_paths())
    assert reach > x1 - 0.05 * (x1 - x0)  # the curves still run to the edge of the wider view


def test_hyperbolas_ambiguous_layout_crosses_twice(win):
    use_mode(win, HyperbolaMode)
    win.load_preset("triangle_ambiguous")
    win.mode.set_step(CROSSING)
    win.refresh()
    assert len(win.mode.candidates) == 2
    assert "cross twice" in win.panel.method._html


def test_hyperbolas_impossible_time_difference_draws_no_curve(win):
    use_mode(win, HyperbolaMode)
    win.load_preset("far_field")  # S4 − S1: c·Δt = −212.58 m across a 212.60 m baseline
    for sensor in win.scenario.sensors:
        sensor.noise_draw = 0.0
    win.scenario.sensors[3].noise_draw = -1.0
    win.sigma_slider.setValue(10)  # 1 ms: S4 now "hears" it 0.34 m too early to be possible
    win.mode.set_step(SIGN)
    win.refresh()
    assert not win.mode.reference_pairs[2].feasible
    assert "impossible: no curve" in win.panel.method._html


def step_through(win):
    """Visit every Method step; return the pages' HTML."""
    pages = []
    for step in range(win.panel._step_count):
        win.mode.set_step(step)
        win.refresh()
        pages.append(win.panel.method._html)
    return pages


def test_closed_form_three_sensors_takes_the_quadratic_route(win):
    use_mode(win, ClosedFormMode)
    win.load_preset("triangle_ambiguous")
    win.refresh()
    assert win.mode.steps == ["Square", "Subtract", "p = u + v·t₀", "Quadratic", "Which root?"]
    pages = step_through(win)
    assert "D = B² − 4AC" in pages[3]
    assert "Both roots survive" in pages[4]
    labels = [h.get_label() for h in win.mode.legend_handles(Snapshot.take(win.scenario, None, win.theme))]
    assert labels == ["p = u + v·t₀", "root kept"]


def test_closed_form_routes_for_four_sensors(win):
    use_mode(win, ClosedFormMode)
    win.load_preset("square_four")
    win.refresh()
    assert win.mode.steps == ["Square", "Subtract", "Solve"]
    step_through(win)
    win.scenario.shot = (500.0, 610.0)  # on the square's line of symmetry: t₀ drops out of the linear system
    win.refresh()
    assert win.mode.steps == ["Square", "Subtract", "Rank", "p = u + v·t₀", "Quadratic", "Which root?"]
    pages = step_through(win)
    assert "rank 2, not 3" in pages[2]
    assert "<b>rejected</b>" in pages[5] and "<b>kept</b>" in pages[5]
    np.testing.assert_allclose(win.mode.fix.position, [500.0, 610.0], atol=1e-6)


def test_closed_form_four_sensors_explains_the_unused_reference_equation(win):
    use_mode(win, ClosedFormMode)
    win.load_preset("square_four")
    win.sigma_slider.setValue(20)
    win.refresh()
    solve = step_through(win)[2]
    assert "only ever subtracted" in solve and "polished by Gauss-Newton" in solve
    gn = win.mode.polished
    assert gn.converged and gn.iterations[-1].cost < 0.01 * gn.iterations[0].cost
    win.sigma_slider.setValue(0)
    win.refresh()
    assert "nothing to polish" in step_through(win)[2]


def test_closed_form_collinear_sensors(win):
    use_mode(win, ClosedFormMode)
    win.clear_scenario()
    win.refresh()
    for xy in [(100, 500), (400, 500), (900, 500)]:
        click(win, xy)
    click(win, (480, 700), modifiers=("shift",))
    assert win.mode.steps == ["Square", "Subtract", "Collinear"]
    assert win.mode.fix is None
    assert "mirror image" in step_through(win)[2]


def test_gauss_newton_converges_from_the_array_center(win):
    use_mode(win, GaussNewtonMode)
    mode = win.mode
    np.testing.assert_allclose(mode.start, win.scenario.positions().mean(axis=0))
    assert mode.result.converged
    assert win.panel._step_count == len(mode.result.iterations) + 1  # Start, each iteration, Result
    np.testing.assert_allclose(mode.fix.position, win.scenario.shot, atol=1e-6)
    pages = step_through(win)
    assert "JᵀJ" in pages[1] and "converged" in pages[-1]


def test_gauss_newton_start_is_draggable_and_far_starts_diverge(win):
    use_mode(win, GaussNewtonMode)
    start = tuple(win.mode.start)
    mouse(win, "press", start)  # the diamond is a handle: dragging it moves the start, not a sensor
    mouse(win, "move", (100.0, 100.0))
    mouse(win, "release", (100.0, 100.0))
    np.testing.assert_allclose(win.mode.start, [100.0, 100.0], atol=1e-6)
    assert len(win.scenario.sensors) == 3
    assert win.mode.result.status == "diverged" and win.mode.fix is None
    assert "Drag the diamond closer" in step_through(win)[-1]
    win.mode.controls().click()  # Reset start
    win.refresh()
    np.testing.assert_allclose(win.mode.start, win.scenario.positions().mean(axis=0))


def test_gauss_newton_can_converge_to_the_other_solution(win):
    use_mode(win, GaussNewtonMode)
    win.load_preset("triangle_ambiguous")
    win.refresh()
    assert win.mode.result.converged
    assert np.linalg.norm(win.mode.fix.position - win.scenario.shot) > 100  # the phantom, from the centroid
    assert "isn't the shot" in step_through(win)[-1]


def test_accuracy_map_and_the_zero_noise_fallback(win):
    win.sigma_slider.setValue(0)
    use_mode(win, AccuracyMode)
    assert len(win.field.ax.images) == 1 and contours_on_field(win)
    assert win.mode.sigma == FALLBACK_SIGMA
    assert "slider is at 0" in win.panel.method._html
    win.sigma_slider.setValue(20)
    win.refresh()
    assert win.mode.sigma == pytest.approx(0.002)


def test_accuracy_monte_carlo_runs_and_is_cleared_when_the_shot_moves(win):
    use_mode(win, AccuracyMode)
    view = win.mode.view()
    view.trials.setValue(400)
    view.run.click()
    win.refresh()
    assert len(win.mode.mc.estimates) == 400
    win.mode.set_step(TRIALS)
    win.refresh()
    html = win.panel.method._html
    assert "inside 1σ ellipse" in html and "inside 95% ellipse" in html
    win.scenario.shot = (520.0, 430.0)
    win.refresh()
    assert win.mode.mc is None


def test_accuracy_monte_carlo_reports_fixes_at_the_other_solution(win):
    use_mode(win, AccuracyMode)
    win.load_preset("triangle_ambiguous")
    win.refresh()
    assert win.mode.other is not None and win.mode.cluster.name == "the shot"
    labels = [h.get_label() for h in win.mode.legend_handles(Snapshot.take(win.scenario, None, win.theme))]
    assert labels[-1] == "other solution (fits the data too)"
    view = win.mode.view()
    view.trials.setValue(50)
    view.run.click()
    win.mode.set_step(TRIALS)
    win.refresh()
    assert win.mode.cluster.name == "the other solution" and win.mode.cluster.share_other > 0.9
    html = win.panel.method._html
    assert "landed at the other solution" in html and "from the other solution" in html


def test_accuracy_far_field_ellipse_is_stretched_along_the_line_of_sight(win):
    use_mode(win, AccuracyMode)
    win.load_preset("far_field")
    win.mode.set_step(ELLIPSE)
    win.refresh()
    assert "along the line of sight" in win.panel.method._html


# ---- wavefront animation ---------------------------------------------------------


def test_wavefront_animation_reaches_sensors_in_order_then_stops(win):
    now = [0.0]
    win.animation._now = lambda: now[0]  # drive the animation's clock by hand
    win.animate_action.setChecked(True)
    assert win.animation.running
    true_times = win.scenario.true_times()
    halfway = (true_times.min() + true_times.max()) / 2
    now[0] = (halfway - win.scenario.t0) * win.animation.slowdown
    win.animation._tick()
    win.refresh()
    reached = true_times <= win.animation.clock
    assert 0 < reached.sum() < len(reached)
    assert ">—</td>" in win.panel.sensors._html  # the unreached sensors' arrival times aren't known yet
    assert len(circles_on_field(win)) == 1  # the wavefront (the mode's overlays are hidden)
    now[0] = 1e6
    win.animation._tick()
    win.refresh()
    assert not win.animation.running and not win.animate_action.isChecked()
    assert ">—</td>" not in win.panel.sensors._html


def test_animation_needs_a_shot(win):
    win.clear_scenario()
    win.refresh()
    win.animate_action.setChecked(True)
    assert not win.animation.running and not win.animate_action.isChecked()
