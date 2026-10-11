"""The main window: field on the left, math panel on the right, controls on top.

Data flow: every edit (mouse, slider, preset, file) changes `self.scenario`,
then calls `request_refresh`. Refreshes are coalesced: a burst of drag events
produces one redraw, not dozens. A refresh takes one Snapshot and hands it
to the mode, the field and the math panel, so all three show the same numbers.

The light or dark theme follows the system palette, including when the user
switches it while the app is running (Qt sends a PaletteChange).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.presets import PRESETS
from ..core.scenario import Scenario
from .animation import WavefrontAnimation
from .field import FieldCanvas
from .math_panel import MathPanel
from .modes import MODES, Mode
from .pages import overview_html, sensors_html
from .snapshot import Snapshot
from .style import theme_for

SIGMA_STEP = 1e-4  # seconds per slider tick (0.1 ms)
SIGMA_MAX_TICKS = 100  # 10 ms
HINT = (
    "Click: add sensor · Drag: move · Click sensor: select · Right-click: delete · "
    "Shift-click: fire · Scroll: zoom · Ctrl-drag: pan · Home: fit view · Esc: deselect"
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.rng = np.random.default_rng()
        self.scenario = PRESETS["triangle_inside"]()
        self.selected: int | None = None
        self.theme = theme_for(self.palette())
        self.modes = [cls() for cls in MODES]
        self.mode = self.modes[0]
        self.animation = WavefrontAnimation(self)
        self._pending_reveal: str | None = None

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self.refresh)

        self._build_toolbar()
        self._build_central()
        self._build_status_bar()
        self._connect()
        self.panel.set_theme(self.theme)
        self._sync_controls()
        self.field.reset_view(self.scenario.width, self.scenario.height)
        self.refresh()

    # ---- construction ----------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        presets = QToolButton()
        presets.setText("Presets")
        presets.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(presets)
        for key, factory in PRESETS.items():
            menu.addAction(factory().name, lambda key=key: self.load_preset(key))
        presets.setMenu(menu)
        tb.addWidget(presets)

        tb.addAction("Clear", self.clear_scenario)
        tb.addAction("Open…", self.open_scenario).setShortcut(QKeySequence.StandardKey.Open)
        tb.addAction("Save…", self.save_scenario).setShortcut(QKeySequence.StandardKey.Save)
        tb.addSeparator()

        self.fire_action = QAction("Fire mode", self)
        self.fire_action.setCheckable(True)
        self.fire_action.setShortcut("F")
        self.fire_action.setToolTip("When on, clicking empty space places the shot (F). Shift-click always does.")
        tb.addAction(self.fire_action)
        tb.addAction("Fit view", lambda: self.field.reset_view()).setShortcut("Home")
        self.animate_action = QAction("Animate", self)
        self.animate_action.setCheckable(True)
        self.animate_action.setShortcut("A")
        self.animate_action.setToolTip("Watch the sound spread from the shot, slowed down (A)")
        tb.addAction(self.animate_action)
        tb.addSeparator()

        tb.addWidget(QLabel(" Mode: "))
        self.mode_combo = QComboBox()
        for mode in self.modes:
            self.mode_combo.addItem(mode.title)
        self.mode_combo.setToolTip(f"Ctrl+1 … Ctrl+{len(self.modes)} switch modes")
        tb.addWidget(self.mode_combo)

    def _build_central(self) -> None:
        self.field = FieldCanvas()
        self.panel = MathPanel()

        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(8, 2, 8, 4)

        row.addWidget(QLabel("Timing noise σ"))
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(0, SIGMA_MAX_TICKS)
        self.sigma_slider.setFixedWidth(180)
        self.sigma_label = QLabel()
        self.sigma_label.setMinimumWidth(60)
        row.addWidget(self.sigma_slider)
        row.addWidget(self.sigma_label)
        self.reroll_button = QPushButton("Re-roll noise")
        row.addWidget(self.reroll_button)
        row.addSpacing(16)

        row.addWidget(QLabel("Speed of sound c"))
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(250.0, 400.0)
        self.c_spin.setDecimals(1)
        self.c_spin.setSingleStep(1.0)
        self.c_spin.setSuffix(" m/s")
        row.addWidget(self.c_spin)
        row.addSpacing(16)

        row.addWidget(QLabel("Reference"))
        self.ref_combo = QComboBox()
        self.ref_combo.setMinimumWidth(70)
        row.addWidget(self.ref_combo)
        row.addStretch()
        # Each mode's own widgets (sliders, toggles, …), shown only while that mode is active.
        self.mode_controls = QWidget()
        mode_row = QHBoxLayout(self.mode_controls)
        mode_row.setContentsMargins(0, 0, 0, 0)
        self._mode_widgets: dict[Mode, list[QWidget]] = {}
        for mode in self.modes:
            widget = mode.controls()
            if widget is not None:
                mode_row.addWidget(widget)
                widget.setVisible(mode is self.mode)
                self._mode_widgets.setdefault(mode, []).append(widget)
        row.addWidget(self.mode_controls)

        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.field, stretch=1)
        for mode in self.modes:  # panels under the field (the sweep's plot), shown only for their mode
            widget = mode.view()
            if widget is not None:
                column.addWidget(widget)
                widget.setVisible(mode is self.mode)
                self._mode_widgets.setdefault(mode, []).append(widget)
        column.addWidget(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([3000, 1000])
        self.setCentralWidget(splitter)

    def _build_status_bar(self) -> None:
        self.cursor_label = QLabel()
        self.statusBar().addWidget(QLabel(HINT))
        self.statusBar().addPermanentWidget(self.cursor_label)

    def _connect(self) -> None:
        f = self.field
        f.sensorAdded.connect(self.on_sensor_added)
        f.sensorMoved.connect(self.on_sensor_moved)
        f.sensorRemoved.connect(self.on_sensor_removed)
        f.sensorClicked.connect(self.on_sensor_clicked)
        f.shotMoved.connect(self.on_shot_moved)
        f.handleMoved.connect(lambda name, x, y: self.mode.move_handle(name, x, y))
        f.cursorMoved.connect(lambda x, y: self.cursor_label.setText(f"x = {x:7.1f} m   y = {y:7.1f} m"))
        f.viewChanged.connect(self.request_refresh)

        self.panel.sensorLinkClicked.connect(self.select_label)
        self.panel.stepRequested.connect(lambda step: self.mode.set_step(step))
        for mode in self.modes:
            mode.changed.connect(self.request_refresh)
        self.mode_combo.currentIndexChanged.connect(self.set_mode)

        self.fire_action.toggled.connect(self.on_fire_toggled)
        self.animate_action.toggled.connect(self.on_animate_toggled)
        self.animation.frame.connect(self.request_refresh)
        self.animation.finished.connect(lambda: self.animate_action.setChecked(False))
        self.sigma_slider.valueChanged.connect(self.on_sigma_changed)
        self.reroll_button.clicked.connect(self.on_reroll)
        self.c_spin.valueChanged.connect(self.on_c_changed)
        self.ref_combo.currentTextChanged.connect(self.on_reference_changed)

        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=lambda: self.set_selected(None))
        QShortcut(QKeySequence(Qt.Key.Key_F11), self, activated=self.toggle_full_screen)
        for n in range(min(len(self.modes), 9)):
            QShortcut(QKeySequence(f"Ctrl+{n + 1}"), self, activated=lambda n=n: self.mode_combo.setCurrentIndex(n))

    # ---- refresh -----------------------------------------------------------

    def request_refresh(self) -> None:
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def refresh(self) -> None:
        snap = Snapshot.take(self.scenario, self.selected, self.theme, self.animation.clock, self.animation.slowdown)
        mode = self.mode
        mode.update(snap)
        steps = mode.step_count(snap)
        mode.step = min(max(mode.step, 0), max(steps - 1, 0))  # the scenario may have changed under it
        self.field.render(snap, mode)
        self.panel.set_pages(overview_html(snap, mode), sensors_html(snap, mode), mode.method_html(snap))
        self.panel.set_steps(mode.step, steps)
        self.setWindowTitle(f"ShotSpot: {self.scenario.name}")
        if self._pending_reveal is not None:
            self.panel.reveal(self._pending_reveal)
            self._pending_reveal = None

    def _sync_controls(self) -> None:
        """Make the widgets show the scenario's values, without echoing changes back."""
        widgets = (self.sigma_slider, self.c_spin, self.ref_combo)
        for w in widgets:
            w.blockSignals(True)
        self.sigma_slider.setValue(round(self.scenario.sigma / SIGMA_STEP))
        self.c_spin.setValue(self.scenario.c)
        self.ref_combo.clear()
        self.ref_combo.addItems(self.scenario.labels())
        self.ref_combo.setCurrentText(self.scenario.labels()[self.scenario.reference_index()] if self.scenario.sensors else "")
        for w in widgets:
            w.blockSignals(False)
        self.sigma_label.setText(f"{self.scenario.sigma * 1e3:.1f} ms")

    def set_mode(self, index: int) -> None:
        self.mode = self.modes[index]
        for mode, widgets in self._mode_widgets.items():
            for widget in widgets:
                widget.setVisible(mode is self.mode)
        self.panel.method.reset_scroll = True
        self.request_refresh()

    # ---- selection -------------------------------------------------------

    def set_selected(self, index: int | None) -> None:
        self.selected = index
        if index is not None:
            self._pending_reveal = f"sensor-{self.scenario.sensors[index].label}"
        self.request_refresh()

    def select_label(self, label: str) -> None:
        labels = self.scenario.labels()
        if label in labels:
            self.set_selected(labels.index(label))

    # ---- field edits -------------------------------------------------------

    def on_sensor_added(self, x: float, y: float) -> None:
        self.scenario.add_sensor(x, y, self.rng)
        self._sync_controls()
        self.set_selected(len(self.scenario.sensors) - 1)

    def on_sensor_moved(self, index: int, x: float, y: float) -> None:
        if 0 <= index < len(self.scenario.sensors):
            self.scenario.sensors[index].x, self.scenario.sensors[index].y = float(x), float(y)
            self.request_refresh()

    def on_sensor_removed(self, index: int) -> None:
        self.scenario.remove_sensor(index)
        if self.selected is not None:
            if self.selected == index:
                self.selected = None
            elif self.selected > index:
                self.selected -= 1
        self._sync_controls()
        self.request_refresh()

    def on_sensor_clicked(self, index: int) -> None:
        self.set_selected(None if self.selected == index else index)

    def on_shot_moved(self, x: float, y: float) -> None:
        self.scenario.shot = (float(x), float(y))
        self.request_refresh()

    # ---- controls --------------------------------------------------------

    def on_fire_toggled(self, on: bool) -> None:
        self.field.fire_mode = on
        self.field.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def on_animate_toggled(self, on: bool) -> None:
        if not on:
            self.animation.stop()
        elif not self.animation.start(self.scenario):
            self.animate_action.setChecked(False)  # no shot to animate
            self.statusBar().showMessage("Fire a shot first (shift-click)", 3000)

    def on_sigma_changed(self, ticks: int) -> None:
        self.scenario.sigma = ticks * SIGMA_STEP
        self.sigma_label.setText(f"{self.scenario.sigma * 1e3:.1f} ms")
        self.request_refresh()

    def on_reroll(self) -> None:
        self.scenario.reroll_noise(self.rng)
        self.request_refresh()

    def on_c_changed(self, value: float) -> None:
        self.scenario.c = float(value)
        self.request_refresh()

    def on_reference_changed(self, label: str) -> None:
        if label:
            self.scenario.reference = label
            self.request_refresh()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            theme = theme_for(self.palette())
            if theme is not self.theme:
                self.theme = theme
                self.panel.set_theme(theme)
                self.request_refresh()

    def toggle_full_screen(self) -> None:
        self.showMaximized() if self.isFullScreen() else self.showFullScreen()

    # ---- scenarios -------------------------------------------------------

    def set_scenario(self, scenario: Scenario) -> None:
        self.animate_action.setChecked(False)
        self.scenario = scenario
        self.selected = None
        self._pending_reveal = None
        self._sync_controls()
        self.field.reset_view(scenario.width, scenario.height)
        self.panel.scroll_to_top()
        self.request_refresh()

    def load_preset(self, key: str) -> None:
        scenario = PRESETS[key]()
        scenario.sigma = self.scenario.sigma  # presets don't carry noise settings; keep the slider's
        self.set_scenario(scenario)

    def clear_scenario(self) -> None:
        self.set_scenario(Scenario(name="Untitled", sigma=self.scenario.sigma))

    def open_scenario(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open scenario", "", "Scenario (*.json)")
        if not path:
            return
        try:
            self.set_scenario(Scenario.load(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Couldn't open scenario", f"{Path(path).name}: {exc}")

    def save_scenario(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save scenario", f"{self.scenario.name}.json", "Scenario (*.json)")
        if not path:
            return
        try:
            self.scenario.save(path)
        except OSError as exc:
            QMessageBox.warning(self, "Couldn't save scenario", str(exc))
