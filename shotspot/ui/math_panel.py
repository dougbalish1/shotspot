"""The tabbed math panel: Overview, Per-sensor and Method.

Each tab is a QTextBrowser showing HTML. Pages are regenerated on every
redraw (while dragging, many times a second), so each browser keeps its
scroll position across updates. Links of the form "sensor:S3" in any page
select that sensor.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, QUrl, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout, QWidget

from .style import PANEL_CSS


class _Page(QTextBrowser):
    def __init__(self) -> None:
        super().__init__()
        self.setOpenLinks(False)
        self.document().setDefaultStyleSheet(PANEL_CSS)
        self._html = ""
        self.reset_scroll = False

    def set_html(self, html: str) -> None:
        if html == self._html:
            return
        self._html = html
        bar = self.verticalScrollBar()
        position = 0 if self.reset_scroll else bar.value()
        self.reset_scroll = False
        self.setHtml(html)
        bar.setValue(position)

    def has_anchor(self, name: str) -> bool:
        return f'name="{name}"' in self._html


class MathPanel(QTabWidget):
    sensorLinkClicked = Signal(str)  # a sensor label
    stepRequested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.overview = _Page()
        self.sensors = _Page()
        self.method = _Page()

        method_tab = QWidget()
        layout = QVBoxLayout(method_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.step_bar = QWidget()
        bar = QHBoxLayout(self.step_bar)
        bar.setContentsMargins(4, 4, 4, 0)
        self.prev_button = QPushButton("◀ Prev")
        self.next_button = QPushButton("Next ▶")
        self.step_label = QLabel()
        bar.addWidget(self.prev_button)
        bar.addStretch()
        bar.addWidget(self.step_label)
        bar.addStretch()
        bar.addWidget(self.next_button)
        layout.addWidget(self.step_bar)
        layout.addWidget(self.method)

        self.addTab(self.overview, "Overview")
        self.addTab(self.sensors, "Per-sensor")
        self.addTab(method_tab, "Method")

        self._step, self._step_count = 0, 0
        self.prev_button.clicked.connect(lambda: self.stepRequested.emit(self._step - 1))
        self.next_button.clicked.connect(lambda: self.stepRequested.emit(self._step + 1))
        for page in (self.overview, self.sensors, self.method):
            page.anchorClicked.connect(self._on_link)
        self.set_steps(0, 0)

    def set_pages(self, overview: str, sensors: str, method: str) -> None:
        self.overview.set_html(overview)
        self.sensors.set_html(sensors)
        self.method.set_html(method)

    def scroll_to_top(self) -> None:
        """For a brand-new scenario, where the old scroll positions mean nothing."""
        for page in (self.overview, self.sensors, self.method):
            page.reset_scroll = True

    def set_steps(self, step: int, count: int) -> None:
        self._step, self._step_count = step, count
        self.step_bar.setVisible(count > 0)
        self.step_label.setText(f"Step {step + 1} of {count}")
        self.prev_button.setEnabled(step > 0)
        self.next_button.setEnabled(step < count - 1)

    def reveal(self, anchor: str) -> None:
        """Scroll to an anchor: in the current tab if it has one, otherwise in Per-sensor."""
        current = self.currentWidget()
        page = self.method if current is not self.overview and current is not self.sensors else current
        if not page.has_anchor(anchor):
            self.setCurrentWidget(self.sensors)
            page = self.sensors
        # Defer until the freshly set HTML has been laid out.
        QTimer.singleShot(0, lambda: page.scrollToAnchor(anchor))

    def _on_link(self, url: QUrl) -> None:
        if url.scheme() == "sensor":
            self.sensorLinkClicked.emit(url.path())
