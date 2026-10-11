"""Colors and the HTML stylesheet shared by the field and the math panel.

There are two themes, LIGHT and DARK, with the same roles. The window picks
the one that matches the system palette (see `theme_for`) and passes it along
in every Snapshot, so the field, the panel and the modes always agree.

Sensor colors are a fixed, colorblind-validated categorical order. A sensor's
color follows its *label* (S3 is always slot 3), never its position in the
list, so deleting S2 doesn't repaint S3. Sensors past the 8th slot fall back
to gray. Every sensor is also labeled with text, so color never carries
identity on its own. DARK uses the same eight hues, each stepped (validated as
a set) to stand out against the dark surface.

The sequential ramp (for magnitudes, e.g. the accuracy heatmap) is one hue,
running from the surface outward: light to dark on the light field, dark to
light on the dark one, so a small value blends into the background and a large
one stands out. Each theme uses only the half of the blue ramp nearest its
surface, so ink drawn on top (markers, labels) keeps about 4:1 contrast or more
anywhere on the map.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QPalette


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str  # field and panel background
    ink: str  # primary text and the shot/estimate markers
    ink_secondary: str
    muted: str  # axis labels, ticks
    grid: str  # hairline grid
    axis: str  # field boundary
    selected_wash: str  # background behind the selected sensor's rows in the panel
    sensor_colors: tuple[str, ...]
    overflow_color: str  # sensors past the last slot
    sequential: tuple[str, ...]  # low → high

    def sensor_color(self, label: str) -> str:
        number = label[1:]
        if number.isdigit() and 1 <= int(number) <= len(self.sensor_colors):
            return self.sensor_colors[int(number) - 1]
        return self.overflow_color

    def panel_css(self) -> str:
        # No `color` on body: Qt's rich text ignores it there. Unstyled text takes the
        # page's palette Text color instead, which MathPanel sets to `ink`.
        return f"""
body {{ font-family: 'DejaVu Sans', 'Segoe UI', sans-serif; font-size: 10pt; }}
h2 {{ font-size: 13pt; margin-top: 4px; margin-bottom: 4px; }}
h3 {{ font-size: 11pt; margin-top: 12px; margin-bottom: 2px; }}
p {{ margin-top: 3px; margin-bottom: 3px; }}
table.data {{ border-collapse: collapse; margin-top: 4px; }}
table.data th {{ color: {self.ink_secondary}; font-weight: normal; text-align: right;
                 padding: 2px 5px; border-bottom: 1px solid {self.axis}; }}
table.data td {{ text-align: right; padding: 2px 5px; border-bottom: 1px solid {self.grid}; }}
table.kv td {{ padding: 1px 6px 1px 0px; }}
.muted {{ color: {self.muted}; }}
.note {{ color: {self.ink_secondary}; }}
.eq {{ margin-left: 14px; }}
.selected {{ background-color: {self.selected_wash}; }}
a {{ color: {self.ink}; text-decoration: none; }}
"""


# The blue ramp's steps 100 to 700, light to dark.
BLUES = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b")

LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    selected_wash="#f0efec",
    sensor_colors=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    overflow_color="#898781",
    sequential=BLUES[:7],  # 100 → 400: black ink stays above 5:1 on the darkest
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#52514e",  # one neutral step brighter than the usual #383835, which vanishes as a dashed line
    selected_wash="#383835",
    sensor_colors=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    overflow_color="#898781",
    sequential=BLUES[:7:-1],  # 700 → 500: light ink stays above 4:1 on the lightest
)


def theme_for(palette: QPalette) -> Theme:
    """DARK if the window background is darker than its text (a dark system theme), else LIGHT."""
    window = palette.color(QPalette.ColorRole.Window).lightness()
    text = palette.color(QPalette.ColorRole.WindowText).lightness()
    return DARK if window < text else LIGHT
