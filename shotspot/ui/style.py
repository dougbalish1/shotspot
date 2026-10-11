"""Colors and the HTML stylesheet shared by the field and the math panel.

Sensor colors are a fixed, colorblind-validated categorical order. A sensor's
color follows its *label* (S3 is always slot 3), never its position in the
list, so deleting S2 doesn't repaint S3. Sensors past the 8th slot fall back
to gray. Every sensor is also labeled with text, so color never carries
identity on its own.
"""

SENSOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OVERFLOW_COLOR = "#898781"

SURFACE = "#fcfcfb"  # field background
INK = "#0b0b0b"  # primary text and the shot/estimate markers
INK_SECONDARY = "#52514e"
MUTED = "#898781"  # axis labels, ticks
GRID = "#e1e0d9"  # hairline grid
AXIS = "#c3c2b7"  # field boundary
SELECTED_WASH = "#f0efec"  # background behind the selected sensor's rows in the panel


def sensor_color(label: str) -> str:
    number = label[1:]
    if number.isdigit() and 1 <= int(number) <= len(SENSOR_COLORS):
        return SENSOR_COLORS[int(number) - 1]
    return OVERFLOW_COLOR


PANEL_CSS = f"""
body {{ font-family: 'DejaVu Sans', 'Segoe UI', sans-serif; font-size: 10pt; color: {INK}; }}
h2 {{ font-size: 13pt; margin-top: 4px; margin-bottom: 4px; }}
h3 {{ font-size: 11pt; margin-top: 12px; margin-bottom: 2px; }}
p {{ margin-top: 3px; margin-bottom: 3px; }}
table.data {{ border-collapse: collapse; margin-top: 4px; }}
table.data th {{ color: {INK_SECONDARY}; font-weight: normal; text-align: right;
                 padding: 2px 5px; border-bottom: 1px solid {AXIS}; }}
table.data td {{ text-align: right; padding: 2px 5px; border-bottom: 1px solid {GRID}; }}
table.kv td {{ padding: 1px 6px 1px 0px; }}
.muted {{ color: {MUTED}; }}
.note {{ color: {INK_SECONDARY}; }}
.eq {{ margin-left: 14px; }}
.selected {{ background-color: {SELECTED_WASH}; }}
a {{ color: {INK}; text-decoration: none; }}
"""
