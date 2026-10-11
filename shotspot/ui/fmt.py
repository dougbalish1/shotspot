"""Number and symbol formatting for the math panel.

Consistent precision everywhere: positions and ranges to the centimeter,
absolute times to the microsecond, time differences in milliseconds.
Negative numbers use a real minus sign (−), and when substituted into an
equation they are wrapped in parentheses so "x − (−3.20)" reads correctly.
"""

from __future__ import annotations

from collections.abc import Sequence

MINUS = "−"


def num(x: float, nd: int = 2) -> str:
    text = f"{x:.{nd}f}"
    if text.startswith("-"):
        text = text[1:] if float(text) == 0.0 else MINUS + text[1:]  # never show "−0.00"
    return text


def wrap(text: str) -> str:
    """Parenthesize a formatted value if it is negative, for use inside an expression."""
    return f"({text})" if text.startswith(MINUS) else text


def term(x: float, nd: int = 2) -> str:
    """A number ready to drop into an expression: negatives get parentheses."""
    return wrap(num(x, nd))


def signed(x: float, nd: int = 2) -> str:
    text = num(x, nd)
    return text if text.startswith(MINUS) or float(x) == 0.0 else "+" + text


def meters(x: float, nd: int = 2) -> str:
    return f"{num(x, nd)} m"


def seconds(t: float, nd: int = 6) -> str:
    return f"{num(t, nd)} s"


def millis(t: float, nd: int = 3) -> str:
    """A time *difference*, given in seconds, shown in milliseconds."""
    return f"{num(t * 1e3, nd)} ms"


def point(p: Sequence[float], nd: int = 2) -> str:
    return f"({num(p[0], nd)}, {num(p[1], nd)})"


def index_of(label: str) -> str:
    """'S3' → '3', the subscript used for that sensor in equations."""
    return label[1:] if label[1:].isdigit() else label


def sub(base: str, label: str) -> str:
    """sub('t', 'S3') → t₃ (as HTML)."""
    return f"{base}<sub>{index_of(label)}</sub>"


_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def usub(base: str, label: str) -> str:
    """usub('s', 'S3') → 's₃' with a Unicode subscript, for plain-text places like plot labels."""
    return base + index_of(label).translate(_SUBSCRIPT_DIGITS)


def swatch(color: str) -> str:
    return f'<span style="color:{color}">&#9632;</span>'
