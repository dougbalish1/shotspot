"""Number and symbol formatting for the math panel.

Consistent precision everywhere: positions and ranges to the centimeter,
absolute times to the microsecond, time differences in milliseconds.
Negative numbers use a real minus sign (−), and when substituted into an
equation they are wrapped in parentheses so "x − (−3.20)" reads correctly.
Numbers of a million or more (squared distances, determinants) switch to
scientific notation, because a 13-digit number can't be read at a glance.
Units are joined to their numbers with a non-breaking space, so a line never
wraps between "2.24" and "m".
"""

from __future__ import annotations

from collections.abc import Sequence

MINUS = "−"


def num(x: float, nd: int = 2) -> str:
    text = f"{x:.{nd}f}"
    if text.startswith("-"):
        text = text[1:] if float(text) == 0.0 else MINUS + text[1:]  # never show "−0.00"
    return text


def big(x: float, nd: int = 2, sig: int = 6) -> str:
    """Like num, but |x| ≥ 10⁶ is written with `sig` significant digits as 7.41603×10⁸."""
    if abs(x) < 1e6:
        return num(x, nd)
    mantissa, exponent = f"{x:.{sig - 1}e}".split("e")
    return f"{num(float(mantissa), sig - 1)}×10<sup>{int(exponent)}</sup>"


def plus(x: float, nd: int = 2) -> str:
    """A number continuing a sum, its sign becoming the operator: ' + 3.20' or ' − 3.20'."""
    text = big(x, nd)
    return f" {MINUS} {text[1:]}" if text.startswith(MINUS) else f" + {text}"


def linear(a_x: float, a_y: float, rhs: float) -> str:
    """The equation a_x·x + a_y·y = rhs, e.g. '1200.00·x − 50.00·y = 618002.84'."""
    return linear_terms([a_x, a_y], ["x", "y"], rhs)


def linear_terms(coefficients, names, rhs: float) -> str:
    """Σ coefficient·name = rhs, e.g. linear_terms([1200, 100, -4296.6], ["x", "y", "t<sub>0</sub>"], 6.0e5)."""
    first, *rest = list(zip(coefficients, names))
    terms = f"{big(first[0])}·{first[1]}" + "".join(f"{plus(a)}·{name}" for a, name in rest)
    return f"{terms} = {big(rhs)}"


def wrap(text: str) -> str:
    """Parenthesize a formatted value if it is negative, for use inside an expression."""
    return f"({text})" if text.startswith(MINUS) else text


def term(x: float, nd: int = 2) -> str:
    """A number ready to drop into an expression: negatives get parentheses."""
    return wrap(big(x, nd))


def signed(x: float, nd: int = 2) -> str:
    text = num(x, nd)
    return text if text.startswith(MINUS) or float(x) == 0.0 else "+" + text


def meters(x: float, nd: int = 2) -> str:
    return f"{num(x, nd)}&nbsp;m"


def length(x: float) -> str:
    """A length in the unit that reads best, for quantities spanning decades: 3.42 m, 9.2 cm, 4.1 mm."""
    a = abs(x)
    if a >= 1 or a == 0:
        return meters(x)
    if a >= 0.01:
        return f"{num(x * 100, 1)}&nbsp;cm"
    return f"{num(x * 1000, 1)}&nbsp;mm"


def seconds(t: float, nd: int = 6) -> str:
    return f"{num(t, nd)}&nbsp;s"


def millis(t: float, nd: int = 3) -> str:
    """A time *difference*, given in seconds, shown in milliseconds."""
    return f"{num(t * 1e3, nd)}&nbsp;ms"


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


def matrix(rows) -> str:
    """A small matrix of numbers as an HTML table, entries formatted with big()."""
    body = "".join("<tr>" + "".join(f'<td align="right" nowrap>{big(x)}</td>' for x in row) + "</tr>" for row in rows)
    return f'<table class="mat" cellspacing="0" cellpadding="3">{body}</table>'


def vector(values, nd: int = 2) -> str:
    """(a, b, c) with each entry formatted by big()."""
    return "(" + ", ".join(big(v, nd) for v in values) + ")"


def wash(html: str, color: str) -> str:
    """Put a block of HTML on a tinted background (how the selected sensor is highlighted)."""
    return f'<table width="100%" cellpadding="4" bgcolor="{color}"><tr><td>{html}</td></tr></table>'
