"""Visualization modes. Each one draws on the field and writes the Method page.

MODES lists them in toolbar order; the window makes one instance of each, so
a mode keeps its state (current step, slider positions) while another is shown.
"""

from .accuracy import AccuracyMode
from .base import Handle, Mode
from .closed_form import ClosedFormMode
from .gauss_newton import GaussNewtonMode
from .hyperbolas import HyperbolaMode
from .known_t0 import KnownT0Mode
from .t0_sweep import SweepMode

MODES: list[type[Mode]] = [Mode, KnownT0Mode, SweepMode, HyperbolaMode, ClosedFormMode, GaussNewtonMode, AccuracyMode]

__all__ = ["MODES", "AccuracyMode", "ClosedFormMode", "GaussNewtonMode", "Handle", "HyperbolaMode", "KnownT0Mode", "Mode", "SweepMode"]
