# ShotSpot

An educational simulator for acoustic gunshot localization (multilateration) that shows the work:
real numbers substituted into real equations, next to the geometry they describe.

## Setup

Python 3.10+.

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # numpy, matplotlib, PySide6, plus pytest
```

Developed with numpy 2.2, matplotlib 3.10, PySide6 6.12 and pytest 9.1.

## Run

```sh
.venv/bin/python -m shotspot          # the app (opens maximized; F11 for full screen)
.venv/bin/python -m pytest            # the tests (the UI tests run offscreen)
```

## Using the field

| Action | Effect |
|---|---|
| Click empty space | Add a sensor (hold and drag to place it) |
| Drag a sensor or the shot | Move it |
| Click a sensor | Select it: its geometry and equations are highlighted |
| Right-click a sensor | Delete it |
| Shift-click, or Fire mode (F) | Place the shot |
| Scroll / middle-drag | Zoom / pan; Home fits the field |
| Esc | Deselect |

## Layout of the code

- `shotspot/core/`: the model and solvers, plain numpy with no Qt. Each solver module opens with
  a docstring deriving its math, and returns every intermediate quantity so the UI can display it.
  Start with `closed_form.py` and `gauss_newton.py`.
- `shotspot/ui/`: the PySide6 window. `field.py` draws and handles the mouse, `pages.py` writes
  the Overview and Per-sensor HTML, and `modes/` holds the visualization modes.
- `tests/`: pytest for every solver, plus offscreen UI smoke tests.
