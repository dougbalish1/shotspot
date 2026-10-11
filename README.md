# ShotSpot

An educational simulator for acoustic gunshot localization (multilateration) that shows the work:
real numbers substituted into real equations, next to the geometry they describe.

> **Disclaimer:** ShotSpot is an independent educational project. It is not affiliated with, endorsed
> by, or connected to SoundThinking, Inc. or its ShotSpotter product, and ShotSpotter is a trademark of
> its owner. This tool only illustrates the general textbook math behind acoustic multilateration so
> the idea is easier to picture. It does not reproduce any real product's methods, data or
> performance, it deliberately ignores real-world effects (terrain, elevation, wind, temperature,
> echoes, signal detection), and it is not meant for locating real gunfire or any other real-world use.

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
| Scroll | Zoom around the cursor; Home fits the field |
| Ctrl-drag (or middle-drag) | Pan, from anywhere: it never moves or adds anything |
| Esc | Deselect |
| Animate (A) | Watch the sound spread from the shot, slowed down; sensors flash as it reaches them |

## Modes

Pick a mode from the toolbar (or Ctrl+1, Ctrl+2, …). Each one draws its own geometry on the field and
writes its derivation, with the numbers filled in, on the Method tab. Step through it with Prev/Next.

- **Field**: just the sensors, the shot and the located position.
- **Known t₀**: pretend the firing time is known. Each arrival time becomes a radius, and the circles meet
  at the shot. The steps show the algebra (subtracting circle equations leaves straight lines), and then
  why this can't be done for real.
- **t₀ sweep**: guess t₀ with the slider under the field (or click the plot). Each guess gives circles; the
  plot shows how badly they miss each other for every guess, and drops to zero only where they meet. Scroll
  the plot to zoom it, double-click to zoom back out.
- **TDOA hyperbolas**: subtract arrival times and t₀ cancels. Each pair of sensors gives one hyperbola
  branch, and the branches cross at the shot. Toggle *All pairs* and *Asymptotes* in the control strip.
- **Closed form**: square, subtract, solve. With 3 sensors the algebra ends in a quadratic in t₀, whose two
  roots are the two-solution ambiguity; with 4+ it is a plain linear solve. The steps follow whichever
  route the solver takes.
- **Gauss-Newton**: drag the diamond to set the starting guess, then step through the iterations. From
  inside the array it converges in a few steps; from far outside, plain Gauss-Newton runs away.
- **Accuracy / GDOP**: the field is colored by the error a shot would have at each point, for the current
  timing noise. The ellipse at the shot shows the shape of the error; *Run trials* under the field fires
  hundreds of noisy shots to check the prediction.

## Layout of the code

- `shotspot/core/`: the model and solvers, plain numpy with no Qt. Each solver module opens with
  a docstring deriving its math, and returns every intermediate quantity so the UI can display it.
  Start with `closed_form.py` and `gauss_newton.py`.
- `shotspot/ui/`: the PySide6 window. `field.py` draws and handles the mouse, `pages.py` writes
  the Overview and Per-sensor HTML, and `modes/` holds the visualization modes.
- `tests/`: pytest for every solver, plus offscreen UI smoke tests.

## License

MIT: free to use, copy, modify and share, provided the copyright and license notice are kept. The
software comes with no warranty, and the authors aren't liable for anything arising from its use. See
[LICENSE](LICENSE) for the full text.
