# ShotSpot: notes for Claude Code

An educational 2D acoustic gunshot localization (multilateration) simulator that shows the work:
real numbers substituted into real equations, beside the geometry. The original requirements are in
[docs/spec.md](docs/spec.md). Read them before working on a mode.

## Commands

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m shotspot      # run the app
.venv/bin/python -m pytest        # all tests; tests/test_ui.py forces QT_QPA_PLATFORM=offscreen
```

## Status: built in stages, agreed with the user

1. Core model, solvers and tests: **done**
2. Field editing (sensors/shot), controls, presets, save/load, Overview and Per-sensor tabs: **done**
3. The six modes, one at a time, in spec order: known t₀ → t₀ sweep → TDOA hyperbolas → closed form →
   Gauss-Newton → accuracy/GDOP (+ Monte Carlo). Then the wavefront animation. **Not started.**

Pause after each mode so the user can review it, unless they say to keep going.

## Architecture

- `shotspot/core/` is pure numpy and must never import Qt or matplotlib. Each solver returns a dataclass
  holding every intermediate quantity (equations, u, v, quadratic coefficients, per-iteration
  Jacobians, …). The UI formats those numbers; it never redoes the math.
- `shotspot/ui/`: `MainWindow` owns the `Scenario`. Every edit calls `request_refresh()` (a coalesced
  0 ms timer), and `refresh()` runs: `Snapshot.take` → `mode.update` → `field.render` → `panel.set_pages`.
- New modes subclass `ui/modes/base.py::Mode` (`update`, `draw`, `method_html`; optionally `controls`,
  `handles`/`move_handle` for draggable points, and `step_count`/`set_step` for Prev/Next). Register them in
  the toolbar's mode combo, which currently lists only "Field". Mode widgets go in `MainWindow.mode_controls`.
- `FieldCanvas` has a single axes. The sweep mode will need a residual-vs-t₀ strip below the field, and
  GDOP a colorbar.
- Colors are in `ui/style.py`. A sensor's color follows its label number, using the validated categorical
  palette in fixed slot order; past 8 sensors it falls back to gray. Text uses ink colors, never series colors.
- Light and dark themes: the window picks `LIGHT` or `DARK` from the system palette (`theme_for`), switches
  live on a `PaletteChange`, and passes the theme in `snap.theme`. Modes draw only with `snap.theme` /
  `snap.colors`: no hardcoded hex, and no matplotlib default colors, which are black and vanish on the dark field.
  The panel's unstyled text takes the page's palette Text color (set to `ink`), because Qt ignores `color` on `body`.
- Panel HTML (QTextBrowser):
  - Avoid combining characters: Qt misplaces the hat in p̂, so write `p<sub>est</sub>`.
  - Parenthesize negatives inside expressions with `fmt.term()` / `fmt.wrap()`.
  - Table cells are `nowrap`.
  - Sensor anchors are `sensor-S3`, and links are `sensor:S3`.

## Numerical lessons (already handled; don't regress)

- **Closed form roots:** computed in τ = t₀ − t_ref, because the absolute-t₀ coefficients (~10⁶)
  cancel. A discriminant within roundoff of zero counts as a double root. The panel still *displays*
  the absolute-t₀ quadratic.
- **4+ sensors:** linear least squares can be rank-deficient even in good layouts (a square with the
  shot on a symmetry line). The result method is then `"constrained"`: p = u + v·t₀, then the quadratic.
- **Gauss-Newton:** the step uses `lstsq` on the column-scaled J, not the normal equations, because JᵀJ
  squares the conditioning and turns near-singular into singular. It is undamped on purpose: it diverges
  from starts well outside the array (realistic, and a teaching point). The initial t₀ doesn't affect the
  path, because the residual is linear in t₀.
- **Covariance:** computed via the SVD of the scaled J. Position error = `dilution(geometry)` × cσ.
- **Presets:**
  - `triangle_ambiguous` uses a smaller triangle so both solutions fit on the field. The shot is at
    (900, 140) and the phantom near (687.7, 296.4). GN started from the array centroid converges to the phantom.
  - `nearly_collinear` has a mirror-image twin across the sensor line.
  - In `far_field`, the t₀-sweep valley is very shallow.

## Working with the user

- They are learning from the code as well as using it. Keep it readable: module docstrings derive the
  math, and comments are moderate.
- If a system package or tool is missing (node, python3-venv, …), say so and stop so they can install
  it. Don't build a workaround.
- Check UI changes with offscreen screenshots (`QWidget.grab()`) as well as the tests.
