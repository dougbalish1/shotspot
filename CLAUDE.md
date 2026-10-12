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
   Gauss-Newton → accuracy/GDOP (+ Monte Carlo). Then the wavefront animation. **Done**, awaiting the
   user's review (known t₀ was reviewed; the rest were built in one session with their OK).

Pause after each mode so the user can review it, unless they say to keep going.

## Architecture

- `shotspot/core/` is pure numpy and must never import Qt or matplotlib. Each solver returns a dataclass
  holding every intermediate quantity (equations, u, v, quadratic coefficients, per-iteration
  Jacobians, …). The UI formats those numbers; it never redoes the math.
- `shotspot/ui/`: `MainWindow` owns the `Scenario`. Every edit calls `request_refresh()` (a coalesced
  0 ms timer), and `refresh()` runs: `Snapshot.take` → `mode.update` → `field.render` → `panel.set_pages`.
  The snapshot also carries the closed form's physical roots (`snap.candidates`, `snap.ambiguous`,
  `snap.other_solution(p)`), so the Overview, the Field view and the Accuracy mode can all say "two solutions"
  instead of quietly reporting the phantom; the base `Mode` draws the other solution as a hollow square.
- New modes subclass `ui/modes/base.py::Mode` (`update`, `draw`, `method_html`; optionally `controls`,
  `handles`/`move_handle` for draggable points, and `step_count`/`set_step` for Prev/Next) and are added to
  `MODES` in `ui/modes/__init__.py`. The window keeps one instance of each (state survives switching), fills the
  mode combo (Ctrl+1…) from it, shows each mode's `controls()` widget only while it is active, and clamps
  `mode.step` after `update`. `ui/modes/known_t0.py` is the pattern: a `STEPS` list, `roadmap()` at the top of
  the Method page, one builder per step, and per-sensor rows anchored and washed via `_sensor_block`.
  Set `t0_is_given` for a mode that doesn't estimate t₀.
- The sweep's Play button and the GDOP map's throttle keep their own QTimers; Play follows a clock (`_now`,
  injectable for tests), so the sweep takes the same time whatever the frame rate.
- `FieldCanvas` has a single axes. A mode that needs another plot gives it its own canvas via `view()`
  (shown between the field and the control strip), as the sweep's residual-vs-t₀ plot does. Extra legend
  entries come from `legend_handles()`.
- `FieldCanvas.render` applies the view *before* `mode.draw`, so a mode can read `ax.get_xlim()` (the
  hyperbola contours are computed on a grid over the visible area). Zoom, pan, resize and Fit view emit
  `viewChanged`, which requests a refresh.
- Field mouse: Ctrl-drag pans (the user is on a touchpad, so middle-drag isn't enough); left-click on empty
  space adds a sensor on press, which is why panning needs the modifier.
- Colors are in `ui/style.py`. A sensor's color follows its label number, using the validated categorical
  palette in fixed slot order; past 8 sensors it falls back to gray. Text uses ink colors, never series colors.
- Light and dark themes: the window picks `LIGHT` or `DARK` from the system palette (`theme_for`), switches
  live on a `PaletteChange`, and passes the theme in `snap.theme`. Modes draw only with `snap.theme` /
  `snap.colors`: no hardcoded hex, and no matplotlib default colors, which are black and vanish on the dark field.
  The panel's unstyled text takes the page's palette Text color (set to `ink`), because Qt ignores `color` on `body`.
- The wavefront animation (`ui/animation.py`) only keeps a clock; the window passes it in `snap.clock`, the
  field then draws the wavefront instead of the mode's overlays, and `snap.reached()` hides arrival times
  the wave hasn't delivered yet.
- Panel HTML (QTextBrowser):
  - Avoid combining characters: Qt misplaces the hat in p̂, so write `p<sub>est</sub>`.
  - Parenthesize negatives inside expressions with `fmt.term()` / `fmt.wrap()`.
  - Table cells are `nowrap`.
  - Sensor anchors are `sensor-S3`, and links are `sensor:S3`. `step:N` (zero-based) jumps to a Method step.
    Give every step's per-sensor rows anchors, so selecting a sensor keeps the reader on the Method tab.
  - `fmt.big()` switches to scientific notation from 10⁶ (`term()` uses it), and units are joined with `&nbsp;`.

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
  `damped=True` (the mode's Damped checkbox) halves a step until Σr² falls; it rescues most far starts but
  from some the cost keeps falling toward the plateau at infinity, so it still "diverges". That is real,
  and the Result page says so.
- **Covariance:** computed via the SVD of the scaled J. Position error = `dilution(geometry)` × cσ.
- **Sweep speed:** `core/sweep.py` refines all guesses at once (batched Gauss-Newton with per-row step halving,
  rows dropping out as they converge or settle) and golden-section-refines each dip. ~40 ms for 400 guesses;
  the one-guess-at-a-time loop took 0.5-1.6 s. The mode still throttles recomputation during drags.
- **Sweep dips:** plateau ripples are not dips (`_dip_indices`); the lowest minimum always counts.
- **Slow modes:** the sweep and the GDOP map use `modes.base.Throttle` to redo their heavy work at most every
  200 ms during drags. The GDOP panel's canvases redraw only when their inputs change.
- **Heatmap colors:** each theme's `sequential` ramp is only the half of the blue ramp nearest its surface, so
  ink on top keeps ≥4:1 contrast. The GDOP map is drawn above the grid (zorder 0.6) and the ellipses carry a
  surface-colored halo.
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
