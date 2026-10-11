# Original specification

The requirements this project is being built against, as originally written.

---

I want to build an educational simulator for acoustic gunshot localization (ShotSpotter-style multilateration) that "shows the work." The goal is to make the math intuitive, not to model reality. Ignore elevation (2D only), temperature/wind (c = 343 m/s, configurable), supersonic shock waves, multipath, and waveform onset detection. Arrival times are computed exactly from geometry, plus optional Gaussian timing noise. Keep the math rigorous and display real numbers substituted into real equations.

## Stack
Python 3, PySide6, matplotlib embedded via FigureCanvasQTAgg, numpy. Keep it to these dependencies unless there's a strong reason. Separate pure-numpy model/solver code (no Qt imports) from the UI, and write pytest tests for the solvers.

## Layout
Full-screen window. Left ~3/4: the field (default 1 km x 1 km, meters, light grid). Right ~1/4: a tabbed math panel (QTextBrowser with HTML is fine; Unicode math like Δ, √, subscripts is acceptable).

## Field interaction
- Left-click empty space: add a sensor (label S1, S2, ...). Drag to move. Right-click a sensor to delete it.
- Shift-click (or a "Fire" mode toggle): place the shot.
- Click a sensor to select it. Its equation and geometry are highlighted, and the math panel scrolls to it.
- Optional "animate" button: the wavefront expands from the shot in slowed-down time, sensors flash as it reaches them, and arrival times populate live.
- Controls: timing noise σ (ms) slider, speed of sound, reference sensor dropdown, re-roll noise, presets, save/load scenario as JSON.
- Presets: 3 sensors with shot inside the triangle; 3 sensors with shot outside; 3 sensors with the shot in a two-solution ambiguity region; 4 sensors in a square; far-field shot; nearly collinear sensors (degenerate).

## Visualization modes (selectable; each updates both the field and the math panel)
1. **Known t₀ (cheating mode):** pretend we know the emission time. Draw circles of radius c·(tᵢ − t₀) around each sensor. They meet at the shot. The panel explains that this is trivial, and why we can't have it in reality.
2. **Unknown t₀ sweep (key intuition mode):** a slider for guessed t₀. Circles have radius c·(tᵢ − t₀_guess). Show the residual (how badly the circles fail to meet: the best-fit point and the sum of squared mismatches) as a small plot of residual vs t₀_guess, with a marker at the current guess. It should be visually obvious that only one t₀ (or two, in ambiguous cases) makes them meet.
3. **TDOA hyperbolas:** relative to the reference sensor, compute Δtᵢ = tᵢ − t_ref and draw the hyperbola branch for each pair as the zero contour of the SIGNED function f(p) = |p − sᵢ| − |p − s_ref| − c·Δtᵢ on a grid. Signed, so only the correct branch appears. Toggle "all pairs" vs "reference pairs only." Optionally draw asymptotes to show why far-field shots give good bearing but poor range. Panel shows each pair's Δt, range difference, and the explanation that A never needs to know its own distance.
4. **Closed-form (linearize + quadratic in t₀):** square each range equation: |p|² − 2sᵢ·p + |sᵢ|² = c²(tᵢ − t₀)². Subtract sensor 1's equation from each other sensor's to cancel |p|² and get equations linear in (x, y, t₀). With exactly 3 sensors: solve the 2 linear equations for p as a linear function of t₀ (p = u + v·t₀), substitute into sensor 1's original equation to get a quadratic in t₀, and show both roots. Reject roots with t₀ > min(tᵢ) (negative range). Plot both candidate positions, and explain when ambiguity is real. With ≥4 sensors: solve the overdetermined linear system by least squares directly. Step through each algebraic stage with the numbers substituted.
5. **Iterative least squares (Gauss-Newton):** unknowns [x, y, t₀]. Residual rᵢ = |p − sᵢ| − c·(tᵢ − t₀). Show the Jacobian row per sensor (unit vector from sensor to estimate, plus the c term), the normal equations, and the update step. Next/Prev buttons step iterations. Draw the estimate's path across the field from a user-draggable initial guess, showing convergence (or divergence/wrong-minimum from bad starts).
6. **Accuracy / GDOP:** heatmap over the field of expected position error (1σ, meters) for the current sensor layout given timing noise σ, computed from the covariance (JᵀJ)⁻¹·(cσ)². Overlay the error ellipse at the current shot. A "Monte Carlo" button fires N noisy trials (e.g. 500) and scatters the estimates so the ellipse can be compared with the empirical spread. This should clearly show good accuracy inside the sensor hull, degraded accuracy outside, and radial stretching.

## Math panel tabs
- **Overview:** true shot position and t₀, the estimate from the current method, the error in meters, and a one-paragraph plain-English summary of what the current mode demonstrates.
- **Per-sensor:** a table with position, true distance, arrival time, Δt vs reference, range difference, and residual at the estimate. Plus the full equation for the selected sensor with numbers substituted.
- **Method:** step-by-step derivation for the active mode with values filled in, and Next/Prev step buttons where relevant. The current step's geometry is highlighted on the field when possible.

## Process
Plan first and show me the plan. Then build in stages: (1) model + solvers + tests verifying that every method recovers the true position at zero noise, including the two-root case; (2) the basic field with sensors, shot, and per-sensor table; (3) the modes, one at a time, in the order above. Keep the code readable, since I want to learn from it as well as use it.
