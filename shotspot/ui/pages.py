"""HTML for the Overview and Per-sensor tabs (the Method tab belongs to each mode)."""

from __future__ import annotations

import numpy as np

from .fmt import index_of, meters, millis, num, point, seconds, sub, swatch, term, wrap
from .modes.base import Mode
from .snapshot import Snapshot
from .style import SELECTED_WASH


def _sensor_link(snap: Snapshot, i: int) -> str:
    label = snap.labels[i]
    ref = " (ref)" if i == snap.reference else ""
    return f'<a href="sensor:{label}">{swatch(snap.colors[i])}&nbsp;{label}</a>{ref}'


# ---- Overview ---------------------------------------------------------------


def overview_html(snap: Snapshot, mode: Mode) -> str:
    sc = snap.scenario
    out = [f"<h2>{mode.title}</h2>", f"<p>{mode.summary}</p>"]

    if snap.n < 3:
        out.append(
            '<p class="note">Click the field to add sensors. At least 3 are needed, '
            "because there are three unknowns: x, y and t<sub>0</sub>.</p>"
        )
    if snap.shot is None:
        out.append('<p class="note">No shot yet. Shift-click the field (or turn on Fire mode) to fire one.</p>')

    rows = []
    if snap.shot is not None:
        rows.append(("True shot", f"p = {point(snap.shot)} m", f"t<sub>0</sub> = {seconds(sc.t0)}"))
    fix = mode.fix
    if fix is not None:
        rows.append(("Estimate", f"p<sub>est</sub> = {point(fix.position)} m", f"t<sub>0,est</sub> = {seconds(fix.t0)}"))
        if snap.shot is not None:
            error = float(np.linalg.norm(fix.position - snap.shot))
            rows.append(("Error", f"|p<sub>est</sub> − p| = {meters(error)}", f"t<sub>0,est</sub> − t<sub>0</sub> = {millis(fix.t0 - sc.t0)}"))
    elif snap.ready:
        rows.append(("Estimate", "no fix: the solver couldn't determine a position", ""))
    rows.append(("Speed of sound", f"c = {num(sc.c, 1)} m/s", ""))
    rows.append((
        "Timing noise",
        f"σ = {millis(sc.sigma, 2)}",
        f"c·σ = {meters(sc.c * sc.sigma)} of range" if sc.sigma > 0 else "exact arrival times",
    ))
    if snap.n:
        rows.append(("Sensors", f"{snap.n}, reference {snap.labels[snap.reference]}", ""))

    out.append('<table class="kv">')
    for name, a, b in rows:
        out.append(f'<tr><td class="note">{name}</td><td>{a}</td><td>{b}</td></tr>')
    out.append("</table>")
    if fix is not None:
        out.append(f'<p class="muted">Estimate from: {mode.estimate_source}.</p>')

    out.append(f"<h3>Scenario: {sc.name}</h3>")
    if sc.description:
        out.append(f"<p>{sc.description}</p>")
    return "\n".join(out)


# ---- Per-sensor -------------------------------------------------------------


def sensors_html(snap: Snapshot, mode: Mode) -> str:
    out = ["<h2>Per-sensor</h2>"]
    if snap.n == 0:
        out.append('<p class="note">No sensors yet. Click the field to add one.</p>')
        return "\n".join(out)

    has_shot = snap.shot is not None
    noisy = snap.scenario.sigma > 0
    fix = mode.fix
    ref = snap.labels[snap.reference]
    k = index_of(ref)

    # One column per sensor, one row per quantity: narrow enough for the side panel.
    rows: list[tuple[str, list[str]]] = [
        ("x (m)", [num(x, 1) for x, _ in snap.positions]),
        ("y (m)", [num(y, 1) for _, y in snap.positions]),
    ]
    if has_shot:
        dt = snap.times - snap.times[snap.reference]
        rows += [
            ("|p − s<sub>i</sub>| (m)", [num(d) for d in snap.distances]),
            ("t<sub>i</sub> (s)", [num(t, 6) for t in snap.times]),
            (f"t<sub>i</sub> − t<sub>{k}</sub> (ms)", [num(v * 1e3, 3) for v in dt]),
            (f"c·(t<sub>i</sub> − t<sub>{k}</sub>) (m)", [num(snap.c * v) for v in dt]),
        ]
        if noisy:
            rows.append(("noise (ms)", [num(v * 1e3, 3) for v in snap.times - snap.true_times]))
        if fix is not None:
            rows.append(("residual r<sub>i</sub> (m)", [num(_residual(snap, fix, i), 3) for i in range(snap.n)]))

    def cell(i: int, text: str, tag: str = "td") -> str:
        bg = f' bgcolor="{SELECTED_WASH}"' if i == snap.selected else ""
        return f"<{tag} nowrap{bg}>{text}</{tag}>"

    out.append('<table class="data" cellspacing="0"><tr><th></th>')
    out += [cell(i, _sensor_link(snap, i), "th") for i in range(snap.n)]
    out.append("</tr>")
    for name, values in rows:
        out.append(f'<tr><td nowrap class="note" style="text-align:left">{name}</td>')
        out += [cell(i, v) for i, v in enumerate(values)]
        out.append("</tr>")
    out.append("</table>")

    if not has_shot:
        out.append('<p class="note">Fire a shot (shift-click) to see arrival times and equations.</p>')
        return "\n".join(out)

    out.append(
        '<p class="muted">Click a sensor on the field, or its name above, to highlight it. '
        "Every sensor's equation is written out below with the numbers substituted.</p>"
    )
    for i in range(snap.n):
        block = _equation_block(snap, fix, i)
        if i == snap.selected:
            block = f'<table width="100%" cellpadding="4" bgcolor="{SELECTED_WASH}"><tr><td>{block}</td></tr></table>'
        out.append(block)
    return "\n".join(out)


def _residual(snap: Snapshot, fix, i: int) -> float:
    """rᵢ = |p_est − sᵢ| − c·(tᵢ − t₀_est): how far the estimate is from satisfying sensor i."""
    return float(np.linalg.norm(fix.position - snap.positions[i]) - snap.c * (snap.times[i] - fix.t0))


def _equation_block(snap: Snapshot, fix, i: int) -> str:
    sc = snap.scenario
    label = snap.labels[i]
    xi, yi = snap.positions[i]
    px, py = snap.shot
    c, ti = snap.c, snap.times[i]
    s_i, t_i = sub("s", label), sub("t", label)
    x_i, y_i = sub("x", label), sub("y", label)
    c_txt = num(c, 1)
    is_ref = i == snap.reference

    out = [f'<h3><a name="sensor-{label}"></a>{swatch(snap.colors[i])} {label}{" — the reference sensor" if is_ref else ""}</h3>']

    out.append('<p class="note">What this sensor\'s measurement says:</p>')
    out.append(f'<p class="eq">|p − {s_i}| = c·({t_i} − t<sub>0</sub>)</p>')
    out.append(f'<p class="eq">√((x − {x_i})² + (y − {y_i})²) = c·({t_i} − t<sub>0</sub>)</p>')
    out.append(f'<p class="eq">√((x − {term(xi)})² + (y − {term(yi)})²) = {c_txt} × ({num(ti, 6)} − t<sub>0</sub>)</p>')

    lhs = snap.distances[i]
    rhs = c * (ti - sc.t0)
    out.append(f'<p class="note">At the true shot p = {point(snap.shot)}, t<sub>0</sub> = {seconds(sc.t0)}:</p>')
    out.append(
        f'<p class="eq">√(({num(px)} − {term(xi)})² + ({num(py)} − {term(yi)})²) = '
        f"{c_txt} × ({num(ti, 6)} − {num(sc.t0, 6)})</p>"
    )
    out.append(f'<p class="eq">{meters(lhs)} = {meters(rhs)}</p>')
    if sc.sigma > 0:
        noise = ti - snap.true_times[i]
        out.append(
            f'<p class="note">The sides differ by {meters(abs(rhs - lhs))}: that is the timing noise, '
            f"c × {wrap(millis(noise))} = {meters(c * noise)}.</p>"
        )

    if fix is not None:
        est = float(np.linalg.norm(fix.position - snap.positions[i]))
        r = est - c * (ti - fix.t0)
        out.append(f'<p class="note">At the estimate p<sub>est</sub> = {point(fix.position)}, t<sub>0,est</sub> = {seconds(fix.t0)}:</p>')
        out.append(
            f'<p class="eq">r<sub>{index_of(label)}</sub> = |p<sub>est</sub> − {s_i}| − c·({t_i} − t<sub>0,est</sub>) = '
            f"{num(est)} − {num(c * (ti - fix.t0))} = {meters(r, 3)}</p>"
        )

    if not is_ref:
        k = snap.reference
        ref = snap.labels[k]
        tk = snap.times[k]
        dt = ti - tk
        dr = c * dt
        out.append(f'<p class="note">Relative to the reference {ref}, t<sub>0</sub> cancels:</p>')
        out.append(f'<p class="eq">|p − {s_i}| − |p − {sub("s", ref)}| = c·({t_i} − {sub("t", ref)})</p>')
        out.append(
            f'<p class="eq">= {c_txt} × ({num(ti, 6)} − {num(tk, 6)}) = {c_txt} × {wrap(millis(dt))} = {meters(dr)}</p>'
        )
        if abs(dt) > 0:
            when, side = ("after", "farther from") if dt > 0 else ("before", "closer to")
            out.append(
                f'<p class="note">{label} heard it {millis(abs(dt))} {when} {ref}, so the shot is '
                f"{meters(abs(dr))} {side} {label} than {ref}, whatever the actual distances are.</p>"
            )
    return "\n".join(out)
