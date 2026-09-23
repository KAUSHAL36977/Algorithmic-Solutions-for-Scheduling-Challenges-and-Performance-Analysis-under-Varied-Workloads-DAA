"""Plotly figure builders for the Streamlit app.

Visual rules (see the README's "Charts" note):

* Identity is carried by axis labels (one Gantt row per process, task or CPU),
  never by colour alone, so Gantt work is a single accent hue.
* *Emphasis* over rainbow: comparisons highlight the best algorithm in the accent
  and grey the rest.
* Status colours (missed deadline, context-switch overhead) are reserved and always
  come with a legend label.
* Magnitude grids use one hue from light to dark; scaling curves use small multiples
  instead of 20+ coloured lines.
"""

from __future__ import annotations

import math
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from schedlab.models import CONTEXT_SWITCH, Schedule, task_of

PALETTES = {
    "light": {
        "accent": "#2a78d6",
        "accent_wash": "rgba(42,120,214,0.12)",
        "muted": "#c3c2b7",
        "ink_muted": "#898781",
        "surface": "#fcfcfb",
        "grid": "#e1e0d9",
        "critical": "#d03b3b",
        "good": "#0ca30c",
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "accent": "#3987e5",
        "accent_wash": "rgba(57,135,229,0.16)",
        "muted": "#52514e",
        "ink_muted": "#898781",
        "surface": "#1a1a19",
        "grid": "#2c2c2a",
        "critical": "#e66767",
        "good": "#0ca30c",
        "seq": ["#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _luminance(hex_color: str) -> float:
    """WCAG relative luminance of a #rrggbb colour."""
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def natural_key(s: str) -> list:
    """Sort "n2" before "n10"."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _layout(fig: go.Figure, p: dict, height: int, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=36 if kw.get("title") else 10, b=10),
        font=dict(family=FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font=dict(family=FONT)),
        **kw,
    )
    fig.update_xaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["muted"])
    fig.update_yaxes(gridcolor=p["grid"], zeroline=False, linecolor=p["muted"])
    return fig


def gantt(
    schedule: Schedule,
    p: dict,
    rows: str = "process",
    missed: set[str] | None = None,
    emphasis: set[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Horizontal timeline. ``rows`` is "process", "task" (group ``T#k`` jobs) or "cpu".

    ``missed``: pids to draw in the critical colour. ``emphasis``: if given, only these
    pids use the accent and the rest are greyed (e.g. the critical path).
    """
    missed = missed or set()
    order_ids: list[str] = []
    if rows == "cpu":
        order_ids = [f"CPU {c}" for c in range(schedule.cpus)]
    else:
        if rows == "task":
            order_ids = sorted({task_of(q.pid) for q in schedule.processes}, key=natural_key)
        else:
            first = schedule.first_start_times()
            order_ids = [q.pid for q in sorted(
                schedule.processes, key=lambda q: (q.arrival, first.get(q.pid, 0.0), natural_key(q.pid))
            )]

    groups: dict[str, dict[str, list]] = {}
    tiny = (schedule.makespan or 1) / 150  # slices this short would vanish under a 2px gap

    def add(kind: str, y: str, s) -> None:
        g = groups.setdefault(kind, {"y": [], "base": [], "x": [], "text": [], "custom": [], "gap": []})
        g["gap"].append(2 if s.end - s.start > tiny else 0)
        g["y"].append(y)
        g["base"].append(s.start)
        g["x"].append(s.end - s.start)
        g["text"].append(s.pid if rows == "cpu" else "")
        g["custom"].append([s.pid, s.start, s.end])

    for s in schedule.slices:
        if s.pid == CONTEXT_SWITCH:
            add("Context switch", "(switch)" if rows != "cpu" else f"CPU {s.cpu}", s)
            continue
        y = f"CPU {s.cpu}" if rows == "cpu" else (task_of(s.pid) if rows == "task" else s.pid)
        if s.pid in missed:
            kind = "Missed deadline ✕"
        elif emphasis is not None and s.pid not in emphasis:
            kind = "Other"
        else:
            kind = "Critical path" if emphasis is not None else "Running"
        add(kind, y, s)

    colors = {
        "Running": p["accent"],
        "Critical path": p["accent"],
        "Other": p["muted"],
        "Context switch": p["muted"],
        "Missed deadline ✕": p["critical"],
    }
    if "Context switch" in groups and rows != "cpu":
        order_ids.append("(switch)")
    fig = go.Figure()
    for kind in ("Running", "Critical path", "Other", "Missed deadline ✕", "Context switch"):
        if kind not in groups:
            continue
        g = groups[kind]
        fig.add_bar(
            name=kind,
            y=g["y"],
            base=g["base"],
            x=g["x"],
            orientation="h",
            marker=dict(color=colors[kind], line=dict(color=p["surface"], width=g["gap"]), cornerradius=3),
            width=0.62,
            text=g["text"],
            textposition="inside",
            insidetextanchor="middle",
            customdata=g["custom"],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.2f} → %{customdata[2]:.2f}"
            "<br>duration %{x:.2f}<extra>" + kind + "</extra>",
        )
    fig.update_layout(barmode="overlay", uniformtext=dict(minsize=9, mode="hide"),
                      showlegend=len(groups) > 1)
    fig.update_yaxes(categoryorder="array", categoryarray=order_ids, autorange="reversed",
                     showgrid=False)
    fig.update_xaxes(title="time", rangemode="tozero")
    height = max(160, 34 * len(order_ids) + 90)
    return _layout(fig, p, height, title=title)


def ranked_bars(
    rows: list[dict], metric: str, label: str, lower_is_better: bool, p: dict, name_key: str = "algorithm"
) -> go.Figure:
    """One bar per algorithm, best first; the winner in the accent, the rest grey."""
    ordered = sorted(rows, key=lambda r: r[metric], reverse=not lower_is_better)
    names = [r[name_key] for r in ordered]
    values = [r[metric] for r in ordered]
    best = values[0] if values else 0
    # Ties for first place share the accent.
    colors = [p["accent"] if abs(v - best) <= 1e-6 * max(1.0, abs(best)) else p["muted"] for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            width=0.6,
            text=[f"{v:,.2f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>" + label + ": %{x:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    top = max(values, default=0) or 1
    # Headroom so the value labels outside the bar ends are never clipped.
    fig.update_xaxes(title=f"{label} ({'lower' if lower_is_better else 'higher'} is better)",
                     range=[0, top * 1.18])
    return _layout(fig, p, max(180, 34 * len(names) + 80))


def heatmap(
    z: list[list[float]], x: list[str], y: list[str], p: dict, colorbar: str, fmt: str = ".2f",
    reverse: bool = False,
) -> go.Figure:
    """Single-hue magnitude grid, values printed in every cell (it is also the table)."""
    scale = p["seq"][::-1] if reverse else p["seq"]
    colorscale = [[i / (len(scale) - 1), c] for i, c in enumerate(scale)]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=x, y=y, colorscale=colorscale, xgap=2, ygap=2,
            colorbar=dict(title=colorbar, thickness=12),
            hovertemplate="<b>%{y}</b> · %{x}<br>" + colorbar + ": %{z:" + fmt + "}<extra></extra>",
        )
    )
    # Label each cell in black or white, whichever contrasts with its fill.
    flat = [v for row in z for v in row]
    lo, hi = (min(flat), max(flat)) if flat else (0, 1)
    for i, row in enumerate(z):
        for j, v in enumerate(row):
            t = (v - lo) / (hi - lo) if hi > lo else 0.5
            fill = scale[round(t * (len(scale) - 1))]
            fig.add_annotation(x=x[j], y=y[i], text=format(v, fmt), showarrow=False,
                               font=dict(color="#0b0b0b" if _luminance(fill) > 0.179 else "#ffffff"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return _layout(fig, p, max(220, 36 * len(y) + 90))


def scaling_multiples(rows: list[dict], fits: dict, p: dict, cols: int = 3) -> go.Figure:
    """Small multiples: one log-log panel per algorithm with its fitted power law."""
    keys = list(dict.fromkeys(r["key"] for r in rows))
    names = {r["key"]: r["algorithm"] for r in rows}
    theory = {r["key"]: r["theory"] for r in rows}
    n_rows = math.ceil(len(keys) / cols)
    titles = []
    for k in keys:
        fit = fits.get(k)
        detail = f"n^{fit.exponent:.2f} measured · {theory[k]} theory" if fit else theory[k]
        titles.append(f"<b>{names[k]}</b><br>{detail}")
    fig = make_subplots(rows=n_rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.22 / max(n_rows - 1, 1) ** 0.5
                        if n_rows > 1 else 0.2)
    for i, k in enumerate(keys):
        r, c = i // cols + 1, i % cols + 1
        pts = sorted((row["n"], row["runtime_ms"]) for row in rows if row["key"] == k)
        ns, ts = [n for n, _ in pts], [t for _, t in pts]
        fig.add_scatter(x=ns, y=ts, mode="markers", row=r, col=c, showlegend=False,
                        marker=dict(size=9, color=p["accent"], line=dict(color=p["surface"], width=2)),
                        hovertemplate="n=%{x}<br>%{y:.3f} ms<extra>" + names[k] + "</extra>")
        fit = fits.get(k)
        if fit:
            fig.add_scatter(x=ns, y=[fit.coefficient * n ** fit.exponent for n in ns], mode="lines",
                            row=r, col=c, showlegend=False, hoverinfo="skip",
                            line=dict(color=p["ink_muted"], width=2))
        # "D2" labels only the 2s and 5s within each decade, keeping log axes readable.
        fig.update_xaxes(type="log", dtick="D2", row=r, col=c,
                         title="input size n" if r == n_rows else None)
        fig.update_yaxes(type="log", dtick="D2", row=r, col=c, title="ms" if c == 1 else None)
    fig.update_annotations(font=dict(size=11))
    fig = _layout(fig, p, 290 * n_rows + 40)
    fig.update_layout(margin=dict(t=50))
    return fig


def histogram(values: list[float], markers: dict[str, float], p: dict, xlabel: str) -> go.Figure:
    fig = go.Figure(
        go.Histogram(x=values, nbinsx=40,
                     marker=dict(color=p["accent"], line=dict(color=p["surface"], width=1)),
                     hovertemplate="%{x}<br>%{y} samples<extra></extra>")
    )
    for i, (label, x) in enumerate(markers.items()):
        fig.add_vline(x=x, line=dict(color=p["ink_muted"], width=2, dash="solid" if i == 0 else "dot"))
        fig.add_annotation(x=x, y=1, yref="paper", text=f"{label} {x:.1f}", showarrow=False,
                           xanchor="left", yanchor="top", xshift=4, yshift=-16 * i)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="samples")
    return _layout(fig, p, 320, bargap=0.02)


def paired_bars(names: list[str], a: list[float], b: list[float], a_label: str, b_label: str,
                p: dict) -> go.Figure:
    """Two series: context (grey) vs result (accent), with a legend."""
    fig = go.Figure()
    fig.add_bar(x=names, y=a, name=a_label, marker=dict(color=p["muted"], cornerradius=4), width=0.35,
                offset=-0.37, hovertemplate="%{x}<br>" + a_label + ": %{y:.2f}<extra></extra>")
    fig.add_bar(x=names, y=b, name=b_label, marker=dict(color=p["accent"], cornerradius=4), width=0.35,
                offset=0.02, text=[f"{v:.2f}" for v in b], textposition="outside", cliponaxis=False,
                hovertemplate="%{x}<br>" + b_label + ": %{y:.2f}<extra></extra>")
    fig.update_xaxes(showgrid=False)
    return _layout(fig, p, 320)
