"""Shared helpers for every Streamlit page (path setup, theme palette, table I/O)."""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "app"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from charts import PALETTES  # noqa: E402

from schedlab.metrics import METRICS  # noqa: E402
from schedlab.models import PeriodicTask, Process  # noqa: E402

PAGES = {
    "recommender": ("pages/1_Recommender.py", "Recommender", "🧭"),
    "cpu": ("pages/2_CPU_Simulator.py", "CPU Simulator", "🖥️"),
    "realtime": ("pages/3_Real_Time.py", "Real-Time", "⏱️"),
    "multi": ("pages/4_Multiprocessor_and_Cloud.py", "Multiprocessor & Cloud", "🧮"),
    "workflow": ("pages/5_Workflows.py", "Workflows", "🧩"),
    "resource": ("pages/6_Resources_and_Sequencing.py", "Resources & Sequencing", "🔐"),
    "bench": ("pages/7_Benchmark_Lab.py", "Benchmark Lab", "📊"),
    "library": ("pages/8_Algorithm_Library.py", "Algorithm Library", "📚"),
}

#: Which page runs each algorithm family.
FAMILY_PAGE = {
    "cpu": "cpu", "realtime": "realtime", "energy": "realtime", "multiprocessor": "multi",
    "cloud": "multi", "workflow": "workflow", "resource": "resource", "sequencing": "resource",
}


def page(title: str, icon: str, intro: str | None = None) -> None:
    st.set_page_config(page_title=f"{title} · Scheduling Lab", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")
    if intro:
        st.caption(intro)


def palette() -> dict:
    try:
        kind = st.context.theme.type
    except Exception:  # older Streamlit or bare mode
        kind = None
    return PALETTES["dark" if kind == "dark" else "light"]


def link(key: str, label: str | None = None) -> None:
    path, name, icon = PAGES[key]
    try:
        st.page_link(path, label=label or name, icon=icon)
    except Exception:  # page links are unavailable when a page runs on its own (tests)
        st.markdown(f"{icon} **{label or name}**")


def _num(v, default=None):
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return default
    return float(v)


def processes_frame(procs: list[Process]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"pid": p.pid, "arrival": p.arrival, "burst": p.burst, "priority": p.priority,
          "deadline": p.deadline} for p in procs],
        columns=["pid", "arrival", "burst", "priority", "deadline"],
    )


def frame_to_processes(df: pd.DataFrame) -> tuple[list[Process], list[str]]:
    """Validate an edited process table; returns (processes, error messages)."""
    procs, errors, seen = [], [], set()
    for i, row in df.reset_index(drop=True).iterrows():
        pid = str(row.get("pid") or "").strip()
        if not pid or pid == "nan":
            if any(_num(row.get(c)) is not None for c in ("arrival", "burst")):
                errors.append(f"row {i + 1}: missing pid")
            continue
        if pid in seen:
            errors.append(f"row {i + 1}: duplicate pid {pid!r}")
            continue
        try:
            procs.append(Process(pid, _num(row.get("arrival"), 0.0), _num(row.get("burst"), 0.0),
                                 int(_num(row.get("priority"), 0)), _num(row.get("deadline"))))
            seen.add(pid)
        except (TypeError, ValueError) as exc:
            errors.append(f"row {i + 1}: {exc}")
    return procs, errors


def tasks_frame(tasks: list[PeriodicTask]) -> pd.DataFrame:
    return pd.DataFrame([{"task": t.name, "C": t.C, "T": t.T, "D": t.deadline} for t in tasks],
                        columns=["task", "C", "T", "D"])


def frame_to_tasks(df: pd.DataFrame) -> tuple[list[PeriodicTask], list[str]]:
    tasks, errors, seen = [], [], set()
    for i, row in df.reset_index(drop=True).iterrows():
        name = str(row.get("task") or "").strip()
        if not name or name == "nan":
            continue
        if name in seen or "#" in name:
            errors.append(f"row {i + 1}: task names must be unique and must not contain '#'")
            continue
        try:
            tasks.append(PeriodicTask(name, _num(row.get("C"), 0.0), _num(row.get("T"), 0.0),
                                      _num(row.get("D"))))
            seen.add(name)
        except (TypeError, ValueError) as exc:
            errors.append(f"row {i + 1}: {exc}")
    return tasks, errors


SHORT_LABELS = {
    "avg_waiting": "Avg wait", "avg_turnaround": "Avg TAT", "avg_response": "Avg response",
    "p95_turnaround": "P95 TAT", "fairness": "Fairness", "context_switches": "Switches",
    "deadline_miss_ratio": "Deadline misses", "throughput": "Throughput",
    "cpu_utilization": "Utilisation", "makespan": "Makespan",
}

METRIC_COLUMNS = ["avg_waiting", "avg_turnaround", "avg_response", "p95_turnaround", "throughput",
                  "cpu_utilization", "context_switches", "fairness", "deadline_miss_ratio",
                  "makespan"]


def metrics_frame(rows: list[dict], columns: list[str] = METRIC_COLUMNS) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keep = ["algorithm", *[c for c in columns if c in df.columns]]
    return df[keep].round(3).rename(columns={c: METRICS[c][0] for c in columns if c in METRICS})


def scorecard(rows: list[dict], columns: list[str]) -> tuple[list[list[float]], list[str], list[str]]:
    """Normalise each metric to 0-1 where 1 is the best algorithm (for a heatmap)."""
    names = [r["algorithm"] for r in rows]
    z = [[0.0] * len(columns) for _ in rows]
    for j, c in enumerate(columns):
        vals = [r[c] for r in rows]
        lo, hi = min(vals), max(vals)
        lower_better = METRICS[c][1]
        for i, v in enumerate(vals):
            if hi - lo < 1e-12:
                z[i][j] = 1.0
            else:
                z[i][j] = (hi - v) / (hi - lo) if lower_better else (v - lo) / (hi - lo)
    return z, names, [SHORT_LABELS.get(c, METRICS[c][0]) for c in columns]
