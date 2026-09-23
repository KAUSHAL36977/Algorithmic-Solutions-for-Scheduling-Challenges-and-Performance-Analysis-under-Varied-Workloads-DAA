import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402,I001
    frame_to_processes,
    metrics_frame,
    page,
    palette,
    processes_frame,
    scorecard,
)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from charts import gantt, heatmap, ranked_bars  # noqa: E402
from schedlab import workloads as wl  # noqa: E402
from schedlab.algorithms import by_family  # noqa: E402
from schedlab.metrics import METRICS, process_table, summarize  # noqa: E402

page("CPU Simulator", "🖥️", "Event-driven simulation of uniprocessor schedulers on any process table.")
P = palette()
SPECS = {s.key: s for s in by_family("cpu")}

with st.sidebar:
    st.header("Workload")
    kind = st.selectbox("Shape", list(wl.WORKLOADS), index=1,
                        format_func=lambda k: f"{k}: {wl.WORKLOAD_DESCRIPTIONS[k]}")
    n = st.slider("Processes", 3, 40, 8)
    seed = st.number_input("Seed", 0, 10_000, 0)
    st.header("Parameters")
    quantum = st.slider("Time quantum (RR, MLFQ)", 0.5, 20.0, 4.0, 0.5)
    aging = st.slider("Aging rate (Priority)", 0.0, 1.0, 0.0, 0.05,
                      help="Effective priority improves by this much per time unit spent waiting.")
    cs = st.slider("Context-switch cost", 0.0, 2.0, 0.0, 0.1,
                   help="Dispatcher overhead added whenever the CPU switches process.")

source_key = (kind, n, int(seed))
if st.session_state.get("cpu_source") != source_key:
    st.session_state.cpu_source = source_key
    st.session_state.cpu_table = processes_frame(wl.generate(kind, n, int(seed)))

st.markdown("**Process table**: edit cells, add or delete rows. Lower priority number = more important.")
edited = st.data_editor(
    st.session_state.cpu_table,
    num_rows="dynamic",
    width="stretch",
    key=f"editor{source_key}",
    column_config={
        "pid": st.column_config.TextColumn("pid", required=True),
        "arrival": st.column_config.NumberColumn("arrival", min_value=0.0, format="%.1f"),
        "burst": st.column_config.NumberColumn("burst", min_value=0.1, format="%.1f"),
        "priority": st.column_config.NumberColumn("priority", step=1),
        "deadline": st.column_config.NumberColumn("deadline (abs.)", format="%.1f"),
    },
)
processes, errors = frame_to_processes(edited)
for e in errors:
    st.error(e)
if not processes:
    st.warning("Add at least one process with a positive burst.")
    st.stop()

chosen = st.multiselect(
    "Algorithms", list(SPECS), default=["fcfs", "sjf", "srtf", "rr", "mlfq"],
    format_func=lambda k: SPECS[k].name,
)
if not chosen:
    st.stop()

schedules = {}
for key in chosen:
    spec = SPECS[key]
    schedules[key] = spec.call(processes, quantum=quantum, aging=aging, levels=3, context_switch=cs)
rows = [{"algorithm": s.algorithm, "key": k, **summarize(s)} for k, s in schedules.items()]

st.subheader("Comparison")
metric = st.selectbox("Rank by", [m for m in METRICS if m not in ("load_imbalance",)],
                      format_func=lambda m: METRICS[m][0])
label, lower = METRICS[metric]
st.plotly_chart(ranked_bars(rows, metric, label, lower, P), width="stretch")
st.markdown("**Scorecard**: each metric rescaled so the best algorithm scores 1 and the worst 0.")
score_cols = ["avg_waiting", "avg_response", "p95_turnaround", "fairness", "context_switches",
              "deadline_miss_ratio"]
z, names, labels = scorecard(rows, score_cols)
st.plotly_chart(heatmap(z, labels, names, P, "score", ".2f"), width="stretch")
with st.expander("All metrics (table)"):
    st.dataframe(metrics_frame(rows), hide_index=True, width="stretch")

st.subheader("Timelines")
tabs = st.tabs([schedules[k].algorithm for k in chosen])
for tab, key in zip(tabs, chosen):
    with tab:
        s = schedules[key]
        missed = {r["pid"] for r in process_table(s)
                  if r["deadline"] is not None and r["completion"] is not None
                  and r["completion"] > r["deadline"] + 1e-9}
        st.plotly_chart(gantt(s, P, rows="process", missed=missed), width="stretch", key=f"gantt_{key}")
        m = summarize(s)
        c = st.columns(4)
        c[0].metric("Avg waiting", f"{m['avg_waiting']:.2f}")
        c[1].metric("Avg response", f"{m['avg_response']:.2f}")
        c[2].metric("Context switches", m["context_switches"])
        c[3].metric("Missed deadlines", f"{len(missed)} ✕" if missed else "0 ✓")
        with st.expander("Per-process metrics"):
            st.dataframe(pd.DataFrame(process_table(s)), hide_index=True, width="stretch")
