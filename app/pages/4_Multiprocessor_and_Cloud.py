import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import page, palette  # noqa: E402,I001

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from charts import gantt, ranked_bars  # noqa: E402
from schedlab import workloads as wl  # noqa: E402
from schedlab.algorithms import cloud, multiprocessor  # noqa: E402
from schedlab.metrics import summarize  # noqa: E402

page("Multiprocessor & Cloud", "🧮", "Packing independent jobs onto many machines: minimise the makespan.")
P = palette()

tab_multi, tab_cloud = st.tabs(["Identical processors", "Heterogeneous machines (ETC)"])

with tab_multi:
    c = st.columns(4)
    kind = c[0].selectbox("Job sizes", ["heavy_tailed", "bimodal", "uniform", "poisson"],
                          format_func=lambda k: f"{k}: {wl.WORKLOAD_DESCRIPTIONS[k]}")
    n = c[1].slider("Jobs", 4, 80, 24)
    m = c[2].slider("Processors m", 2, 16, 4)
    seed = c[3].number_input("Seed", 0, 10_000, 0, key="mp_seed")
    steal = st.slider("Work-stealing cost per steal", 0.0, 5.0, 0.5, 0.1)
    jobs = wl.as_batch(wl.generate(kind, n, int(seed)))
    lb = multiprocessor.makespan_lower_bound(jobs, m)
    schedules = [
        multiprocessor.list_scheduling(jobs, m),
        multiprocessor.lpt(jobs, m),
        multiprocessor.work_stealing(jobs, m, steal, int(seed)),
    ]
    rows = [{"algorithm": s.algorithm, **summarize(s), "ratio": s.makespan / lb} for s in schedules]
    st.markdown(
        f"All {n} jobs are ready at t = 0 (the classic P || Cmax problem, which is NP-hard). "
        f"Lower bound: max(Σp/m, max p) = **{lb:.1f}**. Graham's list scheduling is a "
        f"(2 − 1/m) = {2 - 1 / m:.2f}-approximation; LPT is a "
        f"(4/3 − 1/(3m)) = {4 / 3 - 1 / (3 * m):.2f}-approximation."
    )
    left, right = st.columns([2, 3])
    left.plotly_chart(ranked_bars(rows, "makespan", "Makespan", True, P), width="stretch", key="mp_bars")
    right.dataframe(
        pd.DataFrame(rows)[["algorithm", "makespan", "ratio", "load_imbalance", "cpu_utilization"]]
        .round(3)
        .rename(columns={"ratio": "makespan / lower bound", "load_imbalance": "load imbalance",
                         "cpu_utilization": "utilisation"}),
        hide_index=True, width="stretch",
    )
    ws = schedules[2]
    right.caption(f"Work stealing performed {ws.meta['steals']} steals.")
    for tab, s in zip(st.tabs([s.algorithm for s in schedules]), schedules):
        with tab:
            st.plotly_chart(gantt(s, P, rows="cpu"), width="stretch", key=f"mp_{s.algorithm}")

with tab_cloud:
    c = st.columns(5)
    tn = c[0].slider("Tasks", 3, 40, 12)
    mm = c[1].slider("Machines", 2, 8, 3)
    th = c[2].select_slider("Task heterogeneity", [10, 100, 1000], value=100)
    mh = c[3].select_slider("Machine heterogeneity", [2, 10, 100], value=10)
    cseed = c[4].number_input("Seed", 0, 10_000, 0, key="etc_seed")
    consistent = st.toggle("Consistent machines (a machine faster for one task is faster for all)")
    key = (tn, mm, th, mh, int(cseed), consistent)
    if st.session_state.get("etc_source") != key:
        st.session_state.etc_source = key
        etc = wl.etc_matrix(tn, mm, int(cseed), th, mh, consistent)
        st.session_state.etc_table = pd.DataFrame(etc, columns=[f"M{j}" for j in range(mm)],
                                                  index=[f"t{i}" for i in range(tn)])
    st.markdown("**Expected Time to Compute** (row = task, column = machine), editable:")
    etc_df = st.data_editor(st.session_state.etc_table, width="stretch", key=f"etc_editor{key}")
    etc_rows = etc_df.fillna(1.0).clip(lower=0.1).values.tolist()
    ids = [str(i) for i in etc_df.index]
    res = [cloud.min_min(etc_rows, ids), cloud.max_min(etc_rows, ids)]
    mk = {s.algorithm: s.makespan for s in res}
    better = min(mk, key=mk.get)
    other = max(mk, key=mk.get)
    if mk[better] == mk[other]:
        st.info(f"Both heuristics reach makespan {mk[better]:.1f} on this matrix.")
    else:
        st.success(f"🏆 **{better}** wins: makespan {mk[better]:.1f} vs {mk[other]:.1f} "
                   f"({1 - mk[better] / mk[other]:.0%} shorter).")
    cols = st.columns(2)
    for col, s in zip(cols, res):
        with col:
            st.metric(s.algorithm, f"makespan {s.makespan:.1f}")
            st.plotly_chart(gantt(s, P, rows="cpu"), width="stretch", key=f"etc_{s.algorithm}")
    st.caption("Min-Min finishes the many small tasks first and can strand a big one at the end; "
               "Max-Min places big tasks first. Try raising task heterogeneity.")
