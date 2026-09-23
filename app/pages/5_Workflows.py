import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import page, palette  # noqa: E402,I001

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from charts import gantt, histogram, ranked_bars  # noqa: E402
from schedlab import workloads as wl  # noqa: E402
from schedlab.algorithms import workflow  # noqa: E402

page("Workflows", "🧩",
     "Task graphs (DAGs): critical path, schedule risk, and HEFT on heterogeneous processors.")
P = palette()

with st.sidebar:
    st.header("Task graph")
    source = st.radio("Source", ["Topcuoglu et al. 2002 example", "Random DAG"])
    if source == "Random DAG":
        n = st.slider("Tasks", 4, 60, 15)
        procs = st.slider("Processors", 1, 8, 3)
        ccr = st.slider("Communication / computation ratio", 0.0, 3.0, 0.5, 0.1)
        seed = st.number_input("Seed", 0, 10_000, 0)
        dag = wl.random_dag(n, procs, int(seed), ccr=ccr)
    else:
        dag = wl.topcuoglu_example()

with st.expander(f"Graph: {len(dag.tasks)} tasks, {len(dag.edges)} edges, {dag.processors} processors"):
    costs = pd.DataFrame(dag.costs, index=[f"P{p}" for p in range(dag.processors)]).T
    edges = pd.DataFrame([{"from": u, "to": v, "comm cost": c} for (u, v), c in dag.edges.items()])
    a, b = st.columns([3, 2])
    a.markdown("Execution cost per processor")
    a.dataframe(costs, width="stretch")
    b.markdown("Edges (communication cost when on different processors)")
    b.dataframe(edges, hide_index=True, width="stretch")

tab_cpm, tab_mc, tab_heft = st.tabs(["Critical Path Method", "Monte Carlo risk", "HEFT"])

with tab_cpm:
    cpm = workflow.critical_path_method(dag)
    st.markdown(f"Durations are the mean cost across processors, with unlimited resources. "
                f"Minimum project duration: **{cpm.duration:.1f}**. "
                f"Critical path: **{' → '.join(cpm.critical_path)}**")
    crit = {r["task"] for r in cpm.rows if r["critical"]}
    st.plotly_chart(gantt(cpm.to_schedule(), P, rows="process", emphasis=crit), width="stretch",
                    key="cpm_gantt")
    st.dataframe(pd.DataFrame(cpm.rows).round(2), hide_index=True, width="stretch")
    st.caption("ES/EF = earliest start/finish (forward pass), LS/LF = latest start/finish (backward "
               "pass). A task with slack can slip by that much without delaying the project.")

with tab_mc:
    c1, c2, c3 = st.columns(3)
    spread = c1.slider("Duration uncertainty (±)", 0.0, 1.0, 0.3, 0.05,
                       help="Each task takes triangular(d·(1−s), d, d·(1+2s)): overruns are likelier.")
    samples = c2.select_slider("Samples", [200, 500, 1000, 2000, 5000], value=1000)
    mseed = c3.number_input("Seed", 0, 10_000, 0, key="mc_seed")
    mc = workflow.monte_carlo_cpm(dag, spread=spread, samples=samples, seed=int(mseed))
    k = st.columns(4)
    k[0].metric("Deterministic CPM", f"{mc['deterministic']:.1f}")
    k[1].metric("Mean", f"{mc['mean']:.1f}", delta=f"+{mc['mean'] - mc['deterministic']:.1f}",
                delta_color="inverse")
    k[2].metric("P50", f"{mc['p50']:.1f}")
    k[3].metric("P90 (commit to this)", f"{mc['p90']:.1f}")
    st.plotly_chart(
        histogram(mc["samples"], {"CPM": mc["deterministic"], "P50": mc["p50"], "P90": mc["p90"]},
                  P, "project completion time"),
        width="stretch", key="mc_hist",
    )
    st.markdown("**Criticality index**: how often each task is on the critical path. The "
                "deterministic plan says a task is critical or it isn't; under uncertainty that "
                "becomes a probability.")
    crit_rows = [{"task": t, "criticality": v} for t, v in mc["criticality"].items() if v > 0]
    st.plotly_chart(ranked_bars(crit_rows, "criticality", "Criticality index", False, P, name_key="task"),
                    width="stretch", key="mc_crit")

with tab_heft:
    if dag.processors < 1:
        st.stop()
    h = workflow.heft(dag)
    k = st.columns(4)
    k[0].metric("Makespan", f"{h.makespan:.1f}")
    k[1].metric("SLR", f"{h.meta['slr']:.2f}", help="Makespan ÷ critical path on the fastest processors "
                "(≥ 1; lower is better)")
    k[2].metric("Speed-up", f"{h.meta['speedup']:.2f}×", help="Best single-processor time ÷ makespan")
    k[3].metric("Processors used", len({s.cpu for s in h.slices}))
    st.plotly_chart(gantt(h, P, rows="cpu"), width="stretch", key="heft_gantt")
    ranks = pd.DataFrame(
        [{"order": i + 1, "task": t, "upward rank": round(h.meta["ranks"][t], 2),
          "processor": f"P{h.meta['processor_of'][t]}"} for i, t in enumerate(h.meta["order"])]
    )
    st.dataframe(ranks, hide_index=True, width="stretch")
    if source.startswith("Topcuoglu"):
        st.caption("Matches the published result: priority order n1, n3, n4, n2, n5, n6, n9, n7, n8, "
                   "n10 and makespan 80.")
