import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import page, palette  # noqa: E402,I001

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from charts import heatmap, ranked_bars, scaling_multiples  # noqa: E402
from schedlab import benchmark as bm  # noqa: E402
from schedlab import workloads as wl  # noqa: E402
from schedlab.algorithms import FAMILIES, REGISTRY, by_family  # noqa: E402
from schedlab.metrics import METRICS  # noqa: E402

page("Benchmark Lab", "📊", "Performance analysis under varied workloads, and empirical complexity.")
P = palette()
CPU = {s.key: s.name for s in by_family("cpu")}

tab_quality, tab_scaling = st.tabs(["Schedule quality × workload", "Runtime scaling vs Big-O"])

with tab_quality:
    with st.form("bench"):
        c1, c2 = st.columns(2)
        algos = c1.multiselect("Algorithms", list(CPU), default=list(CPU), format_func=CPU.get)
        loads = c2.multiselect("Workload shapes", list(wl.WORKLOADS), default=list(wl.WORKLOADS),
                               format_func=lambda k: f"{k}: {wl.WORKLOAD_DESCRIPTIONS[k]}")
        c3, c4, c5 = st.columns(3)
        sizes = c3.multiselect("Sizes (processes)", [25, 50, 100, 200, 400], default=[50, 100])
        seeds = c4.slider("Seeds per cell", 1, 10, 3)
        cs = c5.slider("Context-switch cost", 0.0, 2.0, 0.0, 0.1)
        run = st.form_submit_button("Run benchmark", type="primary")
    if run:
        if not (algos and loads and sizes):
            st.warning("Pick at least one algorithm, workload and size.")
        else:
            bar = st.progress(0.0, text="Simulating…")
            st.session_state.bench_rows = bm.run_benchmark(
                algos, loads, sorted(sizes), list(range(seeds)), context_switch=cs,
                progress=lambda d, t: bar.progress(d / t, text=f"Simulating… {d}/{t}"),
            )
            bar.empty()

    rows = st.session_state.get("bench_rows")
    if not rows:
        st.info("Configure the experiment and press **Run benchmark**.")
    else:
        metric = st.selectbox("Metric", [m for m in METRICS if m not in ("load_imbalance",)],
                              format_func=lambda m: METRICS[m][0], key="bench_metric")
        label, lower = METRICS[metric]
        agg = bm.aggregate(rows, metric)
        algo_names = list(dict.fromkeys(r["algorithm"] for r in rows))
        load_names = list(dict.fromkeys(r["workload"] for r in rows))
        cell = {(a["algorithm"], a["workload"]): a["mean"] for a in agg}
        z = [[cell[(a, w)] for w in load_names] for a in algo_names]
        st.markdown(f"**{label}**: mean over {len({r['seed'] for r in rows})} seed(s) × "
                    f"sizes {sorted({r['n'] for r in rows})} ({'lower' if lower else 'higher'} is better)")
        st.plotly_chart(heatmap(z, load_names, algo_names, P, label, ".2f"), width="stretch",
                        key="bench_heat")

        # Rank within each workload, then average: robust to workloads with very different scales.
        ranks: dict[str, list[int]] = {a: [] for a in algo_names}
        winners = []
        for w in load_names:
            col = sorted(algo_names, key=lambda a: cell[(a, w)], reverse=not lower)
            winners.append({"workload": w, "best": col[0], label: cell[(col[0], w)],
                            "worst": col[-1]})
            for r, a in enumerate(col, 1):
                ranks[a].append(r)
        avg_rank = [{"algorithm": a, "avg rank": sum(v) / len(v)} for a, v in ranks.items()]
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Robustness: average rank across workloads** (1 = always best)")
            st.plotly_chart(ranked_bars(avg_rank, "avg rank", "Average rank", True, P), width="stretch",
                            key="bench_rank")
        with right:
            st.markdown("**Winner per workload**")
            st.dataframe(pd.DataFrame(winners), hide_index=True, width="stretch")
        st.download_button("Download raw results (CSV)", bm.to_csv(rows), "benchmark.csv", "text/csv")

with tab_scaling:
    st.markdown(
        "Times each algorithm on inputs of growing size *n* and fits **time ≈ c · nᵏ** on a log-log "
        "scale. Compare the fitted exponent *k* with the textbook bound. *Worst case* swaps in "
        "adversarial inputs: every process ready at once (HRRN's O(n) scan per decision) and a "
        "Banker's state that is only safe in reverse order."
    )
    default = ["fcfs", "sjf", "hrrn", "job_seq", "lpt", "min_min", "cpm", "heft", "bankers"]
    with st.form("scaling"):
        keys = st.multiselect(
            "Algorithms", list(REGISTRY), default=default,
            format_func=lambda k: f"{REGISTRY[k].name} ({FAMILIES[REGISTRY[k].family].lower()})",
        )
        c1, c2, c3 = st.columns(3)
        sizes = c1.multiselect("Sizes n", [125, 250, 500, 1000, 2000], default=[125, 250, 500, 1000])
        case = c2.radio("Inputs", ["typical", "worst"], horizontal=True)
        repeats = c3.slider("Repeats (min is kept)", 1, 5, 2)
        measure = st.form_submit_button("Measure", type="primary")
    if measure and keys and len(sizes) >= 2:
        bar = st.progress(0.0, text="Timing…")
        st.session_state.scaling_rows = bm.scaling_study(
            keys, sorted(sizes), repeats=repeats, case=case,
            progress=lambda d, t: bar.progress(d / t, text=f"Timing… {d}/{t}"),
        )
        bar.empty()
    elif measure:
        st.warning("Pick at least one algorithm and two sizes.")
    srows = st.session_state.get("scaling_rows")
    if srows:
        fits = bm.fit_all(srows)
        st.plotly_chart(scaling_multiples(srows, fits, P), width="stretch", key="scaling_plot")
        table = [{"algorithm": REGISTRY[k].name, "theory": REGISTRY[k].complexity,
                  "empirical exponent k": round(f.exponent, 2), "fit R²": round(f.r2, 3)}
                 for k, f in fits.items()]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("Reading the exponent: k ≈ 1 is linear or n log n (log n barely moves over one "
                   "decade), k ≈ 2 is quadratic. Typical inputs often scale better than the worst-case "
                   "bound; switch to *worst* to see the bound appear.")
        st.download_button("Download timings (CSV)", bm.to_csv(srows), "scaling.csv", "text/csv")
