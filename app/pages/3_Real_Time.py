import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import frame_to_tasks, page, palette, tasks_frame  # noqa: E402,I001

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from charts import gantt  # noqa: E402
from schedlab import workloads as wl  # noqa: E402
from schedlab.algorithms import energy, realtime  # noqa: E402
from schedlab.metrics import process_table, summarize  # noqa: E402

page("Real-Time", "⏱️", "Periodic tasks: schedulability tests, then the timeline over one hyperperiod.")
P = palette()


def verdict(ok: bool, yes: str = "schedulable", no: str = "not schedulable") -> str:
    return f"✅ {yes}" if ok else f"❌ {no}"


with st.sidebar:
    st.header("Task set")
    n = st.slider("Tasks", 2, 8, 3)
    u = st.slider("Target utilisation U", 0.3, 1.2, 0.95, 0.01)
    seed = st.number_input("Seed", 0, 10_000, 0)
    harmonic = st.toggle("Harmonic periods", value=False,
                         help="Harmonic periods (each divides the next) let RMS reach 100% utilisation.")
    constrained = st.toggle("Deadlines shorter than periods (D < T)", value=False)

key = (n, u, int(seed), harmonic, constrained)
if st.session_state.get("rt_source") != key:
    st.session_state.rt_source = key
    periods = wl.HARMONIC_PERIODS if harmonic else wl.NON_HARMONIC_PERIODS
    st.session_state.rt_table = tasks_frame(wl.periodic_taskset(n, u, int(seed), constrained, periods))

st.markdown("**Tasks**: C = worst-case execution time, T = period, D = relative deadline "
            "(defaults to T).")
edited = st.data_editor(st.session_state.rt_table, num_rows="dynamic", width="stretch",
                        key=f"rt_editor{key}")
tasks, errors = frame_to_tasks(edited)
for e in errors:
    st.error(e)
if not tasks:
    st.stop()

rep = realtime.schedulability_report(tasks)
st.subheader("Schedulability analysis")
c = st.columns(4)
c[0].metric("Utilisation U", f"{rep['utilization']:.3f}")
c[1].metric("Liu-Layland bound", f"{rep['liu_layland_bound']:.3f}",
            help="RMS is guaranteed schedulable if U ≤ n(2^(1/n) − 1). Sufficient, not necessary.")
c[2].metric("Hyperperiod", f"{rep['hyperperiod']:g}")
c[3].metric("Jobs per hyperperiod", sum(math.ceil(rep["hyperperiod"] / t.T) for t in tasks))

st.markdown(
    f"| Test | Result |\n|---|---|\n"
    f"| RMS: Liu-Layland bound (sufficient) | {verdict(rep['rms_ll_pass'], 'pass', 'inconclusive')} |\n"
    f"| RMS: hyperbolic bound Π(Uᵢ+1) ≤ 2 (sufficient) | "
    f"{verdict(rep['rms_hyperbolic_pass'], 'pass', 'inconclusive')} |\n"
    f"| RMS: response-time analysis (exact) | {verdict(rep['rms_schedulable'])} |\n"
    f"| DMS: response-time analysis (exact) | {verdict(rep['dms_schedulable'])} |\n"
    f"| EDF: processor-demand test (exact) | {verdict(rep['edf_schedulable'])} |"
)


def rt(r: float) -> str:
    return f"{r:g}" if math.isfinite(r) else "❌ exceeds D"


rta = pd.DataFrame(
    [{"task": t.name, "C": t.C, "T": t.T, "D": t.deadline,
      "RMS worst-case response": rt(rep["rms_rta"][t.name]),
      "DMS worst-case response": rt(rep["dms_rta"][t.name])}
     for t in tasks]
)
st.dataframe(rta, hide_index=True, width="stretch")

MAX_UI_HORIZON = 2000
horizon = min(rep["hyperperiod"], MAX_UI_HORIZON)
st.subheader("Simulation over one hyperperiod")
if horizon < rep["hyperperiod"]:
    st.caption(f"Hyperperiod {rep['hyperperiod']:g} is long; simulating the first {horizon:g} time units.")
sims = {"RMS": realtime.rate_monotonic(tasks, horizon), "DMS": realtime.deadline_monotonic(tasks, horizon),
        "EDF": realtime.edf_periodic(tasks, horizon)}
summary = []
for name, s in sims.items():
    m = summarize(s)
    summary.append({"algorithm": name, "jobs": m["processes"],
                    "missed": round(m["deadline_miss_ratio"] * m["processes"]),
                    "miss ratio": m["deadline_miss_ratio"], "max lateness": m["max_lateness"],
                    "preemptions": s.meta["preemptions"], "avg response": m["avg_response"]})
st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
for tab, (name, s) in zip(st.tabs(list(sims)), sims.items()):
    with tab:
        missed = {r["pid"] for r in process_table(s)
                  if r["completion"] is None or r["completion"] > r["deadline"] + 1e-9}
        if missed:
            st.error(f"❌ {len(missed)} job(s) miss their deadline under {name} (shown in red).")
        else:
            st.success(f"✅ Every job meets its deadline under {name}.")
        st.plotly_chart(gantt(s, P, rows="task", missed=missed), width="stretch", key=f"rt_{name}")

st.subheader("Energy")
e1, e2 = st.columns(2)
with e1:
    st.markdown("**Static DVFS for EDF**: run at the lowest frequency f ≥ U.")
    alpha = st.slider("Power exponent α (P ∝ fᵅ)", 2.0, 3.5, 3.0, 0.1)
    d = energy.dvfs_edf(tasks, alpha=alpha)
    if d["feasible"] and d["energy_ratio"] < 1:
        st.metric("Chosen frequency", f"{d['frequency']:.2f} × f_max",
                  delta=f"−{1 - d['energy_ratio']:.0%} dynamic energy", delta_color="inverse")
    elif d["feasible"]:
        st.metric("Chosen frequency", "1.00 × f_max")
        st.caption("No slack: U is too close to 1 to slow down. Lower the utilisation to see savings.")
    else:
        st.error("❌ U > 1: no frequency can meet every deadline.")
with e2:
    st.markdown("**Sleep states** on the EDF idle gaps: sleep when a gap exceeds the break-even time.")
    tr = st.slider("Transition energy", 0.0, 10.0, 2.0, 0.5)
    sl = energy.sleep_schedule(sims["EDF"], transition_energy=tr, horizon=horizon)
    slept = sum(g["sleep"] for g in sl["gaps"])
    st.metric("Energy saved by sleeping", f"{sl['savings_ratio']:.1%}",
              help=f"break-even gap {sl['break_even']:.2f}; slept in {slept}/{len(sl['gaps'])} idle gaps")
