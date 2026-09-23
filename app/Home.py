"""Scheduling Lab: Streamlit entry point.  Run with:  streamlit run app/Home.py"""

from common import link, page, palette  # noqa: I001  (sets up sys.path first)

import streamlit as st

from charts import ranked_bars
from schedlab.algorithms import REGISTRY, cpu
from schedlab.metrics import summarize
from schedlab.models import Process
from schedlab.recommender import KNOWLEDGE_BASE
from schedlab.workloads import WORKLOADS

page(
    "Scheduling Lab",
    "🗓️",
    "Design & Analysis of Algorithms: implement, simulate, benchmark and recommend "
    "scheduling algorithms under varied workloads.",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Runnable algorithms", len(REGISTRY))
c2.metric("Knowledge-base entries", len(KNOWLEDGE_BASE))
c3.metric("Workload shapes", len(WORKLOADS))
c4.metric("Algorithm families", len({s.family for s in REGISTRY.values()}))

st.subheader("Where to start")
left, right = st.columns(2)
with left:
    link("recommender")
    st.caption("Describe your problem in plain English; get ranked, explained picks and a "
               "simulated head-to-head on a real workload.")
    link("cpu")
    st.caption("Edit a process table, run FCFS / SJF / SRTF / RR / Priority / HRRN / MLFQ / EDF, "
               "compare Gantt charts and metrics.")
    link("realtime")
    st.caption("Liu-Layland, response-time analysis and the EDF demand test, plus RMS / DMS / EDF "
               "timelines, DVFS and sleep-state energy.")
    link("multi")
    st.caption("List scheduling, LPT and work stealing on identical cores; Min-Min vs Max-Min on "
               "heterogeneous cloud machines.")
with right:
    link("workflow")
    st.caption("Critical Path Method, Monte Carlo schedule risk and HEFT on the classic "
               "Topcuoglu et al. DAG.")
    link("resource")
    st.caption("Banker's algorithm, max-min fair sharing and greedy job sequencing with a "
               "disjoint-set union.")
    link("bench")
    st.caption("Benchmark every CPU scheduler across workload shapes, and measure empirical "
               "runtime scaling against the theoretical Big-O.")
    link("library")
    st.caption("Browse every algorithm: complexity, trade-offs, applications.")

st.subheader("Why the choice of algorithm matters")
st.markdown(
    "The textbook example (Silberschatz et al.): three processes arrive together with bursts "
    "**24, 3 and 3**. The order they run in changes the average waiting time more than five-fold."
)
procs = [Process("P1", 0, 24), Process("P2", 0, 3), Process("P3", 0, 3)]
rows = [
    {"algorithm": s.algorithm, **summarize(s)}
    for s in (cpu.fcfs(procs), cpu.sjf(procs), cpu.round_robin(procs, 4))
]
st.plotly_chart(
    ranked_bars(rows, "avg_waiting", "Average waiting time", True, palette()),
    width="stretch",
)
st.caption("SJF is provably optimal for average waiting time when all jobs are ready together; "
           "RR trades some waiting time for responsiveness; FCFS suffers the *convoy effect*.")
