import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import page, palette  # noqa: E402,I001

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from charts import gantt, paired_bars  # noqa: E402
from schedlab.algorithms import resource, sequencing  # noqa: E402
from schedlab.models import Job  # noqa: E402

page("Resources & Sequencing", "🔐", "Deadlock avoidance, fair sharing and profit-maximising job sequencing.")
P = palette()

tab_bank, tab_fair, tab_seq = st.tabs(["Banker's algorithm", "Max-min fair share", "Job sequencing"])

with tab_bank:
    st.markdown("Silberschatz's classic state: 5 processes, resource types A, B and C. Edit freely.")
    res = ["A", "B", "C"]
    avail_df = st.data_editor(pd.DataFrame([[3, 3, 2]], columns=res, index=["available"]),
                              width="stretch", key="bank_avail")
    c1, c2 = st.columns(2)
    c1.markdown("**Maximum claim**")
    max_df = c1.data_editor(
        pd.DataFrame([[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]], columns=res,
                     index=[f"P{i}" for i in range(5)]), width="stretch", key="bank_max")
    c2.markdown("**Currently allocated**")
    alloc_df = c2.data_editor(
        pd.DataFrame([[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]], columns=res,
                     index=[f"P{i}" for i in range(5)]), width="stretch", key="bank_alloc")
    available = avail_df.fillna(0).values.tolist()[0]
    max_claim = max_df.fillna(0).values.tolist()
    allocation = alloc_df.fillna(0).values.tolist()
    try:
        safety = resource.bankers_safety(available, max_claim, allocation)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if safety.safe:
        st.success("✅ SAFE state. Safe sequence: "
                   + " → ".join(f"P{i}" for i in safety.sequence))
    else:
        stuck = sorted(set(range(len(allocation))) - set(safety.sequence))
        st.error("❌ UNSAFE state: " + ", ".join(f"P{i}" for i in stuck) + " can never be guaranteed "
                 "to finish.")
    st.dataframe(pd.DataFrame([{"step": i + 1, "process": f"P{t['process']}", "need": t["need"],
                                "work after release": t["work_after"]}
                               for i, t in enumerate(safety.trace)]), hide_index=True, width="stretch")

    st.markdown("**Try a request**")
    r1, r2 = st.columns([1, 3])
    pid = r1.selectbox("Process", range(len(allocation)), format_func=lambda i: f"P{i}")
    req_df = r2.data_editor(pd.DataFrame([[1, 0, 2]], columns=res, index=["request"]),
                            width="stretch", key="bank_req")
    outcome = resource.bankers_request(pid, req_df.fillna(0).values.tolist()[0], available,
                                       max_claim, allocation)
    (st.success if outcome.granted else st.warning)(
        ("✅ " if outcome.granted else "⛔ ") + outcome.reason
    )

with tab_fair:
    c1, c2 = st.columns([1, 3])
    capacity = c1.number_input("Capacity", 0.0, 1e6, 10.0)
    demands_txt = c2.text_input("Demands (comma-separated)", "2, 2.6, 4, 5")
    weights_txt = c2.text_input("Weights (optional, comma-separated)", "")
    try:
        demands = [float(x) for x in demands_txt.split(",") if x.strip()]
        weights = [float(x) for x in weights_txt.split(",") if x.strip()] or None
        alloc = resource.max_min_fair(capacity, demands, weights)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
    else:
        names = [f"user {i + 1}" for i in range(len(demands))]
        st.plotly_chart(paired_bars(names, demands, alloc, "demand", "allocation", P), width="stretch",
                        key="fair_bars")
        st.caption(f"Allocated {sum(alloc):.2f} of {capacity:.2f}. Users asking for less than the fair "
                   "level get all they ask for; the rest split the remainder (water-filling).")

with tab_seq:
    st.markdown("Unit-time jobs with a deadline (latest slot) and a profit. Greedy by profit, placing "
                "each job in the latest free slot, which a disjoint-set union finds in near O(1).")
    jobs_df = st.data_editor(
        pd.DataFrame([["a", 2, 100], ["b", 1, 19], ["c", 2, 27], ["d", 1, 25], ["e", 3, 15]],
                     columns=["id", "deadline", "profit"]),
        num_rows="dynamic", hide_index=True, width="stretch", key="jobs",
    )
    jobs = [Job(str(r.id), int(r.deadline), float(r.profit)) for r in jobs_df.dropna().itertuples()]
    try:
        result = sequencing.job_sequencing(jobs)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.success(f"Sequence **{' → '.join(j.id for j in result.sequence)}**, total profit "
                   f"**{result.total_profit:g}**. Rejected: "
                   + (", ".join(j.id for j in result.rejected) or "none"))
        if result.sequence:
            st.plotly_chart(gantt(result.to_schedule(), P, rows="process"), width="stretch",
                            key="seq_gantt")
