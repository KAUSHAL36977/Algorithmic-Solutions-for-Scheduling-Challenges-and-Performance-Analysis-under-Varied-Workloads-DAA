import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import FAMILY_PAGE, link, metrics_frame, page, palette  # noqa: E402,I001

import streamlit as st  # noqa: E402

from charts import ranked_bars  # noqa: E402
from schedlab.algorithms import get  # noqa: E402
from schedlab.metrics import METRICS  # noqa: E402
from schedlab.recommender import (  # noqa: E402
    DOMAINS,
    FLAGS,
    OBJECTIVES,
    GeneralAlgorithmAdvisor,
    ProblemProfile,
    evaluate_on_workload,
    parse_description,
    recommend,
)

page("Recommender", "🧭", "Explainable recommendations: every point of the score comes with a reason.")

EXAMPLES = {
    "Custom…": "",
    "Interactive OS": "Interactive desktop OS; run times are unknown; we want good response time "
                      "and no starvation.",
    "Batch queue": "Overnight batch jobs with known runtimes; minimize average waiting time.",
    "Hard real-time": "Hard real-time periodic control loops with deadlines on an embedded board.",
    "Cloud mapping": "Map tasks onto heterogeneous cloud VMs and finish everything as soon as possible.",
    "Parallel loop": "Parallel loop on 8 cores with very uneven task sizes; minimise makespan.",
    "Risky project": "Project with task dependencies and uncertain durations; need a P90 completion date.",
    "Deadlock": "Avoid deadlock when processes request several resource types.",
    "Profit": "Maximise profit: each job takes one unit of time and has a deadline.",
    "Battery device": "Battery-powered sensor node running periodic tasks; save energy.",
}

tab_sched, tab_general = st.tabs(["Scheduling problem", "General algorithm problem"])

with tab_sched:
    choice = st.segmented_control("Examples", list(EXAMPLES), default="Interactive OS")
    text = st.text_area(
        "Describe your scheduling problem",
        value=EXAMPLES.get(choice or "Custom…", ""),
        height=90,
        placeholder="What is being scheduled, constraints (deadlines, priorities, dependencies), "
                    "goal (waiting time, makespan, energy…)",
    )
    parsed = parse_description(text)
    sig = abs(hash(text))  # widget keys reset when the text changes

    with st.expander("What I understood (edit to refine)", expanded=True):
        c1, c2 = st.columns(2)
        domains = c1.multiselect(
            "Domain", list(DOMAINS), default=parsed.domains, format_func=DOMAINS.get, key=f"d{sig}"
        )
        objectives = c2.multiselect(
            "Goals, most important first", list(OBJECTIVES),
            default=parsed.effective_objectives() if parsed.domains or parsed.objectives else [],
            format_func=OBJECTIVES.get, key=f"o{sig}",
        )
        flags = c1.multiselect(
            "Problem features", list(FLAGS), default=sorted(parsed.flags), format_func=FLAGS.get,
            key=f"f{sig}",
        )
        with c2:
            k1, k2, k3 = st.columns(3)
            known = k1.radio("Run times known?", ["Not stated", "Yes", "No"],
                             index={None: 0, True: 1, False: 2}[parsed.bursts_known], key=f"k{sig}")
            pre = k2.radio("Preemption", ["Either", "Allowed", "Not allowed"],
                           index={None: 0, True: 1, False: 2}[parsed.preemptive], key=f"p{sig}")
            procs = k3.number_input("Processors", 0, 1024, parsed.processors or 0, key=f"n{sig}",
                                    help="0 = not stated")

    profile = ProblemProfile(
        domains=domains,
        objectives=objectives,
        flags=set(flags),
        bursts_known={"Not stated": None, "Yes": True, "No": False}[known],
        preemptive={"Either": None, "Allowed": True, "Not allowed": False}[pre],
        processors=procs or None,
    )
    recs = recommend(profile, top_k=5)

    if not recs:
        st.info("Nothing scheduling-related recognised yet. Mention the resources (CPU, machines, "
                "cloud, project tasks), constraints (deadlines, priorities, dependencies) and the "
                "goal, or try the **General algorithm problem** tab for sorting/graph/string questions.")
    else:
        st.subheader("Recommendations")
        for i, rec in enumerate(recs, 1):
            with st.container(border=True):
                head, meter = st.columns([3, 1])
                head.markdown(f"**{i}. {rec.name}**  \n`{rec.info.time}`")
                meter.progress(rec.match, text=f"match {rec.match:.0%}")
                pos = [w for s, w in rec.reasons if s == "+"]
                neg = [w for s, w in rec.reasons if s == "-"]
                cols = st.columns(2)
                cols[0].markdown("  \n".join(f"✅ {w}" for w in pos) or "–")
                cols[1].markdown("  \n".join(f"⚠️ {w}" for w in neg) or "No concerns for this profile")
                with st.expander("Details"):
                    st.write(rec.info.summary)
                    st.markdown(
                        f"**Space:** {rec.info.space}  \n"
                        f"**Parameters:** {', '.join(rec.info.params)}  \n"
                        f"**Advantages:** {'; '.join(rec.info.pros)}  \n"
                        f"**Limitations:** {'; '.join(rec.info.cons)}  \n"
                        f"**Applications:** {rec.info.applications}"
                    )
                    if rec.info.implemented:
                        fam = get(rec.info.registry_keys[0]).family
                        link(FAMILY_PAGE[fam], f"Try {rec.name.split(' (')[0]} in the simulator")

        st.subheader("Data-backed check")
        st.caption("Advice is a hypothesis. Run the implemented candidates (plus a baseline) on a "
                   "simulated workload and rank them by your primary goal.")
        c1, c2, c3 = st.columns([1, 1, 2])
        n = c1.slider("Workload size", 10, 200, 40, step=10)
        seed = c2.number_input("Seed", 0, 10_000, 0)
        if c3.button("Run head-to-head", type="primary"):
            ev = evaluate_on_workload(recs, profile, n=n, seed=int(seed))
            if ev is None:
                st.info("The top recommendation's family has no simulated head-to-head (e.g. Banker's "
                        "or job sequencing are exact/optimal); open its page to explore it.")
            else:
                label, lower = METRICS[ev.metric]
                st.success(f"🏆 **{ev.winner['algorithm']}** wins on *{label.lower()}* for workload "
                           f"`{ev.workload}` ({'lower' if lower else 'higher'} is better).")
                st.plotly_chart(ranked_bars(ev.rows, "value", label, lower, palette()), width="stretch")
                st.dataframe(metrics_frame(ev.rows), hide_index=True, width="stretch")

with tab_general:
    advisor = GeneralAlgorithmAdvisor()
    first = st.text_input("What kind of problem is it?",
                          placeholder="e.g. sorting a huge log file, shortest path with negative weights")
    if first:
        details = st.text_input(advisor.follow_up_question(first), key="general_details")
        results = advisor.recommend(f"{first} {details}")
        if not results:
            st.warning("I couldn't find a specific recommendation. Try describing it differently.")
        for topic, algo in results:
            with st.container(border=True):
                st.markdown(f"**{algo.name}** · `{algo.complexity}` · _{topic}_")
                st.write(algo.description)
