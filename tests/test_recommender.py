"""Recommender: parsing, scoring, explanations, data-backed evaluation, legacy advisor."""

import importlib

import pytest

from schedlab.algorithms import REGISTRY
from schedlab.recommender import (
    BY_NAME,
    KNOWLEDGE_BASE,
    GeneralAlgorithmAdvisor,
    ProblemProfile,
    evaluate_on_workload,
    parse_description,
    recommend,
)
from schedlab.recommender.engine import DOMAINS, OBJECTIVES, score_algorithm


def top(text, k=3):
    return [r.name for r in recommend(text, top_k=k)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hard real-time periodic control loops with deadlines", {"Rate Monotonic Scheduling (RMS)",
                                                                   "Earliest Deadline First (EDF)",
                                                                   "Deadline Monotonic Scheduling (DMS)"}),
        ("batch jobs with known runtimes, minimize average waiting time", {"Shortest Job First (SJF)"}),
        ("map tasks to heterogeneous cloud VMs, minimise makespan", {"Min-Min Algorithm",
                                                                      "Max-Min Algorithm"}),
        ("interactive desktop, run times unknown, good response time", {"Round Robin (RR)",
                                                                        "Multi-Level Feedback Queue (MLFQ)"}),
        ("avoid deadlock among processes sharing several resource types", {"Banker's Algorithm"}),
        ("maximise profit; unit-time jobs each with a deadline", {"Greedy Job Sequencing"}),
        ("project with task dependencies and uncertain durations, need P90 date",
         {"Monte Carlo Scheduling"}),
        ("workflow DAG on heterogeneous GPUs and CPUs", {"Heterogeneous Earliest Finish Time (HEFT)"}),
        ("battery powered device with periodic tasks, save energy", {"DVFS-based Scheduling"}),
        ("slurm HPC cluster, improve utilization", {"Backfilling"}),
        ("parallel loop on 8 cores, uneven tasks", {"List Scheduling (Graham / LPT)", "Work Stealing"}),
    ],
)
def test_prompts_map_to_expected_algorithms(text, expected):
    assert expected & set(top(text)), top(text)


def test_processor_is_not_process():
    """Regression: the original substring matcher treated "processor" as "process"."""
    profile = parse_description("we have 4 processors and identical machines")
    assert "cpu" not in profile.domains
    assert "multiprocessor" in profile.domains
    assert profile.processors == 4


def test_parse_extracts_features():
    p = parse_description("Non-preemptive jobs with known burst times, priorities and deadlines")
    assert p.bursts_known is True
    assert p.preemptive is False
    assert {"priorities", "deadlines"} <= p.flags
    q = parse_description("CPU tasks whose run times are unknown; preemption is fine")
    assert q.bursts_known is False and q.preemptive is True


def test_objectives_are_ordered_by_mention():
    p = parse_description("minimise response time first, and then fairness")
    assert p.objectives[:2] == ["response", "fairness"]


def test_unrecognised_text_gives_no_recommendation():
    assert recommend("sort some numbers") == []


def test_unknown_burst_times_penalise_sjf():
    sjf = BY_NAME["Shortest Job First (SJF)"]

    def score(**kw):
        return score_algorithm(sjf, ProblemProfile(domains=["cpu"], objectives=["avg_wait"], **kw))

    known, unstated, unknown = score(bursts_known=True), score(), score(bursts_known=False)
    assert known.score > unstated.score > unknown.score
    assert ("-", "needs burst / run times known in advance") in unknown.reasons


def test_non_preemptive_constraint_penalises_preemptive_algorithms():
    rr = BY_NAME["Round Robin (RR)"]
    free = score_algorithm(rr, ProblemProfile(domains=["cpu"]))
    fixed = score_algorithm(rr, ProblemProfile(domains=["cpu"], preemptive=False))
    assert fixed.score < free.score


def test_reasons_and_match_are_reported():
    recs = recommend("hard real-time periodic tasks with deadlines")
    assert all(0 <= r.match <= 1 for r in recs)
    assert all(r.reasons and all(sign in "+-" for sign, _ in r.reasons) for r in recs)
    assert recs == sorted(recs, key=lambda r: -r.score)


def test_knowledge_base_is_consistent_with_registry():
    for spec in REGISTRY.values():
        assert spec.kb in BY_NAME, f"{spec.key} points at a missing KB entry {spec.kb!r}"
    for info in KNOWLEDGE_BASE:
        assert set(info.traits["domains"]) <= set(DOMAINS), info.name
        assert set(info.traits["objectives"]) <= set(OBJECTIVES), info.name
        assert info.as_text()
    assert len(KNOWLEDGE_BASE) >= 20


def test_evaluate_on_workload_cpu():
    profile = parse_description("batch jobs with known runtimes, minimize average waiting time")
    ev = evaluate_on_workload(recommend(profile), profile, n=30)
    assert ev.family == "cpu" and ev.metric == "avg_waiting"
    assert "FCFS" in {r["algorithm"] for r in ev.rows}  # baseline always included
    assert ev.winner["algorithm"] in {"SJF", "SRTF"}
    values = [r["value"] for r in ev.rows]
    assert values == sorted(values)


def test_evaluate_on_workload_realtime_prefers_edf():
    profile = parse_description("hard real-time periodic control loops with deadlines")
    ev = evaluate_on_workload(recommend(profile), profile)
    assert ev.family == "realtime"
    assert ev.winner["algorithm"].startswith("EDF")


def test_evaluate_stays_in_top_family():
    profile = parse_description("avoid deadlock when processes request several resource types")
    assert evaluate_on_workload(recommend(profile), profile) is None


def test_evaluate_multiprocessor_uses_processor_count():
    profile = parse_description("parallel loop on 8 cores with very uneven task sizes")
    ev = evaluate_on_workload(recommend(profile), profile)
    assert ev.family == "multiprocessor"
    assert ev.winner["algorithm"] == "LPT"


# ---------------------------------------------------------------- legacy general advisor


def test_general_advisor_topics():
    adv = GeneralAlgorithmAdvisor()
    names = [a.name for _, a in adv.recommend("shortest path in a graph with negative weights")]
    assert "Bellman-Ford Algorithm" in names
    assert len(names) == len(set(names))  # deduplicated across topics
    assert adv.recommend("xyzzy") == []
    assert "shortest path" in adv.follow_up_question("graph problem")


def test_general_advisor_word_boundaries():
    adv = GeneralAlgorithmAdvisor()
    assert "sorting" in adv.match_topics("I need to sort records")
    assert "primes" not in adv.match_topics("a supreme court case")


def test_general_advisor_interactive_flow():
    answers = iter(["optimization", "the knapsack problem"])
    out: list[str] = []
    recs = GeneralAlgorithmAdvisor().run_interactive(lambda _: next(answers), out.append)
    assert any("Knapsack" in a.name for _, a in recs)
    assert any("recommendations" in line for line in out)


def test_general_module_does_not_prompt_on_import(monkeypatch):
    def boom(*_):
        raise AssertionError("input() called at import time")

    monkeypatch.setattr("builtins.input", boom)
    import schedlab.recommender.general as general

    importlib.reload(general)
