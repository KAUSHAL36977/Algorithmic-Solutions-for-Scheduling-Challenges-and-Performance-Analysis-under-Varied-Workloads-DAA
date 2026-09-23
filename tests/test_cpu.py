"""Uniprocessor schedulers: textbook answers plus invariants on random workloads."""

import pytest

from schedlab import workloads as wl
from schedlab.algorithms import by_family, cpu
from schedlab.metrics import process_table, summarize
from schedlab.models import CONTEXT_SWITCH, Process

CPU_SPECS = by_family("cpu")

# Silberschatz, Galvin & Gagne, "Operating System Concepts", ch. 5.
CONVOY = [Process("P1", 0, 24), Process("P2", 0, 3), Process("P3", 0, 3)]
SRTF_EXAMPLE = [Process("P1", 0, 8), Process("P2", 1, 4), Process("P3", 2, 9), Process("P4", 3, 5)]
PRIORITY_EXAMPLE = [
    Process("P1", 0, 10, 3), Process("P2", 0, 1, 1), Process("P3", 0, 2, 4),
    Process("P4", 0, 1, 5), Process("P5", 0, 5, 2),
]


def avg_wait(schedule):
    return summarize(schedule)["avg_waiting"]


@pytest.mark.parametrize(
    ("fn", "procs", "expected"),
    [
        (cpu.fcfs, CONVOY, 17.0),
        (cpu.sjf, CONVOY, 3.0),
        (lambda p: cpu.round_robin(p, quantum=4), CONVOY, 17 / 3),
        (cpu.srtf, SRTF_EXAMPLE, 6.5),
        (cpu.priority, PRIORITY_EXAMPLE, 8.2),
    ],
    ids=["fcfs", "sjf", "rr-q4", "srtf", "priority"],
)
def test_textbook_average_waiting_times(fn, procs, expected):
    assert avg_wait(fn(procs)) == pytest.approx(expected)


def test_round_robin_timeline_matches_textbook():
    slices = [(s.pid, s.start, s.end) for s in cpu.round_robin(CONVOY, 4).slices]
    assert slices == [("P1", 0, 4), ("P2", 4, 7), ("P3", 7, 10), ("P1", 10, 30)]


def test_srtf_preempts_on_shorter_arrival():
    first = cpu.srtf(SRTF_EXAMPLE).slices[:2]
    assert (first[0].pid, first[0].end) == ("P1", 1)
    assert first[1].pid == "P2"


def test_cpu_idles_until_next_arrival():
    s = cpu.fcfs([Process("A", 0, 2), Process("B", 10, 3)])
    assert [(x.pid, x.start, x.end) for x in s.slices] == [("A", 0, 2), ("B", 10, 13)]
    assert summarize(s)["cpu_utilization"] == pytest.approx(5 / 13)


def test_context_switch_overhead_is_inserted_and_costs_time():
    s = cpu.round_robin(CONVOY, 4, context_switch=1.0)
    overhead = [x for x in s.slices if x.pid == CONTEXT_SWITCH]
    assert len(overhead) == 3
    assert s.makespan == 33
    assert avg_wait(s) > avg_wait(cpu.round_robin(CONVOY, 4))


def test_smaller_quantum_means_more_switches():
    procs = wl.poisson(30, seed=3)
    few = summarize(cpu.round_robin(procs, quantum=20))["context_switches"]
    many = summarize(cpu.round_robin(procs, quantum=1))["context_switches"]
    assert many > few


def test_priority_aging_prevents_starvation():
    # A low-priority job competes with a stream of high-priority arrivals.
    procs = [Process("low", 0, 2, priority=9)] + [
        Process(f"h{i}", i * 2, 2, priority=1) for i in range(20)
    ]
    starved = {r["pid"]: r["completion"] for r in process_table(cpu.priority(procs))}
    aged = {r["pid"]: r["completion"] for r in process_table(cpu.priority(procs, aging=1.0))}
    assert aged["low"] < starved["low"]


def test_mlfq_demotes_long_jobs_and_favours_short_ones():
    # "long" burns its level-0 quantum (0-4) and is demoted; "short" arrives at t=5 on
    # level 0 and immediately preempts it.
    procs = [Process("long", 0, 40), Process("short", 5, 2)]
    table = {r["pid"]: r for r in process_table(cpu.mlfq(procs, quantum=4))}
    assert table["short"]["response"] == pytest.approx(0.0)
    assert table["short"]["completion"] < table["long"]["completion"]


def test_hrrn_prefers_long_waiting_job_over_new_short_job():
    procs = [Process("A", 0, 10), Process("old", 1, 6), Process("new", 9, 5)]
    order = [s.pid for s in cpu.hrrn(procs).slices]
    # At t=10: old ratio (9+6)/6 = 2.5 > new ratio (1+5)/5 = 1.2
    assert order == ["A", "old", "new"]


def test_edf_meets_feasible_deadlines_that_fcfs_misses():
    procs = [Process("long", 0, 10, deadline=30), Process("urgent", 1, 2, deadline=4)]
    assert summarize(cpu.fcfs(procs))["deadline_miss_ratio"] == 0.5
    assert summarize(cpu.edf(procs))["deadline_miss_ratio"] == 0.0


def test_duplicate_pids_are_rejected():
    with pytest.raises(ValueError):
        cpu.fcfs([Process("A", 0, 1), Process("A", 1, 1)])


def test_invalid_process_is_rejected():
    with pytest.raises(ValueError):
        Process("bad", 0, 0)
    with pytest.raises(ValueError):
        cpu.round_robin(CONVOY, quantum=0)


@pytest.mark.parametrize("spec", CPU_SPECS, ids=[s.key for s in CPU_SPECS])
@pytest.mark.parametrize("workload", list(wl.WORKLOADS))
@pytest.mark.parametrize("seed", [0, 1])
def test_schedule_invariants(spec, workload, seed):
    procs = wl.generate(workload, 25, seed)
    schedule = spec.call(procs, **spec.params, context_switch=0.3)
    by_pid = {p.pid: p for p in procs}
    executed = dict.fromkeys(by_pid, 0.0)
    ordered = sorted(schedule.slices, key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:]):
        assert a.end <= b.start + 1e-9, "one CPU never runs two things at once"
    for s in schedule.work_slices():
        assert s.start >= by_pid[s.pid].arrival - 1e-9, "never runs before arrival"
        executed[s.pid] += s.duration
    for pid, p in by_pid.items():
        assert executed[pid] == pytest.approx(p.burst), "runs exactly its burst"
    rows = process_table(schedule)
    assert all(r["completion"] is not None for r in rows)
    assert all(r["waiting"] >= -1e-9 for r in rows)


@pytest.mark.parametrize("seed", range(5))
def test_sjf_is_optimal_for_batch_average_waiting(seed):
    procs = wl.batch(30, seed)
    best = avg_wait(cpu.sjf(procs))
    for fn in (cpu.fcfs, cpu.round_robin, cpu.priority, cpu.hrrn, cpu.mlfq):
        assert best <= avg_wait(fn(procs)) + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_srtf_never_worse_than_sjf_on_waiting(seed):
    procs = wl.poisson(40, seed)
    assert avg_wait(cpu.srtf(procs)) <= avg_wait(cpu.sjf(procs)) + 1e-9
