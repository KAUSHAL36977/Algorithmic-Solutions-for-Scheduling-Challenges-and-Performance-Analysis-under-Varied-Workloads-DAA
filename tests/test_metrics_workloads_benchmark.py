"""Metrics, workload generators, the benchmark harness and the complexity fit."""

import math

import pytest

from schedlab import benchmark as bm
from schedlab import workloads as wl
from schedlab.algorithms import REGISTRY, cpu
from schedlab.metrics import jain_index, percentile, process_table, summarize
from schedlab.models import Process, Schedule, Slice

# ---------------------------------------------------------------- metrics


def test_jain_index_bounds():
    assert jain_index([3, 3, 3]) == pytest.approx(1.0)
    assert jain_index([1, 0, 0, 0]) == pytest.approx(0.25)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 95) == 5
    assert percentile([], 90) == 0.0


def test_summary_of_a_hand_built_schedule():
    procs = [Process("A", 0, 4, deadline=4), Process("B", 1, 2, deadline=3)]
    s = Schedule("manual", [Slice("A", 0, 4), Slice("B", 4, 6)], procs)
    m = summarize(s)
    assert m["avg_waiting"] == pytest.approx(1.5)  # A 0, B 3
    assert m["avg_turnaround"] == pytest.approx(4.5)  # A 4, B 5 (completes 6, arrived 1)
    assert m["avg_response"] == pytest.approx(1.5)
    assert m["throughput"] == pytest.approx(2 / 6)
    assert m["deadline_miss_ratio"] == 0.5
    assert m["max_lateness"] == 3
    assert m["context_switches"] == 1


def test_unfinished_process_counts_as_miss():
    procs = [Process("A", 0, 5, deadline=10)]
    s = Schedule("partial", [Slice("A", 0, 2)], procs)
    assert process_table(s)[0]["completion"] is None
    assert summarize(s)["deadline_miss_ratio"] == 1.0


# ---------------------------------------------------------------- workloads


@pytest.mark.parametrize("name", list(wl.WORKLOADS))
def test_workloads_are_reproducible_and_valid(name):
    a, b = wl.generate(name, 50, seed=7), wl.generate(name, 50, seed=7)
    assert a == b
    assert a != wl.generate(name, 50, seed=8)
    assert len(a) == 50 and len({p.pid for p in a}) == 50
    assert all(p.burst > 0 and p.arrival >= 0 for p in a)
    assert all(p.deadline >= p.arrival + p.burst - 1e-9 for p in a)


@pytest.mark.parametrize("name", ["uniform", "poisson", "bimodal", "heavy_tailed"])
def test_offered_load_is_roughly_on_target(name):
    procs = wl.generate(name, 2000, seed=0, load=0.8)
    span = max(p.arrival for p in procs)
    assert 0.55 < sum(p.burst for p in procs) / span < 1.1


def test_heavy_tail_is_heavier_than_uniform():
    def top_share(ps):
        bursts = sorted((p.burst for p in ps), reverse=True)
        return sum(bursts[: len(bursts) // 20]) / sum(bursts)

    assert top_share(wl.heavy_tailed(1000, 0)) > 2 * top_share(wl.uniform(1000, 0))


def test_batch_and_as_batch_release_everything_at_zero():
    assert all(p.arrival == 0 for p in wl.batch(20, 0))
    assert all(p.arrival == 0 for p in wl.as_batch(wl.poisson(20, 0)))


def test_uunifast_sums_to_target():
    import random

    utils = wl.uunifast(10, 0.8, random.Random(1))
    assert sum(utils) == pytest.approx(0.8)
    assert all(u >= 0 for u in utils)


def test_random_dag_is_connected_and_acyclic():
    dag = wl.random_dag(40, 3, seed=2)
    order = dag.topological_order()
    assert len(order) == 40
    pred = dag.predecessors()
    assert all(pred[t] for t in order[1:])  # every task after the first has a parent


def test_unknown_workload_raises():
    with pytest.raises(KeyError):
        wl.generate("nope", 5)


# ---------------------------------------------------------------- benchmark


def test_fit_complexity_recovers_exponents():
    ns = [100, 200, 400, 800, 1600]
    quad = bm.fit_complexity(ns, [3e-6 * n**2 for n in ns])
    lin = bm.fit_complexity(ns, [5e-3 * n for n in ns])
    nlogn = bm.fit_complexity(ns, [n * math.log(n) for n in ns])
    assert quad.exponent == pytest.approx(2.0)
    assert lin.exponent == pytest.approx(1.0)
    assert 1.0 < nlogn.exponent < 1.25
    assert quad.r2 == pytest.approx(1.0)
    assert quad.label == "≈ O(n^2.00)"


def test_fit_complexity_needs_two_sizes():
    with pytest.raises(ValueError):
        bm.fit_complexity([10, 10], [1, 2])


def test_run_benchmark_rows_and_aggregation():
    rows = bm.run_benchmark(["fcfs", "sjf"], ["poisson", "batch"], [20], [0, 1])
    assert len(rows) == 2 * 2 * 1 * 2
    assert {"algorithm", "workload", "n", "seed", "runtime_ms", "avg_waiting"} <= set(rows[0])
    agg = bm.aggregate(rows, "avg_waiting")
    assert len(agg) == 4 and all(a["runs"] == 2 for a in agg)
    board = bm.leaderboard([r for r in rows if r["workload"] == "batch"], "avg_waiting")
    assert board[0]["algorithm"] == "SJF"
    csv_text = bm.to_csv(rows)
    assert csv_text.splitlines()[0].startswith("algorithm,key,workload")
    assert len(csv_text.splitlines()) == len(rows) + 1


def test_run_benchmark_rejects_non_cpu_family():
    with pytest.raises(ValueError):
        bm.run_benchmark(["heft"], ["poisson"], [10], [0])


def test_benchmark_reports_progress():
    seen = []
    bm.run_benchmark(["fcfs"], ["uniform"], [5], [0, 1], progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 2), (2, 2)]


@pytest.mark.parametrize("key", list(REGISTRY))
def test_every_algorithm_has_an_instance_builder(key):
    args, kwargs = bm.make_instance(key, 12, seed=0)
    REGISTRY[key].call(*args, **kwargs)  # must not raise
    bm.make_instance(key, 12, seed=0, case="worst")


def test_scaling_study_rows():
    rows = bm.scaling_study(["fcfs", "cpm"], [50, 100], repeats=1)
    assert [(r["key"], r["n"]) for r in rows] == [("fcfs", 50), ("fcfs", 100), ("cpm", 50), ("cpm", 100)]
    assert all(r["runtime_ms"] > 0 for r in rows)
    assert set(bm.fit_all(rows)) == {"fcfs", "cpm"}


def test_worst_case_bankers_needs_one_pass_per_process():
    args, _ = bm.make_instance("bankers", 6, case="worst")
    from schedlab.algorithms.resource import bankers_safety

    result = bankers_safety(*args)
    assert result.safe and result.sequence == [5, 4, 3, 2, 1, 0]


def test_cpu_schedule_is_deterministic():
    procs = wl.bursty(30, 3)
    assert cpu.mlfq(procs).slices == cpu.mlfq(procs).slices
