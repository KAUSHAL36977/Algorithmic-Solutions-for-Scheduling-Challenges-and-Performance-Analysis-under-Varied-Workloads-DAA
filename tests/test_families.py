"""Sequencing, multiprocessor, cloud, workflow and resource-allocation algorithms.

Where the problem is small enough, results are checked against brute force.
"""

import itertools
import random

import pytest

from schedlab import workloads as wl
from schedlab.algorithms import cloud, multiprocessor, resource, sequencing, workflow
from schedlab.models import DAG, Job, Process

# ---------------------------------------------------------------- job sequencing

CLASSIC_JOBS = [Job("a", 2, 100), Job("b", 1, 19), Job("c", 2, 27), Job("d", 1, 25), Job("e", 3, 15)]


def test_job_sequencing_classic():
    r = sequencing.job_sequencing(CLASSIC_JOBS)
    assert [j.id for j in r.sequence] == ["c", "a", "e"]
    assert r.total_profit == 142
    assert {j.id for j in r.rejected} == {"b", "d"}


def _best_profit_brute_force(jobs):
    best = 0
    for k in range(len(jobs) + 1):
        for subset in itertools.combinations(jobs, k):
            ordered = sorted(subset, key=lambda j: j.deadline)
            if all(j.deadline >= i + 1 for i, j in enumerate(ordered)):
                best = max(best, sum(j.profit for j in subset))
    return best


@pytest.mark.parametrize("seed", range(20))
def test_job_sequencing_is_optimal(seed):
    jobs = wl.job_set(8, seed)
    assert sequencing.job_sequencing(jobs).total_profit == _best_profit_brute_force(jobs)


def test_job_sequencing_schedule_respects_deadlines():
    r = sequencing.job_sequencing(wl.job_set(50, 1))
    for slot, job_id in r.slots.items():
        assert slot <= next(j for j in r.sequence if j.id == job_id).deadline
    assert r.to_schedule().makespan == max(r.slots)


# ---------------------------------------------------------------- multiprocessor


def _optimal_makespan(procs, m):
    best = float("inf")
    for assign in itertools.product(range(m), repeat=len(procs)):
        loads = [0.0] * m
        for p, cpu in zip(procs, assign):
            loads[cpu] += p.burst
        best = min(best, max(loads))
    return best


def test_lpt_worst_case_instance_hits_graham_bound():
    # Graham's tight example for m = 3: LPT = 11, OPT = 9, ratio 11/9 = 4/3 - 1/(3m) exactly.
    jobs = [Process(f"j{i}", 0, b) for i, b in enumerate([5, 5, 4, 4, 3, 3, 3])]
    assert multiprocessor.lpt(jobs, 3).makespan == 11
    assert _optimal_makespan(jobs, 3) == 9


@pytest.mark.parametrize("seed", range(10))
def test_approximation_guarantees(seed):
    rng = random.Random(seed)
    m = rng.choice([2, 3])
    jobs = [Process(f"j{i}", 0, rng.randint(1, 20)) for i in range(7)]
    opt = _optimal_makespan(jobs, m)
    assert multiprocessor.lpt(jobs, m).makespan <= (4 / 3 - 1 / (3 * m)) * opt + 1e-9
    assert multiprocessor.list_scheduling(jobs, m).makespan <= (2 - 1 / m) * opt + 1e-9


@pytest.mark.parametrize("fn", [multiprocessor.list_scheduling, multiprocessor.lpt,
                                multiprocessor.work_stealing])
def test_multiprocessor_invariants(fn):
    jobs = wl.as_batch(wl.heavy_tailed(40, 2))
    s = fn(jobs, 4)
    assert sorted(x.pid for x in s.slices) == sorted(p.pid for p in jobs)
    for cpu in range(4):
        mine = sorted((x for x in s.slices if x.cpu == cpu), key=lambda x: x.start)
        assert all(a.end <= b.start + 1e-9 for a, b in zip(mine, mine[1:]))
    assert s.makespan >= multiprocessor.makespan_lower_bound(jobs, 4) - 1e-9


def test_work_stealing_balances_skewed_chunks():
    # All the big jobs land in worker 0's chunk; stealing must spread them out.
    jobs = [Process(f"big{i}", 0, 20) for i in range(4)] + [Process(f"s{i}", 0, 1) for i in range(12)]
    ws = multiprocessor.work_stealing(jobs, 4, steal_cost=0.0)
    assert ws.meta["steals"] > 0
    assert ws.makespan < sum(p.burst for p in jobs[:4])


# ---------------------------------------------------------------- cloud


def test_min_min_vs_max_min_hand_example():
    etc = [[2, 2], [2, 2], [2, 2], [6, 6]]
    assert cloud.min_min(etc).makespan == 8  # the big task is stranded at the end
    assert cloud.max_min(etc).makespan == 6  # the big task goes first


def test_cloud_maps_every_task_once():
    etc = wl.etc_matrix(20, 4, seed=5)
    for s in (cloud.min_min(etc), cloud.max_min(etc)):
        assert sorted(x.pid for x in s.slices) == sorted(f"t{i}" for i in range(20))
        for x in s.slices:
            assert x.duration == pytest.approx(etc[int(x.pid[1:])][x.cpu])


def test_cloud_rejects_ragged_matrix():
    with pytest.raises(ValueError):
        cloud.min_min([[1, 2], [3]])


# ---------------------------------------------------------------- workflows


def test_heft_reproduces_topcuoglu_2002():
    dag = wl.topcuoglu_example()
    s = workflow.heft(dag)
    assert s.makespan == pytest.approx(80)
    assert s.meta["order"] == ["n1", "n3", "n4", "n2", "n5", "n6", "n9", "n7", "n8", "n10"]
    ranks = s.meta["ranks"]
    assert ranks["n1"] == pytest.approx(108)
    assert ranks["n10"] == pytest.approx(44 / 3)


def test_heft_respects_precedence_and_communication():
    dag = wl.random_dag(30, 3, seed=4, ccr=1.0)
    s = workflow.heft(dag)
    at = {x.pid: x for x in s.slices}
    for (u, v), c in dag.edges.items():
        delay = 0 if at[u].cpu == at[v].cpu else c
        assert at[v].start >= at[u].end + delay - 1e-9
    assert s.meta["slr"] >= 1 - 1e-9


def test_cpm_hand_example():
    #   A(3) → B(2) → D(4)
    #   A(3) → C(5) → D(4)        critical: A C D = 12, B has slack 3
    dag = DAG({"A": [3], "B": [2], "C": [5], "D": [4]},
              {("A", "B"): 0, ("A", "C"): 0, ("B", "D"): 0, ("C", "D"): 0})
    r = workflow.critical_path_method(dag)
    assert r.duration == 12
    assert r.critical_path == ["A", "C", "D"]
    slack = {row["task"]: row["slack"] for row in r.rows}
    assert slack == {"A": 0, "B": 3, "C": 0, "D": 0}


def test_cycle_is_detected():
    with pytest.raises(ValueError):
        DAG({"a": [1], "b": [1]}, {("a", "b"): 0, ("b", "a"): 0}).topological_order()


def test_monte_carlo_without_uncertainty_equals_cpm():
    dag = wl.topcuoglu_example()
    mc = workflow.monte_carlo_cpm(dag, spread=0.0, samples=20)
    assert mc["mean"] == pytest.approx(mc["deterministic"])
    assert mc["criticality"]["n1"] == 1.0


def test_monte_carlo_right_skew_raises_expected_duration():
    mc = workflow.monte_carlo_cpm(wl.topcuoglu_example(), spread=0.4, samples=500, seed=1)
    assert mc["p90"] > mc["p50"] > mc["deterministic"]


# ---------------------------------------------------------------- resource allocation

AVAILABLE = [3, 3, 2]
MAX_CLAIM = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
ALLOCATION = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]


def _sequence_is_valid(seq, available, max_claim, allocation):
    work = list(available)
    for i in seq:
        need = [m - a for m, a in zip(max_claim[i], allocation[i])]
        if any(n > w for n, w in zip(need, work)):
            return False
        work = [w + a for w, a in zip(work, allocation[i])]
    return sorted(seq) == list(range(len(allocation)))


def test_bankers_silberschatz_state_is_safe():
    r = resource.bankers_safety(AVAILABLE, MAX_CLAIM, ALLOCATION)
    assert r.safe
    assert _sequence_is_valid(r.sequence, AVAILABLE, MAX_CLAIM, ALLOCATION)


def test_bankers_requests():
    granted = resource.bankers_request(1, [1, 0, 2], AVAILABLE, MAX_CLAIM, ALLOCATION)
    assert granted.granted and granted.new_available == [2, 3, 0]
    after = granted.new_allocation
    must_wait = resource.bankers_request(4, [3, 3, 0], granted.new_available, MAX_CLAIM, after)
    assert not must_wait.granted and "wait" in must_wait.reason
    unsafe = resource.bankers_request(0, [0, 2, 0], granted.new_available, MAX_CLAIM, after)
    assert not unsafe.granted and "UNSAFE" in unsafe.reason
    too_much = resource.bankers_request(3, [1, 1, 1], AVAILABLE, MAX_CLAIM, ALLOCATION)
    assert not too_much.granted and "maximum" in too_much.reason


def test_bankers_detects_unsafe_state():
    r = resource.bankers_safety([0, 0], [[2, 2], [2, 2]], [[1, 1], [1, 1]])
    assert not r.safe and r.sequence == []


def test_bankers_rejects_allocation_over_claim():
    with pytest.raises(ValueError):
        resource.bankers_safety([1], [[1]], [[2]])


def test_max_min_fair_classic_example():
    assert resource.max_min_fair(10, [2, 2.6, 4, 5]) == pytest.approx([2, 2.6, 2.7, 2.7])


def test_max_min_fair_weighted_and_surplus():
    assert resource.max_min_fair(12, [10, 10], [1, 2]) == pytest.approx([4, 8])
    assert resource.max_min_fair(100, [1, 2, 3]) == pytest.approx([1, 2, 3])


@pytest.mark.parametrize("seed", range(10))
def test_max_min_fair_properties(seed):
    rng = random.Random(seed)
    demands = [rng.uniform(0, 10) for _ in range(8)]
    cap = rng.uniform(5, 60)
    alloc = resource.max_min_fair(cap, demands)
    assert sum(alloc) <= cap + 1e-9
    assert all(a <= d + 1e-9 for a, d in zip(alloc, demands))
    # Work-conserving: either everyone is satisfied or the capacity is used up.
    assert sum(alloc) == pytest.approx(min(cap, sum(demands)))
    # Nobody unsatisfied gets less than anybody else.
    unsatisfied = [a for a, d in zip(alloc, demands) if a < d - 1e-9]
    if unsatisfied:
        assert min(unsatisfied) >= max(alloc) - 1e-9
