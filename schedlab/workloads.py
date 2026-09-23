"""Seeded, reproducible workload generators.

Each process workload targets an offered load ``ρ`` (mean burst / mean
inter-arrival time) so that different shapes can be compared fairly at the same
load. The shapes are chosen to trigger the classic behaviours:

* ``uniform``: the baseline.
* ``poisson``: M/G/1-style exponential inter-arrivals.
* ``heavy_tailed``: Pareto bursts; a few huge jobs cause FCFS convoys.
* ``bimodal``: 80% short I/O-bound bursts and 20% long CPU-bound bursts.
* ``bursty``: arrivals come in clusters, as with flash crowds or batch submissions.
* ``batch``: everything arrives at t = 0 (SJF is provably optimal here).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

from .models import DAG, Job, PeriodicTask, Process


def _finish(
    rng: random.Random, arrivals: list[float], bursts: list[float], slack: tuple[float, float]
) -> list[Process]:
    procs = []
    for i, (a, b) in enumerate(zip(arrivals, bursts)):
        b = max(round(b, 1), 0.5)
        a = round(a, 1)
        procs.append(
            Process(
                pid=f"P{i + 1}",
                arrival=a,
                burst=b,
                priority=rng.randint(1, 5),
                deadline=round(a + b * rng.uniform(*slack), 1),
            )
        )
    return procs


def _poisson_arrivals(rng: random.Random, n: int, mean_gap: float) -> list[float]:
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(1 / mean_gap)
    return out


def uniform(n: int, seed: int = 0, load: float = 0.85) -> list[Process]:
    rng = random.Random(seed)
    bursts = [rng.uniform(1, 20) for _ in range(n)]
    horizon = sum(bursts) / load
    arrivals = sorted(rng.uniform(0, horizon) for _ in range(n))
    return _finish(rng, arrivals, bursts, (1.5, 4.0))


def poisson(n: int, seed: int = 0, load: float = 0.85) -> list[Process]:
    rng = random.Random(seed)
    bursts = [rng.expovariate(1 / 8) + 0.5 for _ in range(n)]
    return _finish(rng, _poisson_arrivals(rng, n, 8.5 / load), bursts, (1.5, 4.0))


def heavy_tailed(n: int, seed: int = 0, load: float = 0.85, alpha: float = 1.5) -> list[Process]:
    rng = random.Random(seed)
    bursts = [min(rng.paretovariate(alpha) * 2, 400) for _ in range(n)]
    mean = sum(bursts) / n if n else 1
    return _finish(rng, _poisson_arrivals(rng, n, mean / load), bursts, (1.5, 4.0))


def bimodal(n: int, seed: int = 0, load: float = 0.85) -> list[Process]:
    rng = random.Random(seed)
    bursts = [rng.uniform(1, 4) if rng.random() < 0.8 else rng.uniform(20, 50) for _ in range(n)]
    mean = sum(bursts) / n if n else 1
    return _finish(rng, _poisson_arrivals(rng, n, mean / load), bursts, (1.5, 4.0))


def bursty(n: int, seed: int = 0, load: float = 0.85, cluster: int = 8) -> list[Process]:
    rng = random.Random(seed)
    bursts = [rng.uniform(1, 15) for _ in range(n)]
    horizon = sum(bursts) / load
    centers = sorted(rng.uniform(0, horizon) for _ in range(max(1, math.ceil(n / cluster))))
    arrivals = sorted(max(0.0, rng.choice(centers) + rng.uniform(0, 2)) for _ in range(n))
    return _finish(rng, arrivals, bursts, (1.5, 4.0))


def batch(n: int, seed: int = 0, load: float = 0.85) -> list[Process]:
    rng = random.Random(seed)
    procs = _finish(rng, [0.0] * n, [rng.uniform(1, 20) for _ in range(n)], (1.0, 1.0))
    # With no arrival spread, a deadline only makes sense relative to the total queue.
    total = sum(p.burst for p in procs)
    return [
        Process(p.pid, 0.0, p.burst, p.priority, round(rng.uniform(p.burst, max(p.burst, 0.7 * total)), 1))
        for p in procs
    ]


WORKLOADS: dict[str, Callable[..., list[Process]]] = {
    "uniform": uniform,
    "poisson": poisson,
    "heavy_tailed": heavy_tailed,
    "bimodal": bimodal,
    "bursty": bursty,
    "batch": batch,
}

WORKLOAD_DESCRIPTIONS = {
    "uniform": "Uniform bursts (1-20) and uniformly spread arrivals",
    "poisson": "Exponential bursts with Poisson arrivals (M/M/1-like)",
    "heavy_tailed": "Pareto (α=1.5) bursts: many tiny jobs, a few giants",
    "bimodal": "80% short I/O-bound + 20% long CPU-bound bursts",
    "bursty": "Arrivals in tight clusters (flash crowds)",
    "batch": "Everything arrives at t=0",
}


def as_batch(processes: list[Process]) -> list[Process]:
    """The same jobs, all released at t = 0 (deadlines keep their relative slack)."""
    return [
        Process(p.pid, 0.0, p.burst, p.priority, None if p.deadline is None else p.deadline - p.arrival)
        for p in processes
    ]


def generate(name: str, n: int, seed: int = 0, **kwargs) -> list[Process]:
    try:
        gen = WORKLOADS[name]
    except KeyError:
        raise KeyError(f"unknown workload {name!r}; choose from {', '.join(WORKLOADS)}") from None
    return gen(n, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Inputs for the other algorithm families
# ---------------------------------------------------------------------------

#: Periods whose LCM is 200, so hyperperiods stay short enough to simulate.
HARMONIC_PERIODS = (10, 20, 25, 40, 50, 100, 200)
#: Non-harmonic periods (LCM 120): where RMS falls short of EDF.
NON_HARMONIC_PERIODS = (6, 8, 10, 12, 15, 20)


def uunifast(n: int, total: float, rng: random.Random) -> list[float]:
    """Bini & Buttazzo's UUniFast: n utilisations summing to ``total``, uniformly."""
    utils, remaining = [], total
    for i in range(1, n):
        nxt = remaining * rng.random() ** (1 / (n - i))
        utils.append(remaining - nxt)
        remaining = nxt
    utils.append(remaining)
    return utils


def periodic_taskset(
    n: int,
    utilization: float = 0.8,
    seed: int = 0,
    constrained: bool = False,
    periods: tuple[int, ...] = HARMONIC_PERIODS,
) -> list[PeriodicTask]:
    rng = random.Random(seed)
    tasks = []
    for i, u in enumerate(uunifast(n, utilization, rng)):
        T = rng.choice(periods)
        C = max(round(u * T, 1), 0.1)
        D = round(rng.uniform(C + 0.5 * (T - C), T), 1) if constrained else None
        tasks.append(PeriodicTask(f"T{i + 1}", C, T, D))
    return tasks


def etc_matrix(
    n: int, m: int, seed: int = 0, task_het: float = 100, machine_het: float = 10,
    consistent: bool = False,
) -> list[list[float]]:
    """Range-based ETC (Ali et al., 2000). ``consistent`` makes a machine that is
    faster for one task faster for every task."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        tau = rng.uniform(1, task_het)
        row = [round(tau * rng.uniform(1, machine_het), 1) for _ in range(m)]
        rows.append(sorted(row) if consistent else row)
    return rows


def random_dag(
    n: int, processors: int = 3, seed: int = 0, max_in: int = 3, width: int = 6, ccr: float = 0.5
) -> DAG:
    """Random connected DAG with O(n) edges. Tasks are indexed in topological order.

    ``ccr`` is the communication-to-computation ratio: mean edge cost / mean task cost.
    """
    rng = random.Random(seed)
    costs: dict[str, list[float]] = {}
    for i in range(n):
        base = rng.uniform(5, 25)
        costs[f"t{i}"] = [round(base * rng.uniform(0.5, 1.5), 1) for _ in range(processors)]
    mean_cost = sum(sum(c) / len(c) for c in costs.values()) / max(n, 1)
    edges: dict[tuple[str, str], float] = {}
    for j in range(1, n):
        window = list(range(max(0, j - width), j))
        for i in rng.sample(window, rng.randint(1, min(max_in, len(window)))):
            edges[(f"t{i}", f"t{j}")] = round(ccr * mean_cost * rng.uniform(0.5, 1.5), 1)
    return DAG(costs, edges)


def job_set(n: int, seed: int = 0) -> list[Job]:
    rng = random.Random(seed)
    return [
        Job(f"J{i + 1}", rng.randint(1, max(1, n // 2)), rng.randint(5, 100)) for i in range(n)
    ]


def bankers_state(n: int, m: int = 3, seed: int = 0) -> tuple[list, list, list]:
    """A random (available, max_claim, allocation) triple that is usually safe."""
    rng = random.Random(seed)
    max_claim = [[rng.randint(1, 10) for _ in range(m)] for _ in range(n)]
    allocation = [[rng.randint(0, c) for c in row] for row in max_claim]
    available = [rng.randint(3, 10) for _ in range(m)]
    return available, max_claim, allocation


def topcuoglu_example() -> DAG:
    """The 10-task, 3-processor DAG from Topcuoglu, Hariri & Wu (2002), Fig. 2.

    HEFT schedules it with makespan 80.
    """
    w = {
        1: [14, 16, 9], 2: [13, 19, 18], 3: [11, 13, 19], 4: [13, 8, 17], 5: [12, 13, 10],
        6: [13, 16, 9], 7: [7, 15, 11], 8: [5, 11, 14], 9: [18, 12, 20], 10: [21, 7, 16],
    }
    c = {
        (1, 2): 18, (1, 3): 12, (1, 4): 9, (1, 5): 11, (1, 6): 14, (2, 8): 19, (2, 9): 16,
        (3, 7): 23, (4, 8): 27, (4, 9): 23, (5, 9): 13, (6, 8): 15, (7, 10): 17, (8, 10): 11,
        (9, 10): 13,
    }
    return DAG(
        {f"n{k}": [float(x) for x in v] for k, v in w.items()},
        {(f"n{a}", f"n{b}"): float(cost) for (a, b), cost in c.items()},
    )
