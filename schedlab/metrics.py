"""Performance metrics computed from any :class:`~schedlab.models.Schedule`.

Definitions (per process ``p``):

* completion ``CT``  — time its last slice ends
* turnaround ``TAT`` — ``CT - arrival``
* waiting ``WT``     — ``TAT - burst`` (time spent ready but not running)
* response ``RT``    — first time on a CPU minus ``arrival``
* slowdown          — ``TAT / burst`` (1.0 means it never waited)
* lateness          — ``CT - deadline`` (positive means a missed deadline)
"""

from __future__ import annotations

import math
from collections import defaultdict

from .models import EPS, Schedule


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def jain_index(xs: list[float]) -> float:
    """Jain's fairness index: 1.0 when all values are equal, 1/n when one dominates."""
    if not xs:
        return 1.0
    sq = sum(x * x for x in xs)
    return (sum(xs) ** 2) / (len(xs) * sq) if sq > 0 else 1.0


def process_table(schedule: Schedule) -> list[dict]:
    """One row of timing metrics per process (unfinished processes have ``None`` fields)."""
    completion = schedule.completion_times()
    first = schedule.first_start_times()
    rows = []
    for p in sorted(schedule.processes, key=lambda p: (p.arrival, p.pid)):
        ct = completion.get(p.pid)
        tat = None if ct is None else ct - p.arrival
        rows.append(
            {
                "pid": p.pid,
                "arrival": p.arrival,
                "burst": p.burst,
                "priority": p.priority,
                "deadline": p.deadline,
                "start": first.get(p.pid),
                "completion": ct,
                "turnaround": tat,
                "waiting": None if tat is None else tat - p.burst,
                "response": None if p.pid not in first else first[p.pid] - p.arrival,
                "slowdown": None if tat is None else tat / p.burst,
                "lateness": None if ct is None or p.deadline is None else ct - p.deadline,
            }
        )
    return rows


def context_switches(schedule: Schedule) -> int:
    """Number of times a CPU moved from one process to a *different* one."""
    by_cpu = defaultdict(list)
    for s in schedule.work_slices():
        by_cpu[s.cpu].append(s)
    switches = 0
    for slices in by_cpu.values():
        slices.sort(key=lambda s: s.start)
        switches += sum(1 for a, b in zip(slices, slices[1:]) if a.pid != b.pid)
    return switches


def summarize(schedule: Schedule) -> dict:
    """Aggregate metrics for a whole schedule. Every value is a plain float/int."""
    rows = process_table(schedule)
    done = [r for r in rows if r["completion"] is not None]
    n = len(rows)
    work = schedule.work_slices()
    busy = sum(s.duration for s in work)
    first_arrival = min((p.arrival for p in schedule.processes), default=0.0)
    makespan = schedule.makespan
    span = max(makespan - first_arrival, EPS)

    loads: dict[int, float] = defaultdict(float)
    for s in work:
        loads[s.cpu] += s.duration
    load_values = [loads.get(c, 0.0) for c in range(schedule.cpus)]
    avg_load = _mean(load_values)

    with_deadline = [r for r in rows if r["deadline"] is not None]
    missed = [
        r for r in with_deadline if r["completion"] is None or r["completion"] > r["deadline"] + EPS
    ]
    lateness = [r["lateness"] for r in with_deadline if r["lateness"] is not None]

    waiting = [r["waiting"] for r in done]
    turnaround = [r["turnaround"] for r in done]
    slowdown = [r["slowdown"] for r in done]
    return {
        "processes": n,
        "completed": len(done),
        "makespan": makespan,
        "avg_waiting": _mean(waiting),
        "max_waiting": max(waiting, default=0.0),
        "avg_turnaround": _mean(turnaround),
        "p95_turnaround": percentile(turnaround, 95),
        "avg_response": _mean([r["response"] for r in rows if r["response"] is not None]),
        "avg_slowdown": _mean(slowdown),
        "throughput": len(done) / span,
        "cpu_utilization": busy / (span * max(schedule.cpus, 1)),
        "context_switches": context_switches(schedule),
        "fairness": jain_index(slowdown),
        "load_imbalance": (max(load_values) / avg_load) if avg_load > 0 else 1.0,
        "deadline_miss_ratio": (len(missed) / len(with_deadline)) if with_deadline else 0.0,
        "max_lateness": max(lateness, default=0.0),
    }


#: Metric metadata used by the CLI, recommender and UI: (label, lower_is_better).
METRICS: dict[str, tuple[str, bool]] = {
    "avg_waiting": ("Avg waiting time", True),
    "avg_turnaround": ("Avg turnaround time", True),
    "avg_response": ("Avg response time", True),
    "p95_turnaround": ("P95 turnaround time", True),
    "max_waiting": ("Max waiting time", True),
    "avg_slowdown": ("Avg slowdown", True),
    "makespan": ("Makespan", True),
    "throughput": ("Throughput (jobs / time unit)", False),
    "cpu_utilization": ("CPU utilisation", False),
    "context_switches": ("Context switches", True),
    "fairness": ("Fairness (Jain index of slowdown)", False),
    "load_imbalance": ("Load imbalance (max / mean)", True),
    "deadline_miss_ratio": ("Deadline miss ratio", True),
    "max_lateness": ("Max lateness", True),
}
