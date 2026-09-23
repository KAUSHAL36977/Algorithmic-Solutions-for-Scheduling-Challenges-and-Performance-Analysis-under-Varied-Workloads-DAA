"""Scheduling independent jobs on ``m`` identical processors (minimise makespan).

* **List scheduling** (Graham, 1966): take jobs in the given order and give each
  one to the processor that frees up first. A min-heap of processor finish times
  makes it O(n log m). It is a (2 - 1/m)-approximation.
* **LPT** (Longest Processing Time first): list scheduling after sorting jobs by
  decreasing length, O(n log n). A (4/3 - 1/(3m))-approximation.
* **Work stealing**: every worker owns a deque, pops work from its own tail and,
  when idle, steals from the head of a random victim's deque (Cilk/TBB/Go style).
  Simulated with an event heap.

Processes with ``arrival > 0`` are honoured: a job never starts before it arrives.
"""

from __future__ import annotations

import heapq
import random
from collections import deque

from ..models import Process, Schedule, Slice


def makespan_lower_bound(processes: list[Process], m: int) -> float:
    """No schedule can beat max(total work / m, longest job)."""
    if not processes:
        return 0.0
    return max(sum(p.burst for p in processes) / m, max(p.burst for p in processes))


def _list_schedule(order: list[Process], m: int, name: str) -> Schedule:
    if m < 1:
        raise ValueError("need at least one processor")
    free = [(0.0, cpu) for cpu in range(m)]  # (time processor becomes free, cpu)
    slices: list[Slice] = []
    for p in order:
        t, cpu = heapq.heappop(free)
        start = max(t, p.arrival)
        slices.append(Slice(p.pid, start, start + p.burst, cpu))
        heapq.heappush(free, (start + p.burst, cpu))
    schedule = Schedule(name, slices, list(order), cpus=m)
    lb = makespan_lower_bound(order, m)
    schedule.meta["lower_bound"] = lb
    schedule.meta["approx_ratio"] = schedule.makespan / lb if lb else 1.0
    return schedule


def list_scheduling(processes: list[Process], m: int = 4) -> Schedule:
    order = sorted(processes, key=lambda p: (p.arrival, p.pid))
    return _list_schedule(order, m, "List Scheduling")


def lpt(processes: list[Process], m: int = 4) -> Schedule:
    order = sorted(processes, key=lambda p: (-p.burst, p.arrival, p.pid))
    return _list_schedule(order, m, "LPT")


def work_stealing(
    processes: list[Process], m: int = 4, steal_cost: float = 0.5, seed: int = 0
) -> Schedule:
    """Jobs start in contiguous chunks on each worker's deque (as a parallel loop
    would split them), which is badly unbalanced for skewed job sizes; idle workers
    then steal from random victims, paying ``steal_cost`` per successful steal.
    """
    if m < 1:
        raise ValueError("need at least one processor")
    rng = random.Random(seed)
    order = sorted(processes, key=lambda p: (p.arrival, p.pid))
    chunk = -(-len(order) // m) if order else 0
    deques = [deque(order[w * chunk : (w + 1) * chunk]) for w in range(m)]
    events = [(0.0, w) for w in range(m)]  # (time worker becomes idle, worker)
    heapq.heapify(events)
    slices: list[Slice] = []
    steals = 0
    while events:
        now, w = heapq.heappop(events)
        if deques[w]:
            job = deques[w].pop()  # own work: LIFO from the tail
        else:
            victims = [v for v in range(m) if deques[v]]
            if not victims:
                continue  # nothing left anywhere: this worker retires
            job = deques[rng.choice(victims)].popleft()  # steal FIFO from the head
            now += steal_cost
            steals += 1
        start = max(now, job.arrival)
        slices.append(Slice(job.pid, start, start + job.burst, w))
        heapq.heappush(events, (start + job.burst, w))
    schedule = Schedule("Work Stealing", slices, list(processes), cpus=m)
    lb = makespan_lower_bound(processes, m)
    schedule.meta.update(
        steals=steals, lower_bound=lb, approx_ratio=schedule.makespan / lb if lb else 1.0
    )
    return schedule
