"""Uniprocessor CPU scheduling algorithms.

All algorithms share one event-driven simulator (:func:`simulate`) and differ only in
their *ready-queue policy*. Each policy uses the data structure that gives the
textbook complexity, so the benchmark's empirical scaling can be compared with theory:

=====================  =====================  ===========================
Algorithm              Ready structure        Cost per scheduling event
=====================  =====================  ===========================
FCFS                   deque                  O(1)
SJF / SRTF             binary heap            O(log n)
Priority (+ aging)     binary heap            O(log n)
EDF (aperiodic)        binary heap            O(log n)
Round Robin            deque                  O(1) per quantum
MLFQ                   k deques               O(k)
HRRN                   list scan              O(n)
=====================  =====================  ===========================

Ties are broken deterministically by ``(arrival, pid)``.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass

from ..models import CONTEXT_SWITCH, EPS, Process, Schedule, Slice

INF = math.inf


@dataclass
class _Job:
    proc: Process
    remaining: float
    ready_since: float = 0.0
    level: int = 0  # MLFQ queue level

    @property
    def pid(self) -> str:
        return self.proc.pid

    @property
    def tiebreak(self) -> tuple[float, str]:
        return (self.proc.arrival, self.proc.pid)


class Policy:
    """Ready-queue policy. Subclasses override the hooks they need."""

    #: Re-evaluate the running job whenever a new process arrives.
    preemptive = False

    def __len__(self) -> int:
        raise NotImplementedError

    def add(self, job: _Job, now: float) -> None:
        raise NotImplementedError

    def pop(self, now: float) -> _Job:
        raise NotImplementedError

    def quantum(self, job: _Job) -> float:
        return INF

    def should_preempt(self, running: _Job, now: float) -> bool:
        return False

    def on_quantum_expired(self, job: _Job, now: float) -> None:
        self.add(job, now)

    def on_preempt(self, job: _Job, now: float) -> None:
        self.add(job, now)


class _HeapPolicy(Policy):
    """Min-heap keyed by :meth:`key`; the key must not change while a job waits."""

    def __init__(self) -> None:
        self._heap: list[tuple] = []

    def __len__(self) -> int:
        return len(self._heap)

    def key(self, job: _Job) -> tuple:
        raise NotImplementedError

    def add(self, job: _Job, now: float) -> None:
        job.ready_since = now
        heapq.heappush(self._heap, (*self.key(job), *job.tiebreak, id(job), job))

    def pop(self, now: float) -> _Job:
        return heapq.heappop(self._heap)[-1]

    def peek(self) -> _Job:
        return self._heap[0][-1]


class FCFS(Policy):
    def __init__(self) -> None:
        self._q: deque[_Job] = deque()

    def __len__(self) -> int:
        return len(self._q)

    def add(self, job: _Job, now: float) -> None:
        self._q.append(job)

    def pop(self, now: float) -> _Job:
        return self._q.popleft()


class RoundRobin(FCFS):
    def __init__(self, quantum: float = 4.0) -> None:
        if quantum <= 0:
            raise ValueError("quantum must be positive")
        super().__init__()
        self.q = quantum

    def quantum(self, job: _Job) -> float:
        return self.q


class SJF(_HeapPolicy):
    """Non-preemptive Shortest Job First."""

    def key(self, job: _Job) -> tuple:
        return (job.proc.burst,)


class SRTF(_HeapPolicy):
    """Preemptive SJF: Shortest Remaining Time First."""

    preemptive = True

    def key(self, job: _Job) -> tuple:
        return (job.remaining,)

    def should_preempt(self, running: _Job, now: float) -> bool:
        return len(self) > 0 and self.peek().remaining < running.remaining - EPS


class PriorityPolicy(_HeapPolicy):
    """Priority scheduling (lower number = higher priority) with optional linear aging.

    With aging rate ``a`` a waiting job's effective priority is
    ``priority - a * (now - ready_since)``. Because ``-a * now`` is the same for
    every waiting job, ordering by the *static* key ``priority + a * ready_since``
    is equivalent, so aging costs nothing extra: still O(log n) per operation.
    """

    def __init__(self, preemptive: bool = False, aging: float = 0.0) -> None:
        super().__init__()
        self.preemptive = preemptive
        self.aging = aging

    def key(self, job: _Job) -> tuple:
        return (job.proc.priority + self.aging * job.ready_since,)

    def effective(self, job: _Job, now: float) -> float:
        return job.proc.priority - self.aging * (now - job.ready_since)

    def should_preempt(self, running: _Job, now: float) -> bool:
        return len(self) > 0 and self.effective(self.peek(), now) < running.proc.priority - EPS


class EDFAperiodic(_HeapPolicy):
    """Preemptive Earliest Deadline First over one-shot processes."""

    preemptive = True

    @staticmethod
    def _deadline(job: _Job) -> float:
        return INF if job.proc.deadline is None else job.proc.deadline

    def key(self, job: _Job) -> tuple:
        return (self._deadline(job),)

    def should_preempt(self, running: _Job, now: float) -> bool:
        return len(self) > 0 and self._deadline(self.peek()) < self._deadline(running) - EPS


class HRRN(Policy):
    """Highest Response Ratio Next: ratio = (waiting + burst) / burst. Non-preemptive.

    The ratio of every waiting job changes over time, so each dispatch scans the
    ready list: O(n) per decision, O(n^2) overall.
    """

    def __init__(self) -> None:
        self._ready: list[_Job] = []

    def __len__(self) -> int:
        return len(self._ready)

    def add(self, job: _Job, now: float) -> None:
        job.ready_since = now
        self._ready.append(job)

    def pop(self, now: float) -> _Job:
        def rank(j: _Job):
            ratio = (now - j.proc.arrival + j.proc.burst) / j.proc.burst
            return (-ratio, *j.tiebreak)

        best = min(range(len(self._ready)), key=lambda i: rank(self._ready[i]))
        self._ready[best], self._ready[-1] = self._ready[-1], self._ready[best]
        return self._ready.pop()


class MLFQ(Policy):
    """Multi-Level Feedback Queue.

    New jobs enter level 0. A job that uses its whole quantum is demoted one level;
    the last level is FCFS. A newly arrived job preempts a job running on a lower
    level. ``quanta`` defaults to ``(q, 2q, ∞)``.
    """

    preemptive = True

    def __init__(self, quantum: float = 4.0, levels: int = 3) -> None:
        if levels < 1:
            raise ValueError("MLFQ needs at least one level")
        self.quanta = [quantum * (2**i) for i in range(levels - 1)] + [INF]
        self._queues: list[deque[_Job]] = [deque() for _ in range(levels)]

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues)

    def add(self, job: _Job, now: float) -> None:
        self._queues[job.level].append(job)

    def pop(self, now: float) -> _Job:
        for q in self._queues:
            if q:
                return q.popleft()
        raise IndexError("pop from empty MLFQ")

    def quantum(self, job: _Job) -> float:
        return self.quanta[job.level]

    def on_quantum_expired(self, job: _Job, now: float) -> None:
        job.level = min(job.level + 1, len(self._queues) - 1)
        self.add(job, now)

    def should_preempt(self, running: _Job, now: float) -> bool:
        return any(self._queues[lvl] for lvl in range(running.level))


def simulate(
    processes: list[Process],
    policy: Policy,
    name: str,
    context_switch: float = 0.0,
) -> Schedule:
    """Run ``processes`` through ``policy`` on one CPU.

    ``context_switch`` inserts that much dispatcher overhead (a ``CONTEXT_SWITCH``
    slice) whenever the CPU moves to a different process than the one it last ran.

    The loop is event-driven: time jumps straight to the next arrival, completion
    or quantum expiry, so the cost is proportional to the number of *events*, not
    to the length of the timeline.
    """
    if context_switch < 0:
        raise ValueError("context_switch must be non-negative")
    pids = [p.pid for p in processes]
    if len(set(pids)) != len(pids):
        raise ValueError("process ids must be unique")

    pending = sorted(processes, key=lambda p: (p.arrival, p.pid))
    jobs = {p.pid: _Job(p, p.burst) for p in pending}
    n = len(pending)
    slices: list[Slice] = []
    i = 0
    done = 0
    now = pending[0].arrival if n else 0.0
    running: _Job | None = None
    budget = INF
    last_pid: str | None = None

    def admit(t: float) -> None:
        nonlocal i
        while i < n and pending[i].arrival <= t + EPS:
            policy.add(jobs[pending[i].pid], pending[i].arrival)
            i += 1

    def record(pid: str, start: float, end: float) -> None:
        if end - start <= EPS:
            return
        if slices and slices[-1].pid == pid and abs(slices[-1].end - start) <= EPS:
            slices[-1] = Slice(pid, slices[-1].start, end)
        else:
            slices.append(Slice(pid, start, end))

    while done < n:
        admit(now)
        if running is None:
            if not len(policy):
                now = pending[i].arrival  # CPU idles until the next arrival
                continue
            running = policy.pop(now)
            if last_pid is not None and running.pid != last_pid and context_switch > 0:
                record(CONTEXT_SWITCH, now, now + context_switch)
                now += context_switch
                admit(now)
            budget = policy.quantum(running)

        end = now + min(running.remaining, budget)
        if policy.preemptive and i < n and pending[i].arrival < end - EPS:
            end = pending[i].arrival
        record(running.pid, now, end)
        running.remaining -= end - now
        budget -= end - now
        now = end
        admit(now)

        if running.remaining <= EPS:
            done += 1
        elif budget <= EPS:
            policy.on_quantum_expired(running, now)
        elif policy.preemptive and policy.should_preempt(running, now):
            policy.on_preempt(running, now)
        else:
            continue  # an arrival that did not preempt: keep running
        last_pid, running = running.pid, None

    return Schedule(name, slices, list(processes), meta={"context_switch_cost": context_switch})


# ---------------------------------------------------------------------------
# Public algorithm functions (the registry points at these).
# ---------------------------------------------------------------------------


def fcfs(processes: list[Process], context_switch: float = 0.0) -> Schedule:
    return simulate(processes, FCFS(), "FCFS", context_switch)


def sjf(processes: list[Process], context_switch: float = 0.0) -> Schedule:
    return simulate(processes, SJF(), "SJF", context_switch)


def srtf(processes: list[Process], context_switch: float = 0.0) -> Schedule:
    return simulate(processes, SRTF(), "SRTF", context_switch)


def round_robin(
    processes: list[Process], quantum: float = 4.0, context_switch: float = 0.0
) -> Schedule:
    return simulate(processes, RoundRobin(quantum), f"RR (q={quantum:g})", context_switch)


def priority(
    processes: list[Process], aging: float = 0.0, context_switch: float = 0.0
) -> Schedule:
    return simulate(processes, PriorityPolicy(False, aging), "Priority", context_switch)


def priority_preemptive(
    processes: list[Process], aging: float = 0.0, context_switch: float = 0.0
) -> Schedule:
    return simulate(processes, PriorityPolicy(True, aging), "Priority (preemptive)", context_switch)


def hrrn(processes: list[Process], context_switch: float = 0.0) -> Schedule:
    return simulate(processes, HRRN(), "HRRN", context_switch)


def mlfq(
    processes: list[Process], quantum: float = 4.0, levels: int = 3, context_switch: float = 0.0
) -> Schedule:
    return simulate(processes, MLFQ(quantum, levels), "MLFQ", context_switch)


def edf(processes: list[Process], context_switch: float = 0.0) -> Schedule:
    return simulate(processes, EDFAperiodic(), "EDF", context_switch)
