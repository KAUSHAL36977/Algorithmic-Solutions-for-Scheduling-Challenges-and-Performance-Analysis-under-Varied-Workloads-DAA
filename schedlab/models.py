"""Core data model shared by every simulator, metric, chart and test.

Every scheduling algorithm in :mod:`schedlab.algorithms` returns a :class:`Schedule`,
so metrics (:mod:`schedlab.metrics`) and visualisations never need to know which
algorithm produced a timeline.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

EPS = 1e-9

#: Pseudo-pid used for context-switch overhead slices.
CONTEXT_SWITCH = "⟲ switch"


@dataclass(frozen=True)
class Process:
    """A unit of CPU work.

    ``priority`` follows the Unix convention: a *lower* number is a *higher* priority.
    ``deadline`` is absolute (same time axis as ``arrival``).
    """

    pid: str
    arrival: float
    burst: float
    priority: int = 0
    deadline: float | None = None

    def __post_init__(self) -> None:
        if self.burst <= 0:
            raise ValueError(f"{self.pid}: burst must be positive, got {self.burst}")
        if self.arrival < 0:
            raise ValueError(f"{self.pid}: arrival must be non-negative, got {self.arrival}")


@dataclass(frozen=True)
class PeriodicTask:
    """A periodic real-time task: WCET ``C`` every period ``T`` with relative deadline ``D``."""

    name: str
    C: float
    T: float
    D: float | None = None

    def __post_init__(self) -> None:
        if self.C <= 0 or self.T <= 0:
            raise ValueError(f"{self.name}: C and T must be positive")
        if self.D is not None and self.D <= 0:
            raise ValueError(f"{self.name}: D must be positive")

    @property
    def deadline(self) -> float:
        return self.T if self.D is None else self.D

    @property
    def utilization(self) -> float:
        return self.C / self.T


@dataclass(frozen=True)
class Job:
    """A unit-time job for deadline/profit sequencing."""

    id: str
    deadline: int
    profit: float


@dataclass(frozen=True)
class Slice:
    """``pid`` ran on ``cpu`` during ``[start, end)``."""

    pid: str
    start: float
    end: float
    cpu: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Schedule:
    """The output of a scheduling algorithm: a timeline plus the work it scheduled."""

    algorithm: str
    slices: list[Slice]
    processes: list[Process]
    cpus: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    def work_slices(self) -> list[Slice]:
        """Slices that executed real work (context-switch overhead excluded)."""
        return [s for s in self.slices if s.pid != CONTEXT_SWITCH]

    @property
    def makespan(self) -> float:
        return max((s.end for s in self.slices), default=0.0)

    def completion_times(self) -> dict[str, float]:
        done: dict[str, float] = {}
        executed: dict[str, float] = defaultdict(float)
        for s in self.work_slices():
            executed[s.pid] += s.duration
            done[s.pid] = max(done.get(s.pid, s.end), s.end)
        bursts = {p.pid: p.burst for p in self.processes}
        # A process is complete only if all of its burst was executed.
        return {pid: t for pid, t in done.items() if executed[pid] >= bursts.get(pid, 0) - 1e-6}

    def first_start_times(self) -> dict[str, float]:
        first: dict[str, float] = {}
        for s in self.work_slices():
            if s.pid not in first or s.start < first[s.pid]:
                first[s.pid] = s.start
        return first


def task_of(pid: str) -> str:
    """Real-time job ids look like ``"T1#3"`` (4th job of task T1); return ``"T1"``."""
    return pid.split("#", 1)[0]


@dataclass
class DAG:
    """A task graph for workflow scheduling.

    ``costs[task]`` holds the execution cost of ``task`` on each processor
    (a single-element list for processor-independent durations, as in CPM).
    ``edges[(u, v)]`` is the communication cost paid when ``u`` and ``v`` run on
    different processors.
    """

    costs: dict[str, list[float]]
    edges: dict[tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for u, v in self.edges:
            if u not in self.costs or v not in self.costs:
                raise ValueError(f"edge ({u}, {v}) references an unknown task")
        widths = {len(c) for c in self.costs.values()}
        if len(widths) > 1:
            raise ValueError("every task needs a cost for every processor")

    @property
    def tasks(self) -> list[str]:
        return list(self.costs)

    @property
    def processors(self) -> int:
        return len(next(iter(self.costs.values()))) if self.costs else 0

    def successors(self) -> dict[str, list[str]]:
        succ: dict[str, list[str]] = {t: [] for t in self.costs}
        for u, v in self.edges:
            succ[u].append(v)
        return succ

    def predecessors(self) -> dict[str, list[str]]:
        pred: dict[str, list[str]] = {t: [] for t in self.costs}
        for u, v in self.edges:
            pred[v].append(u)
        return pred

    def topological_order(self) -> list[str]:
        """Kahn's algorithm, O(V + E). Raises ``ValueError`` on a cycle."""
        indeg = {t: 0 for t in self.costs}
        for _, v in self.edges:
            indeg[v] += 1
        succ = self.successors()
        queue = deque(t for t in self.costs if indeg[t] == 0)
        order: list[str] = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        if len(order) != len(self.costs):
            raise ValueError("task graph contains a cycle")
        return order
