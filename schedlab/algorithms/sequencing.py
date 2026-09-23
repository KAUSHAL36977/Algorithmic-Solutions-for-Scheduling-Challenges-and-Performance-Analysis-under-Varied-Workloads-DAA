"""Job sequencing with deadlines (unit-time jobs, maximise total profit).

The textbook greedy takes jobs in decreasing profit order and puts each in the
*latest* free slot on or before its deadline. Scanning for that slot naively
costs O(n·d). A disjoint-set union (DSU) where ``find(s)`` returns the latest
free slot ≤ ``s`` reduces the whole algorithm to O(n log n) for the sort plus
near-constant α(n) per job.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Job, Process, Schedule, Slice


@dataclass
class SequencingResult:
    sequence: list[Job]  # in execution (slot) order
    rejected: list[Job]
    total_profit: float
    slots: dict[int, str]  # slot number (1-based) -> job id

    def to_schedule(self) -> Schedule:
        """Unit slices at ``[slot - 1, slot)``, for Gantt charts."""
        processes = [Process(j.id, 0.0, 1.0, deadline=float(j.deadline)) for j in self.sequence]
        slices = [Slice(job_id, slot - 1, slot) for slot, job_id in sorted(self.slots.items())]
        return Schedule("Greedy Job Sequencing", slices, processes)


class _LatestFreeSlot:
    """DSU over slots 0..d. ``find(s)`` = latest free slot ≤ s (0 means none)."""

    def __init__(self, d: int) -> None:
        self.parent = list(range(d + 1))

    def find(self, s: int) -> int:
        root = s
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[s] != root:  # path compression
            self.parent[s], s = root, self.parent[s]
        return root

    def occupy(self, s: int) -> None:
        self.parent[s] = s - 1


def job_sequencing(jobs: list[Job]) -> SequencingResult:
    if not jobs:
        return SequencingResult([], [], 0.0, {})
    ids = [j.id for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("job ids must be unique")
    # A job with deadline > n can always be placed by slot n, so cap slots at n.
    d = min(max(j.deadline for j in jobs), len(jobs))
    dsu = _LatestFreeSlot(d)
    slots: dict[int, str] = {}
    by_id = {j.id: j for j in jobs}
    rejected: list[Job] = []
    for job in sorted(jobs, key=lambda j: (-j.profit, j.deadline, j.id)):
        slot = dsu.find(min(job.deadline, d)) if job.deadline >= 1 else 0
        if slot == 0:
            rejected.append(job)
            continue
        slots[slot] = job.id
        dsu.occupy(slot)
    sequence = [by_id[slots[s]] for s in sorted(slots)]
    return SequencingResult(sequence, rejected, sum(j.profit for j in sequence), slots)
