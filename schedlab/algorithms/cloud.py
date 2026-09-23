"""Mapping independent tasks onto heterogeneous machines (grid and cloud).

Input is an *Expected Time to Compute* matrix: ``etc[i][j]`` is the run time of
task ``i`` on machine ``j``. Both heuristics repeat until every task is mapped:

1. For every unmapped task, find the machine giving its minimum completion time
   ``ct = ready[j] + etc[i][j]``.
2. **Min-Min** maps the task whose minimum ``ct`` is *smallest*; **Max-Min** maps
   the task whose minimum ``ct`` is *largest*.

O(n² m) for n tasks and m machines. Min-Min finishes many small tasks early but
can leave one big task until last; Max-Min places big tasks first and fits the
small ones around them.
"""

from __future__ import annotations

from ..models import Process, Schedule, Slice


def _validate(etc: list[list[float]]) -> int:
    if not etc:
        return 0
    m = len(etc[0])
    if m == 0 or any(len(row) != m for row in etc):
        raise ValueError("ETC matrix must be rectangular with at least one machine")
    return m


def _batch_map(
    etc: list[list[float]], pick_max: bool, name: str, task_ids: list[str] | None
) -> Schedule:
    m = _validate(etc)
    ids = task_ids or [f"t{i}" for i in range(len(etc))]
    ready = [0.0] * m
    unmapped = set(range(len(etc)))
    slices: list[Slice] = []
    processes: list[Process] = []
    while unmapped:
        best = None  # (min ct, task, machine)
        for i in sorted(unmapped):
            ct, j = min((ready[j] + etc[i][j], j) for j in range(m))
            if best is None or (ct > best[0] if pick_max else ct < best[0]):
                best = (ct, i, j)
        ct, i, j = best
        slices.append(Slice(ids[i], ready[j], ct, j))
        processes.append(Process(ids[i], 0.0, etc[i][j]))
        ready[j] = ct
        unmapped.remove(i)
    return Schedule(name, slices, processes, cpus=m, meta={"machine_ready": ready})


def min_min(etc: list[list[float]], task_ids: list[str] | None = None) -> Schedule:
    return _batch_map(etc, pick_max=False, name="Min-Min", task_ids=task_ids)


def max_min(etc: list[list[float]], task_ids: list[str] | None = None) -> Schedule:
    return _batch_map(etc, pick_max=True, name="Max-Min", task_ids=task_ids)
