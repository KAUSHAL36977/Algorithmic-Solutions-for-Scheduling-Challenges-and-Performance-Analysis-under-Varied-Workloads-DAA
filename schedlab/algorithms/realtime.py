"""Periodic real-time scheduling: RMS, DMS and EDF, plus schedulability analysis.

The simulators release job ``k`` of task ``τ`` at ``k·T`` with absolute deadline
``k·T + D`` over one hyperperiod (``lcm`` of the periods). Each job becomes a
:class:`~schedlab.models.Process` with id ``"<task>#<k>"``, so the generic metrics
(response time, deadline-miss ratio, lateness) apply unchanged.

A job that misses its deadline keeps running (soft real-time semantics), so its
lateness shows how far past the deadline it finished.
"""

from __future__ import annotations

import heapq
import math
from fractions import Fraction

from ..models import EPS, PeriodicTask, Process, Schedule, Slice

MAX_HORIZON = 100_000


def hyperperiod(tasks: list[PeriodicTask]) -> float:
    """LCM of the periods (periods may be non-integers such as 2.5)."""
    fracs = [Fraction(t.T).limit_denominator(1000) for t in tasks]
    den = math.lcm(*(f.denominator for f in fracs))
    num = math.lcm(*(int(f * den) for f in fracs))
    return num / den


def utilization(tasks: list[PeriodicTask]) -> float:
    return sum(t.utilization for t in tasks)


def liu_layland_bound(n: int) -> float:
    """RMS utilisation bound n(2^(1/n) - 1); tends to ln 2 ≈ 0.693."""
    return n * (2 ** (1 / n) - 1) if n > 0 else 1.0


def response_time_analysis(tasks: list[PeriodicTask], order: str = "rm") -> dict[str, float]:
    """Exact worst-case response time of each task under fixed priorities.

    Iterates ``R = C_i + Σ_{j ∈ hp(i)} ⌈R / T_j⌉ C_j`` to a fixed point.
    ``order`` is ``"rm"`` (shorter period first) or ``"dm"`` (shorter deadline first).
    Returns ``inf`` for a task whose response time exceeds its deadline.
    """
    ranked = _fixed_priority_order(tasks, order)
    result: dict[str, float] = {}
    for i, task in enumerate(ranked):
        hp = ranked[:i]
        r = task.C + sum(t.C for t in hp)
        while True:
            nxt = task.C + sum(math.ceil(r / t.T - EPS) * t.C for t in hp)
            if nxt > task.deadline + EPS:
                result[task.name] = math.inf
                break
            if abs(nxt - r) <= EPS:
                result[task.name] = nxt
                break
            r = nxt
    return result


def edf_demand_test(tasks: list[PeriodicTask]) -> bool:
    """Exact EDF test (processor demand criterion), synchronous release.

    Schedulable iff U ≤ 1 and for every absolute deadline ``t`` in one hyperperiod
    ``dbf(t) = Σ ⌊(t - D_i) / T_i + 1⌋ C_i ≤ t``.
    """
    u = utilization(tasks)
    if u > 1 + EPS:
        return False
    if all(abs(t.deadline - t.T) <= EPS for t in tasks):
        return True  # implicit deadlines: U ≤ 1 is necessary and sufficient
    horizon = min(hyperperiod(tasks) + max(t.deadline for t in tasks), MAX_HORIZON)
    points = sorted(
        {
            k * t.T + t.deadline
            for t in tasks
            for k in range(int((horizon - t.deadline) // t.T) + 1)
            if k * t.T + t.deadline <= horizon
        }
    )
    for point in points:
        demand = sum(
            max(0, math.floor((point - t.deadline) / t.T + EPS) + 1) * t.C for t in tasks
        )
        if demand > point + EPS:
            return False
    return True


def schedulability_report(tasks: list[PeriodicTask]) -> dict:
    """All the classic tests in one dict (used by the UI and CLI)."""
    n = len(tasks)
    u = utilization(tasks)
    bound = liu_layland_bound(n)
    hyperbolic = math.prod(t.utilization + 1 for t in tasks)
    rta_rm = response_time_analysis(tasks, "rm")
    rta_dm = response_time_analysis(tasks, "dm")
    return {
        "utilization": u,
        "liu_layland_bound": bound,
        "rms_ll_pass": u <= bound + EPS,
        "rms_hyperbolic_pass": hyperbolic <= 2 + EPS,
        "rms_rta": rta_rm,
        "rms_schedulable": all(math.isfinite(r) for r in rta_rm.values()),
        "dms_rta": rta_dm,
        "dms_schedulable": all(math.isfinite(r) for r in rta_dm.values()),
        "edf_schedulable": edf_demand_test(tasks),
        "hyperperiod": hyperperiod(tasks),
    }


def _fixed_priority_order(tasks: list[PeriodicTask], order: str) -> list[PeriodicTask]:
    if order == "rm":
        return sorted(tasks, key=lambda t: (t.T, t.name))
    if order == "dm":
        return sorted(tasks, key=lambda t: (t.deadline, t.name))
    raise ValueError(f"unknown priority order {order!r}")


def _simulate_periodic(
    tasks: list[PeriodicTask], policy: str, name: str, horizon: float | None
) -> Schedule:
    if not tasks:
        return Schedule(name, [], [])
    names = [t.name for t in tasks]
    if len(set(names)) != len(names) or any("#" in t for t in names):
        raise ValueError("task names must be unique and must not contain '#'")
    H = min(horizon or hyperperiod(tasks), MAX_HORIZON)

    if policy in ("rm", "dm"):
        rank = {t.name: i for i, t in enumerate(_fixed_priority_order(tasks, policy))}

    releases: list[tuple[float, int, PeriodicTask, int]] = []  # (time, idx, task, k)
    for idx, t in enumerate(tasks):
        k = 0
        while k * t.T < H - EPS:
            releases.append((k * t.T, idx, t, k))
            k += 1
    releases.sort(key=lambda r: (r[0], r[1]))
    stop = H + max(t.deadline for t in tasks)  # let late jobs finish, then give up

    processes: list[Process] = []
    slices: list[Slice] = []
    ready: list[tuple] = []
    remaining: dict[str, float] = {}
    i = 0
    now = 0.0
    last: str | None = None
    preemptions = 0

    def key(t: PeriodicTask, k: int, release: float) -> tuple:
        if policy == "edf":
            return (release + t.deadline, release, t.name)
        return (rank[t.name], release)

    while (i < len(releases) or ready) and now < stop - EPS:
        while i < len(releases) and releases[i][0] <= now + EPS:
            release, _, t, k = releases[i]
            pid = f"{t.name}#{k}"
            processes.append(Process(pid, release, t.C, deadline=release + t.deadline))
            remaining[pid] = t.C
            heapq.heappush(ready, (*key(t, k, release), pid))
            i += 1
        if not ready:
            now = releases[i][0]
            continue
        pid = ready[0][-1]
        if last is not None and last != pid and remaining.get(last, 0) > EPS:
            preemptions += 1
        next_release = releases[i][0] if i < len(releases) else math.inf
        end = min(now + remaining[pid], next_release, stop)
        if slices and slices[-1].pid == pid and abs(slices[-1].end - now) <= EPS:
            slices[-1] = Slice(pid, slices[-1].start, end)
        else:
            slices.append(Slice(pid, now, end))
        remaining[pid] -= end - now
        now = end
        last = pid
        if remaining[pid] <= EPS:
            heapq.heappop(ready)

    return Schedule(
        name,
        slices,
        processes,
        meta={
            "hyperperiod": H,
            "preemptions": preemptions,
            "utilization": utilization(tasks),
            "tasks": [t.name for t in tasks],
        },
    )


def rate_monotonic(tasks: list[PeriodicTask], horizon: float | None = None) -> Schedule:
    return _simulate_periodic(tasks, "rm", "RMS", horizon)


def deadline_monotonic(tasks: list[PeriodicTask], horizon: float | None = None) -> Schedule:
    return _simulate_periodic(tasks, "dm", "DMS", horizon)


def edf_periodic(tasks: list[PeriodicTask], horizon: float | None = None) -> Schedule:
    return _simulate_periodic(tasks, "edf", "EDF", horizon)
