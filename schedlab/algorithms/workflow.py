"""Workflow (DAG) scheduling: CPM, Monte Carlo CPM and HEFT.

* **CPM**: a forward pass gives earliest start/finish times and a backward pass
  gives latest start/finish times, both in topological order, so it runs in
  O(V + E). Tasks with zero slack form the critical path.
* **Monte Carlo CPM**: samples uncertain task durations and re-runs the forward
  pass S times, O(S·(V + E)). The output is a completion-time distribution plus
  each task's *criticality index* (how often it lies on the critical path).
* **HEFT** (Topcuoglu, Hariri & Wu, 2002): ranks tasks by *upward rank* (the
  longest remaining path to an exit, using average costs), then places each task
  in turn on the processor with the earliest finish time. It can insert a task
  into an idle gap left earlier. O(V²·P).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..metrics import percentile
from ..models import DAG, EPS, Process, Schedule, Slice


def _durations(dag: DAG, durations: dict[str, float] | None) -> dict[str, float]:
    if durations is not None:
        return durations
    return {t: sum(c) / len(c) for t, c in dag.costs.items()}


@dataclass
class CPMResult:
    rows: list[dict]
    duration: float
    critical_path: list[str]

    def to_schedule(self) -> Schedule:
        processes = [Process(r["task"], 0.0, r["duration"]) for r in self.rows if r["duration"] > 0]
        slices = [Slice(r["task"], r["ES"], r["EF"]) for r in self.rows if r["duration"] > 0]
        return Schedule("CPM (earliest start)", slices, processes, meta={"critical": self.critical_path})


def _forward(order: list[str], pred: dict[str, list[str]], d: dict[str, float]) -> dict[str, float]:
    ef: dict[str, float] = {}
    for t in order:
        ef[t] = max((ef[u] for u in pred[t]), default=0.0) + d[t]
    return ef


def critical_path_method(dag: DAG, durations: dict[str, float] | None = None) -> CPMResult:
    d = _durations(dag, durations)
    order = dag.topological_order()
    pred, succ = dag.predecessors(), dag.successors()
    ef = _forward(order, pred, d)
    es = {t: ef[t] - d[t] for t in order}
    project = max(ef.values(), default=0.0)
    lf: dict[str, float] = {}
    for t in reversed(order):
        lf[t] = min((lf[v] - d[v] for v in succ[t]), default=project)
    ls = {t: lf[t] - d[t] for t in order}
    slack = {t: ls[t] - es[t] for t in order}

    # Walk one critical path: start at a zero-slack source, follow zero-slack
    # successors whose earliest start equals our earliest finish.
    path: list[str] = []
    current = next((t for t in order if not pred[t] and slack[t] <= EPS), None)
    while current is not None:
        path.append(current)
        current = next(
            (v for v in succ[current] if slack[v] <= EPS and abs(es[v] - ef[current]) <= EPS),
            None,
        )
    rows = [
        {
            "task": t,
            "duration": d[t],
            "ES": es[t],
            "EF": ef[t],
            "LS": ls[t],
            "LF": lf[t],
            "slack": slack[t],
            "critical": slack[t] <= EPS,
        }
        for t in order
    ]
    return CPMResult(rows, project, path)


def monte_carlo_cpm(
    dag: DAG,
    durations: dict[str, float] | None = None,
    spread: float = 0.3,
    samples: int = 1000,
    seed: int = 0,
) -> dict:
    """Sample each duration from a right-skewed triangular distribution
    ``(d·(1-spread), d, d·(1+2·spread))`` (tasks overrun more often than they
    finish early, as in PERT) and collect project completion statistics.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    base = _durations(dag, durations)
    order = dag.topological_order()
    pred = dag.predecessors()
    rng = random.Random(seed)
    totals: list[float] = []
    critical_hits = dict.fromkeys(order, 0)
    for _ in range(samples):
        d = {
            t: rng.triangular(v * (1 - spread), v * (1 + 2 * spread), v) if v > 0 else 0.0
            for t, v in base.items()
        }
        ef = _forward(order, pred, d)
        total = max(ef.values(), default=0.0)
        totals.append(total)
        # Backtrack the binding predecessor chain from the latest-finishing task.
        t = max(order, key=lambda x: ef[x]) if order else None
        while t is not None:
            critical_hits[t] += 1
            t = max(pred[t], key=lambda u: ef[u]) if pred[t] else None
    return {
        "deterministic": critical_path_method(dag, base).duration,
        "samples": totals,
        "mean": sum(totals) / len(totals),
        "p50": percentile(totals, 50),
        "p90": percentile(totals, 90),
        "p95": percentile(totals, 95),
        "criticality": {t: hits / samples for t, hits in critical_hits.items()},
    }


def upward_ranks(dag: DAG) -> dict[str, float]:
    avg = {t: sum(c) / len(c) for t, c in dag.costs.items()}
    succ = dag.successors()
    rank: dict[str, float] = {}
    for t in reversed(dag.topological_order()):
        rank[t] = avg[t] + max((dag.edges[(t, s)] + rank[s] for s in succ[t]), default=0.0)
    return rank


def _earliest_gap(busy: list[tuple[float, float]], ready: float, length: float) -> float:
    """Insertion policy: earliest start ≥ ``ready`` that fits in an idle gap of ``busy``."""
    prev_end = 0.0
    for start, end in busy:
        candidate = max(ready, prev_end)
        if candidate + length <= start + EPS:
            return candidate
        prev_end = max(prev_end, end)
    return max(ready, prev_end)


def heft(dag: DAG) -> Schedule:
    P = dag.processors
    if P < 1:
        return Schedule("HEFT", [], [], cpus=1)
    topo = dag.topological_order()
    position = {t: i for i, t in enumerate(topo)}
    rank = upward_ranks(dag)
    order = sorted(topo, key=lambda t: (-round(rank[t], 9), position[t]))
    pred = dag.predecessors()

    proc_of: dict[str, int] = {}
    finish: dict[str, float] = {}
    busy: list[list[tuple[float, float]]] = [[] for _ in range(P)]
    slices: list[Slice] = []
    processes: list[Process] = []
    for t in order:
        best = None  # (eft, start, processor)
        for p in range(P):
            ready = max(
                (finish[u] + (0.0 if proc_of[u] == p else dag.edges[(u, t)]) for u in pred[t]),
                default=0.0,
            )
            w = dag.costs[t][p]
            start = _earliest_gap(busy[p], ready, w)
            if best is None or start + w < best[0] - EPS:
                best = (start + w, start, p)
        eft, start, p = best
        proc_of[t], finish[t] = p, eft
        busy[p].append((start, eft))
        busy[p].sort()
        slices.append(Slice(t, start, eft, p))
        processes.append(Process(t, 0.0, dag.costs[t][p]))

    schedule = Schedule("HEFT", slices, processes, cpus=P)
    # Schedule Length Ratio: makespan over the critical path with every task on its
    # fastest processor and free communication (a lower bound), so SLR ≥ 1.
    fastest = {t: min(c) for t, c in dag.costs.items()}
    cp_min = critical_path_method(dag, fastest).duration
    sequential = min(sum(dag.costs[t][p] for t in topo) for p in range(P))
    schedule.meta.update(
        ranks=rank,
        order=order,
        processor_of=proc_of,
        slr=schedule.makespan / cp_min if cp_min else 1.0,
        speedup=sequential / schedule.makespan if schedule.makespan else 1.0,
    )
    return schedule
