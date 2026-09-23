"""Algorithm registry: the one place the CLI, benchmark, recommender and UI look up
which algorithms exist, what they take, and what complexity theory promises."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import cloud, cpu, energy, multiprocessor, realtime, resource, sequencing, workflow


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    name: str
    family: str
    fn: Callable[..., Any]
    complexity: str
    kb: str  # name of the matching knowledge-base entry
    params: dict[str, Any] = field(default_factory=dict)
    preemptive: bool | None = None

    def call(self, *args: Any, **params: Any) -> Any:
        """Call the algorithm, silently dropping parameters it does not accept."""
        accepted = inspect.signature(self.fn).parameters
        return self.fn(*args, **{k: v for k, v in params.items() if k in accepted})


_SPECS = [
    # --- uniprocessor CPU scheduling -------------------------------------------------
    AlgorithmSpec("fcfs", "FCFS", "cpu", cpu.fcfs, "O(n log n)", "First-Come, First-Served (FCFS)",
                  preemptive=False),
    AlgorithmSpec("sjf", "SJF", "cpu", cpu.sjf, "O(n log n)", "Shortest Job First (SJF)",
                  preemptive=False),
    AlgorithmSpec("srtf", "SRTF", "cpu", cpu.srtf, "O(n log n)",
                  "Shortest Remaining Time First (SRTF)", preemptive=True),
    AlgorithmSpec("rr", "Round Robin", "cpu", cpu.round_robin, "O(n log n + B/q)",
                  "Round Robin (RR)", {"quantum": 4.0}, preemptive=True),
    AlgorithmSpec("priority", "Priority", "cpu", cpu.priority, "O(n log n)",
                  "Priority Scheduling", {"aging": 0.0}, preemptive=False),
    AlgorithmSpec("priority_p", "Priority (preemptive)", "cpu", cpu.priority_preemptive,
                  "O(n log n)", "Priority Scheduling", {"aging": 0.0}, preemptive=True),
    AlgorithmSpec("hrrn", "HRRN", "cpu", cpu.hrrn, "O(n²)",
                  "Highest Response Ratio Next (HRRN)", preemptive=False),
    AlgorithmSpec("mlfq", "MLFQ", "cpu", cpu.mlfq, "O(n log n + k·B/q)",
                  "Multi-Level Feedback Queue (MLFQ)", {"quantum": 4.0, "levels": 3},
                  preemptive=True),
    AlgorithmSpec("edf", "EDF", "cpu", cpu.edf, "O(n log n)", "Earliest Deadline First (EDF)",
                  preemptive=True),
    # --- periodic real-time ----------------------------------------------------------
    AlgorithmSpec("rms", "RMS", "realtime", realtime.rate_monotonic, "O(J log n)",
                  "Rate Monotonic Scheduling (RMS)", preemptive=True),
    AlgorithmSpec("dms", "DMS", "realtime", realtime.deadline_monotonic, "O(J log n)",
                  "Deadline Monotonic Scheduling (DMS)", preemptive=True),
    AlgorithmSpec("edf_rt", "EDF (periodic)", "realtime", realtime.edf_periodic, "O(J log J)",
                  "Earliest Deadline First (EDF)", preemptive=True),
    # --- sequencing --------------------------------------------------------------------
    AlgorithmSpec("job_seq", "Greedy Job Sequencing", "sequencing", sequencing.job_sequencing,
                  "O(n log n)", "Greedy Job Sequencing"),
    # --- identical multiprocessors ---------------------------------------------------
    AlgorithmSpec("list", "List Scheduling", "multiprocessor", multiprocessor.list_scheduling,
                  "O(n log n + n log m)", "List Scheduling (Graham / LPT)", {"m": 4}),
    AlgorithmSpec("lpt", "LPT", "multiprocessor", multiprocessor.lpt, "O(n log n + n log m)",
                  "List Scheduling (Graham / LPT)", {"m": 4}),
    AlgorithmSpec("work_stealing", "Work Stealing", "multiprocessor",
                  multiprocessor.work_stealing, "O(n log m + s·m)", "Work Stealing",
                  {"m": 4, "steal_cost": 0.5}),
    # --- heterogeneous machines --------------------------------------------------------
    AlgorithmSpec("min_min", "Min-Min", "cloud", cloud.min_min, "O(n²·m)", "Min-Min Algorithm"),
    AlgorithmSpec("max_min", "Max-Min", "cloud", cloud.max_min, "O(n²·m)", "Max-Min Algorithm"),
    # --- workflows -----------------------------------------------------------------------
    AlgorithmSpec("cpm", "CPM", "workflow", workflow.critical_path_method, "O(V + E)",
                  "Critical Path Method (CPM)"),
    AlgorithmSpec("heft", "HEFT", "workflow", workflow.heft, "O(V²·P)",
                  "Heterogeneous Earliest Finish Time (HEFT)"),
    AlgorithmSpec("monte_carlo", "Monte Carlo CPM", "workflow", workflow.monte_carlo_cpm,
                  "O(S·(V + E))", "Monte Carlo Scheduling", {"samples": 1000, "spread": 0.3}),
    # --- resource allocation ---------------------------------------------------------------
    AlgorithmSpec("bankers", "Banker's Algorithm", "resource", resource.bankers_safety,
                  "O(n²·m)", "Banker's Algorithm"),
    AlgorithmSpec("max_min_fair", "Max-Min Fair Share", "resource", resource.max_min_fair,
                  "O(n log n)", "Max-Min Fair Allocation"),
    # --- energy ----------------------------------------------------------------------------
    AlgorithmSpec("dvfs", "Static DVFS (EDF)", "energy", energy.dvfs_edf, "O(n + L)",
                  "DVFS-based Scheduling"),
    AlgorithmSpec("sleep", "Break-even Sleep", "energy", energy.sleep_schedule, "O(g log g)",
                  "Sleep Scheduling"),
]

REGISTRY: dict[str, AlgorithmSpec] = {s.key: s for s in _SPECS}

FAMILIES = {
    "cpu": "Uniprocessor CPU scheduling",
    "realtime": "Periodic real-time scheduling",
    "sequencing": "Job sequencing with deadlines",
    "multiprocessor": "Identical multiprocessors",
    "cloud": "Heterogeneous machines (cloud / grid)",
    "workflow": "Workflows & task graphs",
    "resource": "Resource allocation",
    "energy": "Energy-aware scheduling",
}


def get(key: str) -> AlgorithmSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown algorithm {key!r}; choose from {', '.join(REGISTRY)}") from None


def by_family(family: str) -> list[AlgorithmSpec]:
    return [s for s in _SPECS if s.family == family]


def keys_for_kb(kb_name: str) -> list[str]:
    return [s.key for s in _SPECS if s.kb == kb_name]


__all__ = ["AlgorithmSpec", "REGISTRY", "FAMILIES", "get", "by_family", "keys_for_kb"]
