"""Experiment harness: schedule quality under varied workloads, and empirical scaling.

Two kinds of experiment:

* :func:`run_benchmark` runs CPU (or multiprocessor) algorithms on every workload
  shape × size × seed and records every metric from
  :func:`schedlab.metrics.summarize`, plus how long the algorithm itself took.
* :func:`scaling_study` times any registered algorithm at growing input sizes.
  :func:`fit_complexity` then fits ``time ≈ c·n^k`` on a log-log scale, so the
  measured exponent ``k`` can be set beside the theoretical Big-O.
"""

from __future__ import annotations

import csv
import io
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any

from . import workloads as wl
from .algorithms import REGISTRY, get
from .algorithms.cpu import fcfs
from .metrics import summarize

Progress = Callable[[int, int], None] | None


@dataclass(frozen=True)
class ComplexityFit:
    exponent: float
    coefficient: float
    r2: float

    @property
    def label(self) -> str:
        return f"≈ O(n^{self.exponent:.2f})"


def fit_complexity(ns: Iterable[float], times: Iterable[float]) -> ComplexityFit:
    """Least-squares fit of log(time) = k·log(n) + log(c)."""
    pairs = [(n, t) for n, t in zip(ns, times) if n > 0 and t > 0]
    if len({n for n, _ in pairs}) < 2:
        raise ValueError("need timings for at least two distinct sizes")
    xs = [math.log(n) for n, _ in pairs]
    ys = [math.log(t) for _, t in pairs]
    slope, intercept = statistics.linear_regression(xs, ys)
    r = statistics.correlation(xs, ys) if len(set(ys)) > 1 else 1.0
    return ComplexityFit(slope, math.exp(intercept), r * r)


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def run_benchmark(
    algorithms: list[str],
    workloads: list[str],
    sizes: list[int],
    seeds: list[int],
    params: dict[str, dict] | None = None,
    context_switch: float = 0.0,
    m: int = 4,
    progress: Progress = None,
) -> list[dict]:
    """One row per (algorithm, workload, n, seed) with every summary metric."""
    specs = [get(k) for k in algorithms]
    bad = [s.key for s in specs if s.family not in ("cpu", "multiprocessor")]
    if bad:
        raise ValueError(f"run_benchmark handles cpu/multiprocessor algorithms, not {bad}")
    params = params or {}
    total = len(workloads) * len(sizes) * len(seeds) * len(specs)
    rows, done = [], 0
    for workload in workloads:
        for n in sizes:
            for seed in seeds:
                procs = wl.generate(workload, n, seed)
                for spec in specs:
                    kwargs = {**spec.params, **params.get(spec.key, {}), "m": m}
                    kwargs["context_switch"] = context_switch
                    schedule, ms = _timed(partial(spec.call, procs, **kwargs))
                    rows.append(
                        {
                            "algorithm": spec.name,
                            "key": spec.key,
                            "workload": workload,
                            "n": n,
                            "seed": seed,
                            "runtime_ms": ms,
                            **summarize(schedule),
                        }
                    )
                    done += 1
                    if progress:
                        progress(done, total)
    return rows


def aggregate(rows: list[dict], metric: str, by: tuple[str, ...] = ("algorithm", "workload")) -> list[dict]:
    """Mean and standard deviation of ``metric`` over the rows sharing ``by``."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r[metric])
    out = []
    for key, values in groups.items():
        out.append(
            {
                **dict(zip(by, key)),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "runs": len(values),
            }
        )
    return out


def leaderboard(rows: list[dict], metric: str, lower_is_better: bool = True) -> list[dict]:
    """Rank algorithms by ``metric`` averaged over everything in ``rows``."""
    agg = aggregate(rows, metric, by=("algorithm",))
    return sorted(agg, key=lambda r: r["mean"], reverse=not lower_is_better)


def _bankers_worst_case(n: int, m: int = 4) -> tuple[list, list, list]:
    """Only the *last* unfinished process can proceed on each pass, so the safety
    check scans all remaining processes n times: the O(n²·m) worst case."""
    allocation = [[1] * m for _ in range(n)]
    max_claim = [[1 + (n - i)] * m for i in range(n)]
    return [1] * m, max_claim, allocation


def make_instance(key: str, n: int, seed: int = 0, case: str = "typical") -> tuple[tuple, dict]:
    """Build a size-``n`` input suited to the algorithm's family.

    ``case="worst"`` swaps in adversarial inputs where they differ from typical
    ones: every process ready at once (HRRN scans an n-long queue per decision)
    and a Banker's state that is only safe in reverse order.
    """
    if case not in ("typical", "worst"):
        raise ValueError("case must be 'typical' or 'worst'")
    spec = get(key)
    fam = spec.family
    if fam == "cpu":
        procs = wl.batch(n, seed) if case == "worst" else wl.poisson(n, seed)
        return (procs,), dict(spec.params)
    if fam == "multiprocessor":
        return (wl.heavy_tailed(n, seed),), {**spec.params, "m": 8}
    if fam == "realtime":
        return (wl.periodic_taskset(n, 0.8, seed),), {}
    if fam == "sequencing":
        return (wl.job_set(n, seed),), {}
    if fam == "cloud":
        return (wl.etc_matrix(n, 8, seed),), {}
    if fam == "workflow":
        extra = {"samples": 50} if key == "monte_carlo" else {}
        return (wl.random_dag(n, 4, seed),), extra
    if key == "bankers":
        return (_bankers_worst_case(n) if case == "worst" else wl.bankers_state(n, 4, seed)), {}
    if key == "max_min_fair":
        demands = [float(j.profit) for j in wl.job_set(n, seed)]
        return (0.6 * sum(demands), demands), {}
    if key == "dvfs":
        return (wl.periodic_taskset(n, 0.5, seed),), {}
    if key == "sleep":
        return (fcfs(wl.uniform(n, seed, load=0.5)),), {}
    raise KeyError(f"no instance builder for {key!r}")


def scaling_study(
    keys: list[str],
    sizes: list[int],
    seed: int = 0,
    repeats: int = 3,
    case: str = "typical",
    progress: Progress = None,
) -> list[dict]:
    """Time each algorithm at each size; keeps the *minimum* of ``repeats`` runs,
    which is the estimate least disturbed by other activity on the machine."""
    rows, done, total = [], 0, len(keys) * len(sizes)
    for key in keys:
        spec = get(key)
        for n in sizes:
            args, kwargs = make_instance(key, n, seed, case)
            run = partial(spec.call, *args, **kwargs)
            best = min(_timed(run)[1] for _ in range(repeats))
            rows.append(
                {"key": key, "algorithm": spec.name, "family": spec.family, "n": n,
                 "runtime_ms": best, "theory": spec.complexity, "case": case}
            )
            done += 1
            if progress:
                progress(done, total)
    return rows


def fit_all(rows: list[dict]) -> dict[str, ComplexityFit]:
    by_key: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        by_key[r["key"]].append((r["n"], r["runtime_ms"]))
    fits = {}
    for key, pts in by_key.items():
        try:
            fits[key] = fit_complexity([n for n, _ in pts], [t for _, t in pts])
        except ValueError:
            continue
    return fits


def to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    fields = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


__all__ = [
    "ComplexityFit", "fit_complexity", "run_benchmark", "aggregate", "leaderboard",
    "make_instance", "scaling_study", "fit_all", "to_csv", "REGISTRY",
]
