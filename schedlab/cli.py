"""Command-line interface: ``python -m schedlab <command>``.

Commands::

    list        show every implemented algorithm and its complexity
    recommend   explainable recommendations for a scheduling problem (+ --evaluate)
    general     advisor for classic (non-scheduling) algorithm problems
    simulate    run CPU schedulers on a workload: metrics table + ASCII Gantt chart
    realtime    schedulability analysis + simulation for a periodic task set
    benchmark   compare algorithms across workload shapes, sizes and seeds
    scaling     measure empirical runtime growth versus theoretical Big-O
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from collections.abc import Sequence

from . import benchmark as bm
from . import workloads as wl
from .algorithms import FAMILIES, REGISTRY, by_family, get, realtime
from .metrics import METRICS, summarize
from .models import CONTEXT_SWITCH, Schedule, task_of
from .recommender import GeneralAlgorithmAdvisor, evaluate_on_workload, parse_description, recommend

CPU_KEYS = [s.key for s in by_family("cpu")]


def _table(rows: list[dict], columns: list[str], headers: list[str] | None = None) -> str:
    headers = headers or columns

    def fmt(v) -> str:
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    cells = [[fmt(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(h), *(len(row[i]) for row in cells)) if cells else len(h)
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in cells]
    return "\n".join([line, sep, *body])


def ascii_gantt(schedule: Schedule, width: int = 64, group=task_of) -> str:
    """One row per process (or task); '█' where it runs, '·' where the CPU switches."""
    if not schedule.slices:
        return "(empty schedule)"
    end = schedule.makespan
    scale = width / end if end else 1.0
    rows: dict[str, list[str]] = {}
    for s in sorted(schedule.slices, key=lambda s: s.start):
        label = "cs" if s.pid == CONTEXT_SWITCH else group(s.pid)
        if schedule.cpus > 1:
            label = f"CPU{s.cpu}"
        bar = rows.setdefault(label, [" "] * width)
        a, b = int(s.start * scale), max(int(s.start * scale) + 1, int(s.end * scale))
        ch = "·" if s.pid == CONTEXT_SWITCH else ("█" if schedule.cpus == 1 else s.pid[-1])
        for i in range(a, min(b, width)):
            bar[i] = ch
    pad = max(len(k) for k in rows)
    lines = [f"{k.rjust(pad)} |{''.join(v)}|" for k, v in rows.items()]
    lines.append(f"{' ' * pad}  0{str(round(end, 1)).rjust(width - 1)}")
    return "\n".join(lines)


def cmd_list(args) -> int:
    for fam, title in FAMILIES.items():
        print(f"\n{title}")
        for s in by_family(fam):
            params = ", ".join(f"{k}={v}" for k, v in s.params.items())
            print(f"  {s.key:14s} {s.name:26s} {s.complexity:24s} {params}")
    return 0


def cmd_recommend(args) -> int:
    text = " ".join(args.text) or input("Describe your scheduling problem: ")
    profile = parse_description(text)
    recs = recommend(profile, args.top)
    print(f"\nUnderstood: domains={profile.domains or ['(none)']}, "
          f"objectives={profile.effective_objectives()}, features={sorted(profile.flags) or '-'}, "
          f"run times known={profile.bursts_known}, preemptive={profile.preemptive}")
    if not recs:
        print("\nNo scheduling recommendation found. Mention the resources (CPU, machines, cloud, "
              "project tasks), constraints (deadlines, priorities, dependencies) and goal.\n"
              "For classic algorithm questions try:  python -m schedlab general")
        return 1
    print("\n===== RECOMMENDED SCHEDULING ALGORITHMS =====")
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. {rec.name}   match {rec.match:.0%}   ({rec.info.time})")
        for sign, why in rec.reasons:
            print(f"   {'✔' if sign == '+' else '✘'} {why}")
        if args.verbose:
            print(textwrap.indent(rec.info.as_text(), "     "))
        elif rec.info.implemented:
            print(f"   ▶ runnable: {', '.join(rec.info.registry_keys)}")
    if args.evaluate:
        ev = evaluate_on_workload(recs, profile, n=args.n, seed=args.seed)
        if ev is None:
            print("\n(no simulated comparison available for this family)")
        else:
            label, lower = METRICS[ev.metric]
            print(f"\n===== DATA-BACKED CHECK: {label} on '{ev.workload}' "
                  f"({'lower' if lower else 'higher'} is better) =====")
            print(_table(ev.rows, ["algorithm", "value", "avg_waiting", "avg_response", "makespan"]))
            print(f"\nWinner on this workload: {ev.winner['algorithm']}")
    return 0


def cmd_general(args) -> int:
    advisor = GeneralAlgorithmAdvisor()
    if not args.text:
        advisor.run_interactive()
        return 0
    recs = advisor.recommend(" ".join(args.text))
    if not recs:
        print("No specific algorithm recommendation found. Please provide more details.")
        return 1
    for topic, algo in recs:
        print(f"- {algo.name} [{algo.complexity}] ({topic})\n  {algo.description}\n")
    return 0


def cmd_simulate(args) -> int:
    procs = wl.generate(args.workload, args.n, args.seed)
    keys = args.algos.split(",") if args.algos else CPU_KEYS
    print(f"Workload '{args.workload}': {args.n} processes, seed {args.seed}. "
          f"{wl.WORKLOAD_DESCRIPTIONS[args.workload]}")
    if args.show_input:
        print(_table([p.__dict__ for p in procs], ["pid", "arrival", "burst", "priority", "deadline"]))
    rows = []
    for key in keys:
        spec = get(key)
        params = {**spec.params, "quantum": args.quantum, "aging": args.aging}
        schedule = spec.call(procs, **params, context_switch=args.cs)
        rows.append({"algorithm": schedule.algorithm, **summarize(schedule)})
        if args.gantt:
            print(f"\n{schedule.algorithm}\n{ascii_gantt(schedule, group=lambda p: p)}")
    cols = ["algorithm", "avg_waiting", "avg_turnaround", "avg_response", "p95_turnaround",
            "throughput", "cpu_utilization", "context_switches", "fairness", "deadline_miss_ratio"]
    print()
    print(_table(sorted(rows, key=lambda r: r[args.sort], reverse=not METRICS[args.sort][1]), cols,
                 ["algorithm", "avg_wait", "avg_tat", "avg_resp", "p95_tat", "thruput", "util",
                  "switches", "fairness", "miss%"]))
    return 0


def cmd_realtime(args) -> int:
    periods = wl.NON_HARMONIC_PERIODS if args.non_harmonic else wl.HARMONIC_PERIODS
    tasks = wl.periodic_taskset(args.n, args.utilization, args.seed, args.constrained, periods)
    print(_table([{"task": t.name, "C": t.C, "T": t.T, "D": t.deadline, "U": t.utilization}
                  for t in tasks], ["task", "C", "T", "D", "U"]))
    rep = realtime.schedulability_report(tasks)
    print(f"\nU = {rep['utilization']:.3f}   Liu-Layland bound = {rep['liu_layland_bound']:.3f}   "
          f"hyperperiod = {rep['hyperperiod']:g}")
    print(f"RMS: LL test {'pass' if rep['rms_ll_pass'] else 'inconclusive'}, "
          f"exact RTA {'SCHEDULABLE' if rep['rms_schedulable'] else 'NOT schedulable'} {rep['rms_rta']}")
    print(f"DMS: exact RTA {'SCHEDULABLE' if rep['dms_schedulable'] else 'NOT schedulable'}")
    print(f"EDF: {'SCHEDULABLE' if rep['edf_schedulable'] else 'NOT schedulable'} (processor demand test)")
    for fn in (realtime.rate_monotonic, realtime.deadline_monotonic, realtime.edf_periodic):
        s = fn(tasks)
        m = summarize(s)
        print(f"\n{s.algorithm}: miss ratio {m['deadline_miss_ratio']:.1%}, max lateness "
              f"{m['max_lateness']:.1f}, preemptions {s.meta['preemptions']}")
        if args.gantt:
            print(ascii_gantt(s))
    return 0


def cmd_benchmark(args) -> int:
    keys = args.algos.split(",") if args.algos else CPU_KEYS
    loads = args.workloads.split(",") if args.workloads else list(wl.WORKLOADS)
    sizes = [int(x) for x in args.sizes.split(",")]
    seeds = list(range(args.seeds))
    rows = bm.run_benchmark(keys, loads, sizes, seeds, context_switch=args.cs)
    label, lower = METRICS[args.metric]
    print(f"{label} (mean over {len(seeds)} seeds × sizes {sizes}; "
          f"{'lower' if lower else 'higher'} is better)\n")
    agg = bm.aggregate(rows, args.metric)
    algos = list(dict.fromkeys(r["algorithm"] for r in rows))
    table = []
    for a in algos:
        row = {"algorithm": a}
        for w in loads:
            row[w] = next(x["mean"] for x in agg if x["algorithm"] == a and x["workload"] == w)
        table.append(row)
    print(_table(table, ["algorithm", *loads]))
    print("\nOverall leaderboard:")
    for i, r in enumerate(bm.leaderboard(rows, args.metric, lower), 1):
        print(f"  {i}. {r['algorithm']:24s} {r['mean']:.3f}")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            fh.write(bm.to_csv(rows))
        print(f"\nWrote {len(rows)} rows to {args.csv}")
    return 0


def cmd_scaling(args) -> int:
    keys = args.algos.split(",") if args.algos else list(REGISTRY)
    sizes = [int(x) for x in args.sizes.split(",")]
    rows = bm.scaling_study(keys, sizes, repeats=args.repeats, case=args.case)
    fits = bm.fit_all(rows)
    out = []
    for key in keys:
        times = {r["n"]: r["runtime_ms"] for r in rows if r["key"] == key}
        fit = fits.get(key)
        out.append({"algorithm": REGISTRY[key].name, "theory": REGISTRY[key].complexity,
                    "empirical": fit.label if fit else "n/a", "r2": fit.r2 if fit else 0.0,
                    **{f"n={n} (ms)": times[n] for n in sizes}})
    print(_table(out, list(out[0])))
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            fh.write(bm.to_csv(rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="schedlab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list implemented algorithms").set_defaults(fn=cmd_list)

    r = sub.add_parser("recommend", help="recommend scheduling algorithms for a problem")
    r.add_argument("text", nargs="*", help="problem description (prompted if omitted)")
    r.add_argument("--top", type=int, default=5)
    r.add_argument("--evaluate", action="store_true", help="also compare the picks on a simulated workload")
    r.add_argument("-n", type=int, default=40, help="workload size for --evaluate")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("-v", "--verbose", action="store_true", help="print full algorithm descriptions")
    r.set_defaults(fn=cmd_recommend)

    g = sub.add_parser("general", help="advisor for classic algorithm problems (sorting, graphs, ...)")
    g.add_argument("text", nargs="*")
    g.set_defaults(fn=cmd_general)

    s = sub.add_parser("simulate", help="simulate CPU schedulers on a generated workload")
    s.add_argument("--workload", choices=list(wl.WORKLOADS), default="poisson")
    s.add_argument("-n", type=int, default=12)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--algos", help=f"comma-separated keys (default: {','.join(CPU_KEYS)})")
    s.add_argument("--quantum", type=float, default=4.0)
    s.add_argument("--aging", type=float, default=0.0)
    s.add_argument("--cs", type=float, default=0.0, help="context-switch cost")
    s.add_argument("--sort", choices=list(METRICS), default="avg_waiting")
    s.add_argument("--gantt", action="store_true", help="print ASCII Gantt charts")
    s.add_argument("--show-input", action="store_true")
    s.set_defaults(fn=cmd_simulate)

    rt = sub.add_parser("realtime", help="analyse and simulate a random periodic task set")
    rt.add_argument("-n", type=int, default=4)
    rt.add_argument("-u", "--utilization", type=float, default=0.85)
    rt.add_argument("--seed", type=int, default=0)
    rt.add_argument("--constrained", action="store_true", help="deadlines shorter than periods")
    rt.add_argument("--non-harmonic", action="store_true", help="use non-harmonic periods")
    rt.add_argument("--gantt", action="store_true")
    rt.set_defaults(fn=cmd_realtime)

    b = sub.add_parser("benchmark", help="compare algorithms across varied workloads")
    b.add_argument("--algos")
    b.add_argument("--workloads", help=f"comma-separated (default: {','.join(wl.WORKLOADS)})")
    b.add_argument("--sizes", default="50,100")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--cs", type=float, default=0.0)
    b.add_argument("--metric", choices=list(METRICS), default="avg_waiting")
    b.add_argument("--csv", help="write raw rows to this CSV file")
    b.set_defaults(fn=cmd_benchmark)

    sc = sub.add_parser("scaling", help="empirical runtime scaling vs theoretical complexity")
    sc.add_argument("--algos")
    sc.add_argument("--sizes", default="250,500,1000,2000")
    sc.add_argument("--repeats", type=int, default=3)
    sc.add_argument("--case", choices=["typical", "worst"], default="typical")
    sc.add_argument("--csv")
    sc.set_defaults(fn=cmd_scaling)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # output piped into e.g. `head`, which stopped reading
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
