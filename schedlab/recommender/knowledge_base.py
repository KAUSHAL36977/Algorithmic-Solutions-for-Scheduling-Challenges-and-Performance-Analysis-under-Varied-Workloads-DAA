"""Structured knowledge base of scheduling algorithms.

Migrated from the original ``DAA101.py`` dictionary. Each entry adds machine-readable
``traits``, which the recommender (:mod:`schedlab.recommender.engine`) scores against a
problem profile. Several complexity claims were corrected during the migration, for
example Round Robin's "O(n)", SJF's "O(n²)" and Banker's "O(n+m)" space.

Trait keys:

* ``domains``: domain → relevance (1-3)
* ``objectives``: objective → how well the algorithm serves it (0-3)
* ``handles``: problem features it copes with well (earn a bonus when present)
* ``requires``: features it depends on (penalised when absent)
* ``risks``: feature/objective → the reason it is a poor fit when present
* ``preemptive``: True / False / None (not applicable)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..algorithms import keys_for_kb


@dataclass(frozen=True)
class AlgorithmInfo:
    name: str
    category: str
    summary: str
    time: str
    space: str
    params: tuple[str, ...]
    pros: tuple[str, ...]
    cons: tuple[str, ...]
    applications: str
    traits: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def registry_keys(self) -> list[str]:
        """Keys of the runnable implementations in :data:`schedlab.algorithms.REGISTRY`."""
        return keys_for_kb(self.name)

    @property
    def implemented(self) -> bool:
        return bool(self.registry_keys)

    def as_text(self) -> str:
        lines = [
            self.summary,
            "",
            f"Time complexity : {self.time}",
            f"Space complexity: {self.space}",
            "Key parameters  : " + ", ".join(self.params),
            "Advantages      : " + "; ".join(self.pros),
            "Limitations     : " + "; ".join(self.cons),
            f"Applications    : {self.applications}",
        ]
        if self.implemented:
            lines.append("Runnable in schedlab as: " + ", ".join(self.registry_keys))
        return "\n".join(lines)


def _info(name, category, summary, time, space, params, pros, cons, applications, **traits):
    return AlgorithmInfo(
        name, category, summary, time, space, tuple(params), tuple(pros), tuple(cons),
        applications, traits,
    )


KNOWLEDGE_BASE: list[AlgorithmInfo] = [
    # ------------------------------------------------------------------ task scheduling
    _info(
        "First-Come, First-Served (FCFS)", "task scheduling",
        "Runs processes to completion in arrival order using a single FIFO queue.",
        "O(n log n) to order arrivals, then O(1) per dispatch", "O(n)",
        ["arrival order"],
        ["Simplest possible policy", "No starvation", "Minimal scheduling overhead"],
        ["Convoy effect: short jobs wait behind long ones", "Poor average waiting and response time"],
        "Batch systems, print queues, FIFO message processing, baselines for comparison.",
        domains={"cpu": 2, "batch": 2},
        objectives={"fairness": 1, "throughput": 1, "utilization": 1},
        handles=["bursts_unknown"],
        risks={"interactive": "convoy effect makes interactive jobs wait behind long ones",
               "response": "one long job delays the response of everything behind it"},
        preemptive=False,
    ),
    _info(
        "Round Robin (RR)", "task scheduling",
        "Gives each ready process a fixed time quantum in cyclic order, preempting it "
        "when the quantum expires.",
        "O(n log n + B/q): O(1) per quantum, B = total burst time, q = quantum",
        "O(n) for the ready queue",
        ["time quantum q", "context-switch cost"],
        ["Fair allocation of CPU time", "Low response time for short and interactive processes",
         "Needs no knowledge of burst lengths"],
        ["Higher average waiting time than SJF", "Very sensitive to the quantum",
         "Context-switch overhead grows as q shrinks"],
        "Time-sharing operating systems, web servers handling many client requests, network packet schedulers.",
        domains={"cpu": 3},
        objectives={"response": 3, "fairness": 3, "avg_wait": 1},
        handles=["interactive", "bursts_unknown", "preemptive"],
        risks={"avg_wait": "rotating equal-length jobs inflates average waiting time"},
        preemptive=True,
    ),
    _info(
        "Shortest Job First (SJF)", "task scheduling",
        "Picks the ready process with the smallest burst time and runs it to completion.",
        "O(n log n) with a min-heap ready queue (O(n²) with a naive scan)", "O(n)",
        ["burst-time estimates"],
        ["Provably minimises average waiting time when all jobs are ready together",
         "Maximises throughput"],
        ["Starvation of long processes", "Requires burst times in advance",
         "Non-preemptive: a long job blocks later short ones"],
        "Batch processing, offline job scheduling where run times are known or well estimated.",
        domains={"cpu": 2, "batch": 3},
        objectives={"avg_wait": 3, "throughput": 3},
        requires=["bursts_known"],
        risks={"fairness": "long jobs can starve", "interactive": "no preemption for newly arrived short jobs"},
        preemptive=False,
    ),
    _info(
        "Shortest Remaining Time First (SRTF)", "task scheduling",
        "Preemptive SJF: a newly arrived process preempts the running one if it has less remaining work.",
        "O(n log n) with a min-heap keyed on remaining time", "O(n)",
        ["burst-time estimates"],
        ["Optimal average waiting time among all policies (for known bursts)",
         "Short jobs get through very quickly"],
        ["Starves long jobs under steady short-job arrivals", "Needs remaining-time estimates",
         "More context switches than SJF"],
        "Schedulers with good runtime prediction, SRPT scheduling in web servers and networks.",
        domains={"cpu": 3, "batch": 2},
        objectives={"avg_wait": 3, "response": 2, "throughput": 3},
        requires=["bursts_known"],
        handles=["preemptive"],
        risks={"fairness": "long jobs can starve under steady short arrivals"},
        preemptive=True,
    ),
    _info(
        "Priority Scheduling", "task scheduling",
        "Runs the process with the highest priority (lowest number) first; available preemptive "
        "or non-preemptive, with optional aging to prevent starvation.",
        "O(n log n) with a priority queue (aging stays O(log n): uniform linear aging preserves heap order)",
        "O(n)",
        ["priority values", "aging rate", "preemptive or not"],
        ["Important tasks run first", "Flexible: priorities can encode any policy",
         "Aging removes starvation cheaply"],
        ["Starvation of low-priority tasks without aging", "Priority inversion", "Priorities must be assigned well"],
        "Operating systems, industrial control, mission-critical systems with task classes.",
        domains={"cpu": 3, "realtime": 1},
        objectives={"response": 1, "throughput": 1},
        handles=["priorities", "preemptive"],
        risks={"fairness": "low-priority work starves unless aging is enabled"},
        preemptive=None,
    ),
    _info(
        "Highest Response Ratio Next (HRRN)", "task scheduling",
        "Non-preemptive; picks the process with the highest (waiting + burst) / burst, "
        "favouring short jobs while ageing long ones.",
        "O(n) per decision, O(n²) total (ratios change continuously, so no static heap)", "O(n)",
        ["burst-time estimates"],
        ["Near-SJF waiting time without starvation", "Built-in aging"],
        ["Requires burst estimates", "O(n) scan per decision", "Non-preemptive"],
        "Batch systems wanting SJF-like efficiency with fairness guarantees.",
        domains={"cpu": 2, "batch": 3},
        objectives={"avg_wait": 2, "fairness": 2, "throughput": 2},
        requires=["bursts_known"],
        risks={"interactive": "non-preemptive: short interactive jobs can wait behind long ones"},
        preemptive=False,
    ),
    _info(
        "Multi-Level Feedback Queue (MLFQ)", "task scheduling",
        "Several RR queues with growing quanta; jobs that use their whole quantum are demoted, so "
        "interactive jobs stay on top without knowing burst lengths in advance.",
        "O(n log n + k·B/q) for k levels", "O(n + k)",
        ["number of levels", "quantum per level", "boost interval"],
        ["Approximates SRTF without burst knowledge", "Excellent interactive response",
         "Adapts to job behaviour"],
        ["Many knobs to tune", "Can be gamed by yielding just before the quantum ends",
         "Long CPU-bound jobs sink to the bottom"],
        "General-purpose OS schedulers (BSD, Windows, macOS lineage), mixed interactive and batch workloads.",
        domains={"cpu": 3},
        objectives={"response": 3, "avg_wait": 2, "fairness": 2},
        handles=["interactive", "bursts_unknown", "preemptive", "uncertainty"],
        preemptive=True,
    ),
    # ------------------------------------------------------------------ job sequencing
    _info(
        "Greedy Job Sequencing", "job sequencing",
        "Unit-time jobs with deadlines and profits: take jobs by decreasing profit and place each in "
        "the latest free slot before its deadline.",
        "O(n log n) with a disjoint-set of free slots (O(n·d) with a linear slot scan)",
        "O(n) (slots capped at n)",
        ["deadlines", "profits"],
        ["Optimal for unit-time jobs (a matroid greedy)", "Very fast with DSU", "Simple to explain"],
        ["Assumes unit processing times", "No dependencies", "Single machine"],
        "Freelance/contract selection, ad-slot allocation, manufacturing orders with due dates and value.",
        domains={"sequencing": 3, "batch": 1},
        objectives={"profit": 3, "deadlines": 2},
        handles=["deadlines"],
        preemptive=False,
    ),
    _info(
        "Earliest Deadline First (EDF)", "job sequencing",
        "Dynamic-priority scheduling: always run the ready job with the nearest absolute deadline.",
        "O(log n) per event with a heap; O(n log n) overall", "O(n)",
        ["deadlines", "execution times"],
        ["Optimal on one processor: meets all deadlines whenever any schedule can",
         "Schedulable up to 100% utilisation (implicit deadlines)", "Works for periodic and one-shot jobs"],
        ["Unpredictable under overload (domino effect of misses)",
         "Dynamic priorities are harder to certify than fixed ones"],
        "Real-time operating systems (Linux SCHED_DEADLINE), multimedia, telecom and embedded systems.",
        domains={"realtime": 3, "sequencing": 2, "cpu": 2},
        objectives={"deadlines": 3, "utilization": 3},
        handles=["deadlines", "periodic", "preemptive"],
        preemptive=True,
    ),
    # ------------------------------------------------------------------ resource allocation
    _info(
        "Banker's Algorithm", "resource allocation",
        "Deadlock avoidance: grant a request only if the system stays in a safe state, i.e. some order "
        "still lets every process get its maximum claim and finish.",
        "O(n²·m) per safety check (n processes, m resource types)", "O(n·m)",
        ["available vector", "maximum claims", "current allocation"],
        ["Guarantees deadlock freedom", "Allows more concurrency than static prevention"],
        ["Needs maximum claims in advance", "Conservative: may refuse safe-in-practice requests",
         "Check cost grows quadratically"],
        "OS resource managers, database lock managers, cloud quota admission.",
        domains={"resource": 3},
        objectives={"safety": 3},
        handles=["multiple_resources"],
        preemptive=None,
    ),
    _info(
        "Max-Min Fair Allocation", "resource allocation",
        "Water-filling: nobody receives more than they ask for, and the smallest allocation is as large "
        "as possible. (The original DAA101 entry called this 'Min-Max'.)",
        "O(n log n) (sort demands, one pass)", "O(n)",
        ["capacity", "demands", "optional weights"],
        ["Provably fair (max-min / weighted fairness)", "No starvation", "Work-conserving"],
        ["Ignores differing job values", "Single resource (use DRF for multi-resource)"],
        "Network bandwidth sharing, cluster fair-share schedulers (YARN, Mesos), multi-tenant cloud quotas.",
        domains={"resource": 3, "cloud": 1},
        objectives={"fairness": 3, "utilization": 2},
        handles=["priorities"],
        preemptive=None,
    ),
    # ------------------------------------------------------------------ workflow scheduling
    _info(
        "Critical Path Method (CPM)", "workflow scheduling",
        "Forward and backward passes over a task DAG give earliest/latest start times and slack; "
        "zero-slack tasks form the critical path that fixes the minimum project duration.",
        "O(V + E)", "O(V + E)",
        ["task durations", "precedence constraints"],
        ["Finds the minimum completion time", "Shows which tasks can slip (slack)", "Linear time"],
        ["Assumes unlimited resources", "Deterministic durations"],
        "Project management, construction, build systems, CI pipelines.",
        domains={"workflow": 3},
        objectives={"makespan": 2},
        requires=["dependencies"],
        handles=["dependencies"],
        risks={"uncertainty": "deterministic durations hide schedule risk (use Monte Carlo)"},
        preemptive=None,
    ),
    _info(
        "Heterogeneous Earliest Finish Time (HEFT)", "workflow scheduling",
        "List scheduler for DAGs on heterogeneous processors: rank tasks by upward rank, then put each "
        "on the processor with the earliest finish time, inserting into idle gaps.",
        "O(V²·P)", "O(V + E + P)",
        ["per-processor execution costs", "communication costs", "dependencies"],
        ["Strong makespan in practice", "Accounts for computation and communication",
         "Insertion policy fills idle gaps"],
        ["Heuristic, not optimal", "Static: needs cost estimates in advance"],
        "Scientific workflows, grid and cloud computing, heterogeneous CPU/GPU systems.",
        domains={"workflow": 3, "cloud": 2, "multiprocessor": 2},
        objectives={"makespan": 3, "utilization": 2},
        handles=["dependencies", "heterogeneous", "multiple_processors"],
        requires=["bursts_known"],
        preemptive=False,
    ),
    # ------------------------------------------------------------------ real-time
    _info(
        "Rate Monotonic Scheduling (RMS)", "real-time scheduling",
        "Fixed-priority preemptive scheduling of periodic tasks: shorter period gives higher priority.",
        "O(n log n) to assign priorities; exact test via response-time analysis", "O(n)",
        ["periods", "worst-case execution times"],
        ["Optimal among fixed-priority policies (implicit deadlines)",
         "Predictable, easy to certify", "Supported by every RTOS"],
        ["Guaranteed only up to n(2^(1/n) - 1) ≈ 69% utilisation by the Liu-Layland bound",
         "Not optimal when deadlines differ from periods"],
        "Hard real-time embedded, automotive, avionics control loops.",
        domains={"realtime": 3},
        objectives={"deadlines": 2, "predictability": 3},
        requires=["periodic"],
        handles=["deadlines", "periodic", "preemptive", "priorities"],
        preemptive=True,
    ),
    _info(
        "Deadline Monotonic Scheduling (DMS)", "real-time scheduling",
        "Fixed priorities by relative deadline (shorter deadline gives higher priority).",
        "O(n log n); exact test via response-time analysis", "O(n)",
        ["relative deadlines", "worst-case execution times"],
        ["Optimal fixed-priority order when deadlines ≤ periods", "Static priorities are easy to certify"],
        ["Not optimal for arbitrary deadlines", "Needs response-time analysis"],
        "Hard real-time systems whose deadlines are shorter than their periods.",
        domains={"realtime": 3},
        objectives={"deadlines": 3, "predictability": 3},
        requires=["periodic"],
        handles=["deadlines", "periodic", "preemptive", "priorities"],
        preemptive=True,
    ),
    # ------------------------------------------------------------------ multiprocessor
    _info(
        "List Scheduling (Graham / LPT)", "multiprocessor scheduling",
        "Assign each job, in list order, to the processor that frees up first; LPT sorts jobs "
        "longest-first beforehand.",
        "O(n log m) with a heap of processors (+ O(n log n) sort for LPT)", "O(n + m)",
        ["job lengths", "number of processors"],
        ["Graham: (2 - 1/m)-approximation", "LPT: (4/3 - 1/(3m))-approximation", "Very fast"],
        ["Not optimal (the problem is NP-hard)", "Independent jobs only"],
        "Parallel batch jobs, compute clusters, load balancing, multi-core task pools.",
        domains={"multiprocessor": 3, "batch": 2},
        objectives={"makespan": 3, "utilization": 2},
        handles=["multiple_processors"],
        requires=["bursts_known"],
        preemptive=False,
    ),
    _info(
        "Work Stealing", "multiprocessor scheduling",
        "Each worker processes its own deque; an idle worker steals from the head of a random "
        "victim's deque.",
        "Expected T₁/P + O(T∞) for fork-join computations; simulation O(n log m + steals·m)",
        "O(n + m)",
        ["chunking strategy", "steal cost", "victim selection"],
        ["Decentralised dynamic load balancing", "Good cache locality", "Needs no job-length knowledge"],
        ["Steal synchronisation overhead", "Performance depends on task granularity"],
        "Cilk, Intel TBB, Java ForkJoinPool, Go and Tokio runtimes.",
        domains={"multiprocessor": 3},
        objectives={"utilization": 3, "makespan": 2, "adaptivity": 3},
        handles=["multiple_processors", "bursts_unknown", "uncertainty"],
        preemptive=False,
    ),
    # ------------------------------------------------------------------ stochastic
    _info(
        "Monte Carlo Scheduling", "stochastic scheduling",
        "Samples uncertain task durations many times and re-runs the critical-path pass to estimate "
        "the distribution of completion time and each task's criticality.",
        "O(S·(V + E)) for S samples", "O(V + E + S)",
        ["duration distributions", "number of samples"],
        ["Quantifies schedule risk (P90/P95 dates)", "Finds tasks that are *often* critical"],
        ["Estimates, not guarantees", "Needs plausible distributions"],
        "Project risk analysis, PERT, release planning, SLA estimation.",
        domains={"stochastic": 3, "workflow": 2},
        objectives={"risk": 3, "makespan": 1},
        handles=["uncertainty", "dependencies"],
        preemptive=None,
    ),
    _info(
        "Q-Learning for Dynamic Scheduling", "stochastic scheduling",
        "Reinforcement learning learns a dispatch policy from experience, trading off objectives via a "
        "reward signal.",
        "O(|A|) per decision after training; training needs many episodes", "O(|S|·|A|) Q-table",
        ["learning rate", "discount factor", "exploration rate", "state/action design"],
        ["Adapts to changing workloads", "Can optimise multiple objectives"],
        ["Needs a long training period", "State explosion", "Reward design is hard", "No guarantees"],
        "Adaptive cloud autoscaling, job-shop dispatching, network routing.",
        domains={"stochastic": 2, "cloud": 1, "cpu": 1},
        objectives={"adaptivity": 3},
        handles=["uncertainty", "bursts_unknown"],
        risks={"deadlines": "offers no hard deadline guarantees"},
        preemptive=None,
    ),
    # ------------------------------------------------------------------ cloud
    _info(
        "Min-Min Algorithm", "cloud scheduling",
        "Repeatedly maps the task with the smallest minimum completion time to the machine achieving it.",
        "O(n²·m)", "O(n·m) for the ETC matrix",
        ["expected time-to-compute (ETC) matrix", "machine ready times"],
        ["Finishes many small tasks early", "Good makespan for small-task-heavy mixes", "Simple"],
        ["Large tasks can be postponed indefinitely", "Needs accurate estimates", "Centralised"],
        "Cloud VM task mapping, grid meta-schedulers, heterogeneous clusters.",
        domains={"cloud": 3, "multiprocessor": 1},
        objectives={"makespan": 2, "avg_wait": 2, "throughput": 2},
        handles=["heterogeneous", "multiple_processors"],
        requires=["bursts_known"],
        risks={"fairness": "big tasks are delayed until the end"},
        preemptive=False,
    ),
    _info(
        "Max-Min Algorithm", "cloud scheduling",
        "Repeatedly maps the task with the *largest* minimum completion time first, so big tasks are "
        "placed early and small ones fill around them.",
        "O(n²·m)", "O(n·m) for the ETC matrix",
        ["expected time-to-compute (ETC) matrix", "machine ready times"],
        ["Better makespan when a few tasks are much larger", "Balances machines"],
        ["Small tasks wait", "Needs accurate estimates"],
        "Cloud resource allocation with mixed task sizes, grid computing.",
        domains={"cloud": 3, "multiprocessor": 1},
        objectives={"makespan": 3, "utilization": 2},
        handles=["heterogeneous", "multiple_processors"],
        requires=["bursts_known"],
        risks={"response": "small tasks are delayed behind big ones"},
        preemptive=False,
    ),
    # ------------------------------------------------------------------ dynamic / HPC
    _info(
        "Backfilling", "dynamic scheduling",
        "Lets smaller jobs jump ahead into holes in the schedule as long as they do not delay the "
        "reservation of the job at the head of the queue (EASY backfilling).",
        "O(n log n) per scheduling pass", "O(n)",
        ["node requirements", "runtime estimates", "queue priorities"],
        ["Big utilisation gains on clusters", "Reduces waiting time of small jobs",
         "Keeps priority order for the head job"],
        ["Relies on user runtime estimates (often wrong)", "Implementation complexity"],
        "HPC batch schedulers (Slurm, PBS, LSF), supercomputing centres.",
        domains={"hpc": 3, "batch": 2, "multiprocessor": 2},
        objectives={"utilization": 3, "avg_wait": 2},
        handles=["priorities", "multiple_processors"],
        requires=["bursts_known"],
        preemptive=False,
    ),
    _info(
        "Gang Scheduling", "dynamic scheduling",
        "Schedules all threads of a parallel job simultaneously across processors, time-slicing gangs.",
        "O(n·p·m) for n jobs, p time slots, m processors", "O(n·m)",
        ["gang groupings", "time-slice length", "migration cost"],
        ["Communicating threads never wait on descheduled peers", "Improves parallel application speed"],
        ["Fragmentation leaves processors idle", "Coordination overhead"],
        "Tightly coupled MPI jobs, HPC clusters, parallel database operators.",
        domains={"hpc": 3, "multiprocessor": 2},
        objectives={"response": 1, "throughput": 2},
        handles=["multiple_processors", "dependencies"],
        preemptive=True,
    ),
    # ------------------------------------------------------------------ energy-aware
    _info(
        "DVFS-based Scheduling", "energy-aware scheduling",
        "Dynamic Voltage and Frequency Scaling: run at the lowest speed that still meets deadlines. "
        "For EDF the energy-optimal constant speed is the lowest frequency f ≥ U.",
        "O(n + L) for n tasks and L frequency levels (static); YDS optimum is O(n³)", "O(n)",
        ["frequency levels", "power model (P ∝ f^α)", "deadlines"],
        ["Energy falls roughly with f²", "Deadlines still guaranteed", "Complements sleep states"],
        ["Needs hardware support", "Static leakage limits the savings", "Longer execution times"],
        "Mobile SoCs, battery-powered embedded devices, green data centres.",
        domains={"energy": 3, "realtime": 2},
        objectives={"energy": 3, "deadlines": 2},
        handles=["deadlines", "periodic"],
        preemptive=True,
    ),
    _info(
        "Sleep Scheduling", "energy-aware scheduling",
        "Puts idle resources into low-power states when the idle period exceeds the break-even time "
        "E_transition / (P_idle - P_sleep).",
        "O(g) for g idle gaps (offline break-even rule)", "O(g)",
        ["idle-period prediction", "transition energy", "wake-up latency"],
        ["Large savings for intermittent workloads", "Simple break-even rule is 2-competitive online"],
        ["Wake-up latency hurts responsiveness", "Mispredicted idle periods waste energy"],
        "Server farms, wireless sensor networks, IoT devices, laptops.",
        domains={"energy": 3},
        objectives={"energy": 2},
        risks={"interactive": "wake-up latency delays interactive requests",
               "response": "wake-up latency adds to response time"},
        preemptive=None,
    ),
]

BY_NAME: dict[str, AlgorithmInfo] = {a.name: a for a in KNOWLEDGE_BASE}
CATEGORIES: list[str] = list(dict.fromkeys(a.category for a in KNOWLEDGE_BASE))


def lookup(name: str) -> AlgorithmInfo:
    return BY_NAME[name]
