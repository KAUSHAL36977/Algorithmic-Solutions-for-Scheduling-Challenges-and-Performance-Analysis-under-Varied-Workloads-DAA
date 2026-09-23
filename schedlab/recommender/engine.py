"""Explainable scheduling-algorithm recommender.

Pipeline:

1. :func:`parse_description` turns free text into a :class:`ProblemProfile` using
   whole-word regular expressions with synonyms. The original version matched raw
   substrings, so "processor" also counted as "process"; whole-word matching fixes that.
2. :func:`recommend` scores every knowledge-base entry against the profile: domain
   fit, objective strength, features handled, unmet requirements and known risks.
   Each point comes with a human-readable reason.
3. :func:`evaluate_on_workload` runs the implemented candidates on a real
   workload and ranks them by the user's objective metric, turning the advice
   into a data-backed pick.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import workloads as wl
from ..algorithms import get
from ..metrics import METRICS, summarize
from ..models import Process
from .knowledge_base import KNOWLEDGE_BASE, AlgorithmInfo

DOMAINS = {
    "cpu": "CPU / OS process scheduling",
    "batch": "Batch jobs",
    "realtime": "Real-time / embedded",
    "sequencing": "Deadline & profit job sequencing",
    "multiprocessor": "Multiprocessor / parallel",
    "cloud": "Cloud / grid (heterogeneous machines)",
    "workflow": "Workflows & project DAGs",
    "resource": "Resource allocation & deadlock",
    "energy": "Energy-aware",
    "hpc": "HPC cluster batch queues",
    "stochastic": "Uncertain / stochastic",
}

OBJECTIVES = {
    "avg_wait": "Minimise average waiting / turnaround",
    "response": "Fast response for interactive work",
    "fairness": "Fairness / no starvation",
    "deadlines": "Meet deadlines",
    "makespan": "Finish everything soonest (makespan)",
    "throughput": "Maximise throughput",
    "utilization": "Keep resources busy (utilisation)",
    "profit": "Maximise profit / value",
    "energy": "Save energy",
    "safety": "Avoid deadlock (safe states)",
    "predictability": "Predictable, certifiable timing",
    "adaptivity": "Adapt to changing workloads",
    "risk": "Quantify schedule risk",
}

FLAGS = {
    "deadlines": "Tasks have deadlines",
    "periodic": "Tasks repeat periodically",
    "priorities": "Tasks have different priorities",
    "dependencies": "Tasks depend on each other (DAG)",
    "heterogeneous": "Machines have different speeds",
    "multiple_processors": "Several processors / machines",
    "multiple_resources": "Several resource types",
    "interactive": "Interactive / latency-sensitive",
    "uncertainty": "Durations are uncertain",
}

# Objective → metric used when evaluating on a workload.
OBJECTIVE_METRIC = {
    "avg_wait": "avg_waiting",
    "response": "avg_response",
    "fairness": "fairness",
    "deadlines": "deadline_miss_ratio",
    "makespan": "makespan",
    "throughput": "throughput",
    "utilization": "cpu_utilization",
    "predictability": "deadline_miss_ratio",
    "energy": "cpu_utilization",
    "adaptivity": "avg_turnaround",
    "risk": "p95_turnaround",
}

DEFAULT_OBJECTIVES = {
    "cpu": ["avg_wait", "response"],
    "batch": ["avg_wait", "throughput"],
    "realtime": ["deadlines"],
    "sequencing": ["profit"],
    "multiprocessor": ["makespan"],
    "cloud": ["makespan"],
    "workflow": ["makespan"],
    "resource": ["safety", "fairness"],
    "energy": ["energy"],
    "hpc": ["utilization"],
    "stochastic": ["risk"],
}


@dataclass
class ProblemProfile:
    domains: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)  # most important first
    flags: set[str] = field(default_factory=set)
    bursts_known: bool | None = None  # None = not stated
    preemptive: bool | None = None  # None = either is fine
    processors: int | None = None

    def effective_objectives(self) -> list[str]:
        if self.objectives:
            return self.objectives
        out: list[str] = []
        for d in self.domains or ["cpu"]:
            out += [o for o in DEFAULT_OBJECTIVES.get(d, []) if o not in out]
        return out

    def is_empty(self) -> bool:
        """True when nothing scheduling-related was recognised."""
        return not (
            self.domains or self.objectives or self.flags or self.processors
            or self.bursts_known is not None or self.preemptive is not None
        )

    def has(self, feature: str) -> bool:
        if feature == "bursts_known":
            return self.bursts_known is True
        if feature == "bursts_unknown":
            return self.bursts_known is False
        if feature == "preemptive":
            return self.preemptive is True
        return feature in self.flags


# ---------------------------------------------------------------------------
# Text → profile
# ---------------------------------------------------------------------------


def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


_DOMAIN_PATTERNS = {
    "cpu": _rx(r"cpu", r"process(?:es)?", r"operating systems?", r"os", r"threads?",
               r"time[- ]?sharing", r"context switch(?:es|ing)?", r"kernel", r"dispatch(?:er)?"),
    "batch": _rx(r"batch(?:es)?", r"offline", r"overnight", r"backlog", r"queued jobs?",
                 r"all (?:arrive|available) at once"),
    "realtime": _rx(r"real[- ]?time", r"rtos", r"embedded", r"control loops?", r"hard deadlines?",
                    r"sensors?", r"avionics", r"automotive", r"periodic", r"robot(?:s|ics)?"),
    "sequencing": _rx(r"job sequencing", r"sequenc(?:e|ing)", r"profits?", r"revenue",
                      r"unit[- ]time jobs?", r"order(?:ing)? of jobs"),
    "multiprocessor": _rx(r"multi[- ]?processors?", r"multi[- ]?core", r"cores", r"parallel",
                          r"processors", r"identical machines", r"workers", r"thread pools?",
                          r"load balanc(?:e|ing)"),
    "cloud": _rx(r"cloud", r"vms?", r"virtual machines?", r"data ?cent(?:er|re)s?", r"grid",
                 r"heterogeneous", r"servers", r"kubernetes", r"containers?", r"instances"),
    "workflow": _rx(r"workflows?", r"projects?", r"dags?", r"dependenc(?:y|ies)", r"depends? on",
                    r"precedence", r"prerequisites?", r"pipelines?", r"milestones?", r"task graphs?",
                    r"build steps?"),
    "resource": _rx(r"resource allocation", r"deadlocks?", r"bandwidth", r"allocat(?:e|ing|ion)",
                    r"fair share", r"quotas?", r"memory", r"locks?", r"semaphores?"),
    "energy": _rx(r"energy", r"power", r"battery", r"green", r"thermal", r"dvfs",
                  r"frequency scaling", r"sleep"),
    "hpc": _rx(r"hpc", r"supercomput(?:er|ing)", r"slurm", r"pbs", r"backfill(?:ing)?", r"gang",
               r"mpi", r"cluster jobs?", r"compute nodes?"),
    "stochastic": _rx(r"uncertain(?:ty)?", r"stochastic", r"probabilistic", r"random",
                      r"variab(?:le|ility)", r"unpredictable", r"risk", r"estimates? (?:vary|are rough)",
                      r"monte carlo"),
}

_FLAG_PATTERNS = {
    "deadlines": _rx(r"deadlines?", r"due dates?", r"slas?", r"on time", r"time limits?",
                     r"latency bounds?"),
    "periodic": _rx(r"periodic(?:ally)?", r"every \d+ ?(?:ms|milliseconds?|s|seconds?)",
                    r"sampling", r"cyclic", r"repeat(?:s|ing)?", r"frame rates?"),
    "priorities": _rx(r"priorit(?:y|ies|ise|ize)", r"urgent", r"critical tasks?", r"important",
                      r"vip", r"classes of"),
    "dependencies": _rx(r"dependenc(?:y|ies)", r"depends? on", r"precedence", r"prerequisites?",
                        r"dags?", r"workflows?", r"task graphs?", r"after .{1,30} finishes"),
    "heterogeneous": _rx(r"heterogeneous", r"different speeds?", r"gpus?", r"mixed hardware",
                         r"faster (?:and|or) slower", r"different machines"),
    "multiple_processors": _rx(r"multi[- ]?processors?", r"multi[- ]?core", r"cores", r"parallel",
                               r"processors", r"machines", r"servers", r"vms?", r"nodes", r"workers",
                               r"cluster"),
    "multiple_resources": _rx(r"resource types", r"multiple resources", r"several resources"),
    "interactive": _rx(r"interactive", r"responsive(?:ness)?", r"user[- ]facing", r"ui",
                       r"web servers?", r"(?:web|http|user|client) requests?", r"latency",
                       r"desktop"),
    "uncertainty": _DOMAIN_PATTERNS["stochastic"],
}

_OBJECTIVE_PATTERNS = {
    "avg_wait": _rx(r"waiting times?", r"wait(?:ing)?", r"turnaround", r"completion times?",
                    r"average (?:time|delay)"),
    "response": _rx(r"response(?: times?)?", r"latency", r"responsive(?:ness)?", r"snappy",
                    r"quick feedback"),
    "fairness": _rx(r"fair(?:ly|ness)?", r"starv(?:e|ation|ing)", r"equal share", r"equitabl[ey]"),
    "deadlines": _rx(r"meet(?:ing)? (?:all |the |their )?deadlines?", r"deadlines?", r"on time",
                     r"no misses", r"miss(?:ed|es)?"),
    "makespan": _rx(r"makespan", r"finish (?:everything|all|as soon)", r"total (?:completion|execution) time",
                    r"minimi[sz]e (?:the )?(?:total|overall) time", r"schedule length", r"project duration"),
    "throughput": _rx(r"throughput", r"jobs per (?:second|hour|minute)", r"as many jobs"),
    "utilization": _rx(r"utili[sz]ation", r"idle", r"keep .{0,20}busy"),
    "profit": _rx(r"profits?", r"revenue", r"value", r"rewards?", r"payoff"),
    "energy": _rx(r"energy", r"power", r"battery", r"green"),
    "safety": _rx(r"deadlocks?", r"safe states?", r"safety"),
    "predictability": _rx(r"predictab(?:le|ility)", r"certif(?:y|ied|iable)", r"guarantee[sd]?",
                          r"hard real[- ]?time", r"safety[- ]critical"),
    "adaptivity": _rx(r"adapt(?:ive|s|ing)?", r"dynamic(?:ally)?", r"changing", r"self[- ]tuning"),
    "risk": _rx(r"risk", r"confidence", r"p\d\d", r"percentiles?", r"worst[- ]case dates?"),
}

_DURATION = r"(?:(?:burst|run|execution|processing|service)[- ]?times?|runtimes?|durations?|lengths?)"
_BURSTS_KNOWN = _rx(rf"known {_DURATION}", rf"{_DURATION}(?: that)?(?: are| is)? (?:known|fixed|given)",
                    r"we know (?:how long|the durations?)",
                    r"estimates? (?:are|is) (?:accurate|reliable|good)")
_BURSTS_UNKNOWN = _rx(rf"unknown {_DURATION}", rf"unpredictable {_DURATION}",
                      rf"{_DURATION}(?: that)?(?: are| is)? (?:unknown|not known|unpredictable)",
                      r"don'?t know how long", r"no (?:runtime |duration )?estimates?")
_NON_PREEMPTIVE = _rx(r"non[- ]?preemptive", r"cannot be (?:interrupted|preempted)",
                      r"run to completion", r"no preemption", r"without interruption")
_PREEMPTIVE = _rx(r"preemptive", r"preempt(?:ion|ed)?", r"can be interrupted", r"interrupt(?:ible)?")
_PROCESSORS = re.compile(r"\b(\d+)\s*(?:cpus?|cores?|processors?|machines?|servers?|nodes?|vms?|workers?)\b",
                         re.IGNORECASE)


def parse_description(text: str) -> ProblemProfile:
    """Extract a :class:`ProblemProfile` from a free-text problem description."""
    profile = ProblemProfile()
    for domain, rx in _DOMAIN_PATTERNS.items():
        if rx.search(text):
            profile.domains.append(domain)
    for flag, rx in _FLAG_PATTERNS.items():
        if rx.search(text):
            profile.flags.add(flag)

    # Objectives are ordered by where they are first mentioned in the text.
    found = []
    for obj, rx in _OBJECTIVE_PATTERNS.items():
        m = rx.search(text)
        if m:
            found.append((m.start(), obj))
    profile.objectives = [o for _, o in sorted(found)]
    # "deadline" alone is a constraint; only treat it as the goal if nothing else is.
    if "deadlines" in profile.objectives and len(profile.objectives) > 1 and not re.search(
        r"\bmeet", text, re.IGNORECASE
    ):
        profile.objectives.remove("deadlines")
        profile.objectives.append("deadlines")

    if _BURSTS_UNKNOWN.search(text):
        profile.bursts_known = False
    elif _BURSTS_KNOWN.search(text):
        profile.bursts_known = True
    if _NON_PREEMPTIVE.search(text):
        profile.preemptive = False
    elif _PREEMPTIVE.search(text):
        profile.preemptive = True
    m = _PROCESSORS.search(text)
    if m:
        profile.processors = int(m.group(1))
        if profile.processors > 1:
            profile.flags.add("multiple_processors")
            if not {"multiprocessor", "cloud", "hpc"} & set(profile.domains):
                profile.domains.append("multiprocessor")
    if "dependencies" in profile.flags and "workflow" not in profile.domains:
        profile.domains.append("workflow")
    if "periodic" in profile.flags and "realtime" not in profile.domains:
        profile.domains.append("realtime")
    return profile


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

W_DOMAIN = 1.5  # × relevance (0-3)
W_OBJECTIVE = (2.0, 1.25, 0.75)  # weights of the 1st, 2nd, 3rd+ objective, × strength (0-3)
W_HANDLES = 1.5
P_REQUIRE_UNMET = 4.0
P_REQUIRE_UNKNOWN = 1.0
P_RISK = 2.0
P_PREEMPT_CONFLICT = 3.0

_FEATURE_TEXT = {
    **FLAGS,
    "bursts_known": "burst / run times known in advance",
    "bursts_unknown": "burst / run times unknown",
    "preemptive": "preemption allowed",
}


@dataclass
class Recommendation:
    info: AlgorithmInfo
    score: float
    match: float  # 0-1, score relative to a perfect fit for this profile
    reasons: list[tuple[str, str]]  # ("+" | "-", text)

    @property
    def name(self) -> str:
        return self.info.name


def _objective_weight(rank: int) -> float:
    return W_OBJECTIVE[min(rank, len(W_OBJECTIVE) - 1)]


def score_algorithm(info: AlgorithmInfo, profile: ProblemProfile) -> Recommendation:
    t = info.traits
    reasons: list[tuple[str, str]] = []
    score = 0.0

    domains = profile.domains or ["cpu"]
    rel, best = max((t.get("domains", {}).get(d, 0), d) for d in domains)
    if rel:
        score += W_DOMAIN * rel
        reasons.append(("+", f"designed for {DOMAINS[best].lower()}"))

    for rank, obj in enumerate(profile.effective_objectives()):
        strength = t.get("objectives", {}).get(obj, 0)
        if strength:
            score += _objective_weight(rank) * strength
            level = {1: "helps with", 2: "good for", 3: "excellent for"}[strength]
            reasons.append(("+", f"{level}: {OBJECTIVES[obj].lower()}"))

    for feature in t.get("handles", []):
        if profile.has(feature):
            score += W_HANDLES
            reasons.append(("+", f"handles {_FEATURE_TEXT.get(feature, feature).lower()}"))

    for feature in t.get("requires", []):
        if profile.has(feature):
            continue
        if feature == "bursts_known" and profile.bursts_known is None:
            score -= P_REQUIRE_UNKNOWN
            reasons.append(("-", "assumes run times can be estimated in advance"))
        else:
            score -= P_REQUIRE_UNMET if feature == "bursts_known" else 2 * P_REQUIRE_UNKNOWN
            reasons.append(("-", f"needs {_FEATURE_TEXT.get(feature, feature).lower()}"))

    for feature, why in t.get("risks", {}).items():
        if profile.has(feature) or feature in profile.effective_objectives():
            score -= P_RISK
            reasons.append(("-", why))

    pre = t.get("preemptive")
    if profile.preemptive is False and pre is True:
        score -= P_PREEMPT_CONFLICT
        reasons.append(("-", "requires preemption, but tasks cannot be interrupted"))

    return Recommendation(info, score, 0.0, reasons)


def _ideal_score(profile: ProblemProfile) -> float:
    ideal = W_DOMAIN * 3
    ideal += sum(_objective_weight(r) * 3 for r in range(len(profile.effective_objectives())))
    features = len(profile.flags) + (profile.bursts_known is not None) + bool(profile.preemptive)
    return ideal + W_HANDLES * min(features, 3)


def recommend(
    profile: ProblemProfile | str, top_k: int = 5, min_match: float = 0.2
) -> list[Recommendation]:
    """Rank knowledge-base algorithms for ``profile`` (a profile or free text).

    Entries matching less than ``min_match`` of an ideal fit are dropped (the best
    one is always kept) so the list does not trail off into noise.
    """
    if isinstance(profile, str):
        profile = parse_description(profile)
    if profile.is_empty():
        return []
    ideal = _ideal_score(profile)
    domains = profile.domains or ["cpu"]
    scored = []
    for info in KNOWLEDGE_BASE:
        rec = score_algorithm(info, profile)
        relevant = any(info.traits.get("domains", {}).get(d) for d in domains)
        if not relevant or rec.score <= 0:
            continue
        rec.match = max(0.0, min(1.0, rec.score / ideal)) if ideal else 0.0
        scored.append(rec)
    scored.sort(key=lambda r: (-r.score, r.name))
    return [r for i, r in enumerate(scored[:top_k]) if i == 0 or r.match >= min_match]


# ---------------------------------------------------------------------------
# Data-backed evaluation
# ---------------------------------------------------------------------------

_BASELINE = {"cpu": "fcfs", "multiprocessor": "list", "cloud": "min_min", "realtime": "rms"}


@dataclass
class Evaluation:
    family: str
    metric: str
    lower_is_better: bool
    rows: list[dict]  # sorted best first
    workload: str

    @property
    def winner(self) -> dict | None:
        return self.rows[0] if self.rows else None


def evaluate_on_workload(
    recommendations: list[Recommendation],
    profile: ProblemProfile,
    processes: list[Process] | None = None,
    workload: str | None = None,
    n: int = 40,
    seed: int = 0,
) -> Evaluation | None:
    """Run the implemented recommendations on a workload; best first by the
    profile's primary objective. Returns ``None`` if nothing is runnable."""
    candidates: list[str] = []
    for rec in recommendations:
        candidates += [k for k in rec.info.registry_keys if k not in candidates]
    if not candidates:
        return None
    # Compare within the family of the best implemented recommendation only:
    # a Banker's-algorithm problem should not be "evaluated" with CPU schedulers.
    family = get(candidates[0]).family
    if family not in _BASELINE:
        return None
    keys = [k for k in candidates if get(k).family == family]
    if _BASELINE[family] not in keys:
        keys.append(_BASELINE[family])

    objective = next(
        (o for o in profile.effective_objectives() if o in OBJECTIVE_METRIC), "avg_wait"
    )
    metric = OBJECTIVE_METRIC[objective]
    if family in ("multiprocessor", "cloud") and metric not in ("makespan", "cpu_utilization"):
        metric = "makespan"
    if family == "realtime":
        metric = "deadline_miss_ratio"
    lower = METRICS[metric][1]

    if family == "cpu":
        if processes is None:
            workload = workload or ("batch" if "batch" in profile.domains else "poisson")
            processes = wl.generate(workload, n, seed)
        else:
            workload = workload or "custom"
        inputs, extra = (processes,), {}
    elif family == "multiprocessor":
        workload = workload or "bimodal"
        # P||Cmax: every job is available at t = 0, so makespan reflects the packing.
        inputs = (wl.as_batch(processes or wl.generate(workload, n, seed)),)
        extra = {"m": profile.processors or 4}
    elif family == "cloud":
        workload = "etc_matrix"
        inputs, extra = (wl.etc_matrix(n, profile.processors or 4, seed),), {}
    else:  # realtime
        workload = "periodic_taskset (U=0.95, non-harmonic periods)"
        taskset = wl.periodic_taskset(5, 0.95, seed, periods=wl.NON_HARMONIC_PERIODS)
        inputs, extra = (taskset,), {}

    rows = []
    for key in keys:
        spec = get(key)
        schedule = spec.call(*inputs, **{**spec.params, **extra})
        metrics = summarize(schedule)
        rows.append({"key": key, "algorithm": spec.name, "value": metrics[metric], **metrics})
    rows.sort(key=lambda r: r["value"], reverse=not lower)
    return Evaluation(family, metric, lower, rows, workload)
