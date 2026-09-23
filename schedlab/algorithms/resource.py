"""Resource allocation: Banker's algorithm and max-min fair sharing.

* **Banker's algorithm** (Dijkstra): grant a request only if some order still
  exists in which every process can obtain its maximum claim and finish, i.e. the
  state stays *safe*. The safety check is O(n²·m) for n processes and m resource
  types.
* **Max-min fairness** (water-filling): no one gets more than they ask for, and
  the smallest allocation is as large as possible. After sorting demands it takes
  O(n log n).
"""

from __future__ import annotations

from dataclasses import dataclass, field

Vector = list[float]
Matrix = list[list[float]]


@dataclass
class SafetyResult:
    safe: bool
    sequence: list[int]
    trace: list[dict] = field(default_factory=list)  # work vector after each step


@dataclass
class RequestResult:
    granted: bool
    reason: str
    safety: SafetyResult | None = None
    new_available: Vector | None = None
    new_allocation: Matrix | None = None


def _need(max_claim: Matrix, allocation: Matrix) -> Matrix:
    return [[mx - al for mx, al in zip(mrow, arow)] for mrow, arow in zip(max_claim, allocation)]


def _le(a: Vector, b: Vector) -> bool:
    return all(x <= y + 1e-9 for x, y in zip(a, b))


def _check_shapes(available: Vector, max_claim: Matrix, allocation: Matrix) -> None:
    m = len(available)
    if len(max_claim) != len(allocation):
        raise ValueError("max_claim and allocation must list the same processes")
    for i, (mrow, arow) in enumerate(zip(max_claim, allocation)):
        if len(mrow) != m or len(arow) != m:
            raise ValueError(f"process {i}: expected {m} resource types")
        if not _le(arow, mrow):
            raise ValueError(f"process {i}: allocation exceeds its maximum claim")


def bankers_safety(available: Vector, max_claim: Matrix, allocation: Matrix) -> SafetyResult:
    _check_shapes(available, max_claim, allocation)
    need = _need(max_claim, allocation)
    work = list(available)
    finished = [False] * len(allocation)
    sequence: list[int] = []
    trace: list[dict] = []
    progress = True
    while progress:
        progress = False
        for i in range(len(allocation)):
            if not finished[i] and _le(need[i], work):
                work = [w + a for w, a in zip(work, allocation[i])]
                finished[i] = True
                sequence.append(i)
                trace.append({"process": i, "need": need[i], "work_after": list(work)})
                progress = True
    return SafetyResult(all(finished), sequence, trace)


def bankers_request(
    pid: int, request: Vector, available: Vector, max_claim: Matrix, allocation: Matrix
) -> RequestResult:
    """Resource-request algorithm: tentatively grant, then keep it only if safe."""
    _check_shapes(available, max_claim, allocation)
    if not 0 <= pid < len(allocation):
        raise ValueError(f"unknown process {pid}")
    need = _need(max_claim, allocation)[pid]
    if not _le(request, need):
        return RequestResult(False, f"P{pid} asked for more than its declared maximum need {need}")
    if not _le(request, available):
        return RequestResult(False, f"P{pid} must wait: only {available} available")
    new_available = [a - r for a, r in zip(available, request)]
    new_allocation = [list(row) for row in allocation]
    new_allocation[pid] = [a + r for a, r in zip(allocation[pid], request)]
    safety = bankers_safety(new_available, max_claim, new_allocation)
    if not safety.safe:
        return RequestResult(False, "granting would leave the system in an UNSAFE state", safety)
    order = " → ".join(f"P{i}" for i in safety.sequence)
    return RequestResult(True, f"granted; safe sequence {order}", safety, new_available, new_allocation)


def max_min_fair(
    capacity: float, demands: list[float], weights: list[float] | None = None
) -> list[float]:
    """Weighted water-filling. Returns each user's allocation in input order."""
    if capacity < 0 or any(d < 0 for d in demands):
        raise ValueError("capacity and demands must be non-negative")
    weights = weights or [1.0] * len(demands)
    if len(weights) != len(demands) or any(w <= 0 for w in weights):
        raise ValueError("need one positive weight per demand")
    alloc = [0.0] * len(demands)
    # Satisfy users in order of demand per unit weight: a user whose normalised
    # demand fits under the current fair level is fully satisfied.
    order = sorted(range(len(demands)), key=lambda i: demands[i] / weights[i])
    remaining = capacity
    weight_left = sum(weights)
    for idx, i in enumerate(order):
        level = remaining / weight_left if weight_left > 0 else 0.0
        if demands[i] / weights[i] <= level:
            alloc[i] = demands[i]
            remaining -= demands[i]
            weight_left -= weights[i]
        else:
            for j in order[idx:]:
                alloc[j] = level * weights[j]
            break
    return alloc
