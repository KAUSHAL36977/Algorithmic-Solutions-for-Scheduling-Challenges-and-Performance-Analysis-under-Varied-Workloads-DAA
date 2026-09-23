"""Energy-aware scheduling: static DVFS for EDF and break-even sleep scheduling.

Energy model: dynamic power ∝ fᵅ (α ≈ 3 for CMOS, since P ≈ C·V²·f and V scales
with f). Running W units of work at normalised frequency f takes W/f time, so
dynamic energy ∝ W·f^(α-1). Slowing down saves energy as long as deadlines
still hold.

* **Static DVFS (EDF)**: EDF meets every deadline iff U/f ≤ 1, so the lowest
  available frequency f ≥ U is the energy-optimal constant speed.
* **Sleep scheduling**: entering a sleep state costs a fixed transition energy
  E_tr. The offline optimum sleeps through an idle gap exactly when the gap is
  longer than the *break-even time* E_tr / (P_idle - P_sleep).
"""

from __future__ import annotations

from ..models import EPS, PeriodicTask, Schedule
from .realtime import edf_periodic, utilization

DEFAULT_LEVELS = (0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0)


def dvfs_edf(
    tasks: list[PeriodicTask],
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    alpha: float = 3.0,
) -> dict:
    u = utilization(tasks)
    feasible = sorted(f for f in levels if f >= u - EPS)
    if not feasible:
        return {
            "utilization": u,
            "frequency": max(levels),
            "feasible": False,
            "energy_ratio": 1.0,
            "schedule": edf_periodic(tasks),
        }
    f = feasible[0]
    scaled = [PeriodicTask(t.name, t.C / f, t.T, t.D) for t in tasks]
    return {
        "utilization": u,
        "frequency": f,
        "feasible": True,
        # Same work, energy ∝ W·f^(α-1), relative to always running at f = 1.
        "energy_ratio": f ** (alpha - 1),
        "schedule": edf_periodic(scaled),
    }


def idle_gaps(schedule: Schedule, horizon: float | None = None) -> list[tuple[float, float]]:
    """Idle intervals on CPU 0 between time 0 and ``horizon`` (default: makespan)."""
    end = schedule.makespan if horizon is None else horizon
    gaps, t = [], 0.0
    for s in sorted((s for s in schedule.slices if s.cpu == 0), key=lambda s: s.start):
        if s.start > t + EPS:
            gaps.append((t, s.start))
        t = max(t, s.end)
    if end > t + EPS:
        gaps.append((t, end))
    return gaps


def sleep_schedule(
    schedule: Schedule,
    p_active: float = 1.0,
    p_idle: float = 0.6,
    p_sleep: float = 0.05,
    transition_energy: float = 2.0,
    horizon: float | None = None,
) -> dict:
    """Apply the break-even sleep policy to every idle gap of ``schedule``."""
    if p_idle <= p_sleep:
        raise ValueError("idle power must exceed sleep power")
    break_even = transition_energy / (p_idle - p_sleep)
    gaps = idle_gaps(schedule, horizon)
    busy = sum(s.duration for s in schedule.work_slices() if s.cpu == 0)
    idle_energy = sum(b - a for a, b in gaps) * p_idle
    sleep_energy = 0.0
    decisions = []
    for a, b in gaps:
        length = b - a
        sleep = length > break_even
        sleep_energy += (transition_energy + length * p_sleep) if sleep else length * p_idle
        decisions.append({"start": a, "end": b, "length": length, "sleep": sleep})
    base = busy * p_active + idle_energy
    with_sleep = busy * p_active + sleep_energy
    return {
        "break_even": break_even,
        "gaps": decisions,
        "energy_always_idle": base,
        "energy_with_sleep": with_sleep,
        "savings_ratio": 1 - with_sleep / base if base > 0 else 0.0,
    }
