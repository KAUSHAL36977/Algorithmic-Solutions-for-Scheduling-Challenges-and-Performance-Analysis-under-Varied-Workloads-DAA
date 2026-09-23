"""Periodic real-time scheduling and schedulability analysis."""

import math

import pytest

from schedlab import workloads as wl
from schedlab.algorithms import energy, realtime
from schedlab.metrics import process_table, summarize
from schedlab.models import PeriodicTask, task_of

# U = 2/5 + 4/7 ≈ 0.971 > Liu-Layland bound 0.828: RMS fails, EDF succeeds.
CLASSIC = [PeriodicTask("T1", 2, 5), PeriodicTask("T2", 4, 7)]


def test_liu_layland_bound_values():
    assert realtime.liu_layland_bound(1) == pytest.approx(1.0)
    assert realtime.liu_layland_bound(2) == pytest.approx(0.8284, abs=1e-4)
    assert realtime.liu_layland_bound(1000) == pytest.approx(math.log(2), abs=1e-3)


def test_response_time_analysis_classic():
    rta = realtime.response_time_analysis(CLASSIC, "rm")
    assert rta["T1"] == 2
    assert math.isinf(rta["T2"])  # R = 4 + ⌈8/5⌉·2 = 8 > 7


def test_rta_converges_to_known_value():
    tasks = [PeriodicTask("a", 1, 4), PeriodicTask("b", 2, 6), PeriodicTask("c", 3, 13)]
    # R_c: 3+1+2=6 → 3+2·1+1·2=7 → 3+2+4=9 → 3+3+4=10 → 3+3+4=10 (fixed point)
    assert realtime.response_time_analysis(tasks)["c"] == 10


def test_rms_misses_and_edf_does_not():
    rms, edf = realtime.rate_monotonic(CLASSIC), realtime.edf_periodic(CLASSIC)
    assert summarize(rms)["deadline_miss_ratio"] > 0
    assert summarize(rms)["max_lateness"] == pytest.approx(1.0)
    assert summarize(edf)["deadline_miss_ratio"] == 0
    rep = realtime.schedulability_report(CLASSIC)
    assert not rep["rms_schedulable"] and rep["edf_schedulable"]
    assert rep["hyperperiod"] == 35


def test_harmonic_task_set_is_rms_schedulable_at_full_utilisation():
    tasks = [PeriodicTask("a", 5, 10), PeriodicTask("b", 10, 20)]
    rep = realtime.schedulability_report(tasks)
    assert rep["utilization"] == pytest.approx(1.0)
    assert not rep["rms_ll_pass"] and rep["rms_schedulable"]
    assert summarize(realtime.rate_monotonic(tasks))["deadline_miss_ratio"] == 0


def test_edf_demand_test_constrained_deadlines():
    ok = [PeriodicTask("a", 1, 4, 2), PeriodicTask("b", 1, 4, 3)]
    bad = [PeriodicTask("a", 2, 4, 2), PeriodicTask("b", 1, 4, 2)]
    assert realtime.edf_demand_test(ok)
    assert not realtime.edf_demand_test(bad)  # dbf(2) = 3 > 2 although U = 0.75


def test_dms_beats_rms_with_short_deadlines():
    tasks = [PeriodicTask("long_period", 1, 10, 2), PeriodicTask("short_period", 2, 5, 5)]
    assert summarize(realtime.deadline_monotonic(tasks))["deadline_miss_ratio"] == 0
    assert summarize(realtime.rate_monotonic(tasks))["deadline_miss_ratio"] > 0


def test_fractional_hyperperiod():
    assert realtime.hyperperiod([PeriodicTask("a", 1, 2.5), PeriodicTask("b", 1, 4)]) == 20


@pytest.mark.parametrize("seed", range(8))
def test_edf_never_misses_when_utilisation_at_most_one(seed):
    tasks = wl.periodic_taskset(5, 0.97, seed, periods=wl.NON_HARMONIC_PERIODS)
    if realtime.utilization(tasks) <= 1:
        assert summarize(realtime.edf_periodic(tasks))["deadline_miss_ratio"] == 0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("fn", [realtime.rate_monotonic, realtime.deadline_monotonic,
                                realtime.edf_periodic])
def test_rta_prediction_matches_simulation(seed, fn):
    tasks = wl.periodic_taskset(4, 0.9, seed, periods=wl.NON_HARMONIC_PERIODS)
    rep = realtime.schedulability_report(tasks)
    predicted = {realtime.rate_monotonic: rep["rms_schedulable"],
                 realtime.deadline_monotonic: rep["dms_schedulable"],
                 realtime.edf_periodic: rep["edf_schedulable"]}[fn]
    missed = summarize(fn(tasks))["deadline_miss_ratio"] > 0
    assert predicted == (not missed)


@pytest.mark.parametrize("seed", range(4))
def test_periodic_simulation_invariants(seed):
    tasks = wl.periodic_taskset(4, 0.85, seed)
    s = realtime.edf_periodic(tasks)
    C = {t.name: t.C for t in tasks}
    release = {p.pid: p.arrival for p in s.processes}
    ordered = sorted(s.slices, key=lambda x: x.start)
    for a, b in zip(ordered, ordered[1:]):
        assert a.end <= b.start + 1e-9
    assert all(sl.start >= release[sl.pid] - 1e-9 for sl in s.slices)
    for row in process_table(s):
        assert row["burst"] == C[task_of(row["pid"])]


def test_dvfs_picks_lowest_feasible_frequency():
    tasks = [PeriodicTask("a", 1, 10), PeriodicTask("b", 2, 20)]  # U = 0.2
    d = energy.dvfs_edf(tasks)
    assert d["frequency"] == 0.25
    assert d["energy_ratio"] == pytest.approx(0.25**2)
    assert summarize(d["schedule"])["deadline_miss_ratio"] == 0
    assert not energy.dvfs_edf(CLASSIC + [PeriodicTask("c", 1, 5)])["feasible"]


def test_sleep_uses_break_even_rule():
    s = realtime.edf_periodic([PeriodicTask("a", 1, 10)])
    res = energy.sleep_schedule(s, p_idle=0.5, p_sleep=0.0, transition_energy=2.0, horizon=10)
    assert res["break_even"] == 4
    assert [g["sleep"] for g in res["gaps"]] == [True]  # the 9-unit idle gap
    assert res["energy_with_sleep"] < res["energy_always_idle"]
