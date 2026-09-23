"""CLI smoke tests and Streamlit page smoke tests (AppTest)."""

import pathlib
import subprocess
import sys

import pytest

from schedlab.cli import main

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("argv", "expect"),
    [
        (["list"], "Uniprocessor CPU scheduling"),
        (["recommend", "batch", "jobs", "with", "known", "runtimes", "--evaluate", "-n", "15"],
         "DATA-BACKED CHECK"),
        (["general", "shortest", "path", "with", "negative", "weights"], "Bellman-Ford"),
        (["simulate", "-n", "5", "--algos", "fcfs,rr", "--gantt", "--cs", "0.5"], "RR (q=4)"),
        (["realtime", "-n", "2", "--gantt"], "Liu-Layland"),
        (["benchmark", "--algos", "fcfs,sjf", "--workloads", "batch", "--sizes", "10", "--seeds", "1"],
         "Overall leaderboard"),
        (["scaling", "--algos", "sjf,cpm", "--sizes", "50,100", "--repeats", "1"], "empirical"),
    ],
)
def test_cli_commands(argv, expect, capsys):
    assert main(argv) == 0
    assert expect in capsys.readouterr().out


def test_cli_reports_bad_input(capsys):
    assert main(["simulate", "--algos", "nope"]) == 2
    assert "unknown algorithm" in capsys.readouterr().err


def test_cli_recommend_without_match_returns_1(capsys):
    assert main(["recommend", "sort", "numbers"]) == 1


def test_cli_benchmark_writes_csv(tmp_path):
    out = tmp_path / "rows.csv"
    main(["benchmark", "--algos", "fcfs", "--workloads", "uniform", "--sizes", "8", "--seeds", "2",
          "--csv", str(out)])
    assert len(out.read_text().splitlines()) == 3


def test_daa101_entry_point_still_works():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "DAA101.py")],
        input="hard real-time periodic control loops with deadlines\n",
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "RECOMMENDED SCHEDULING ALGORITHMS" in proc.stdout
    assert "Winner on this workload" in proc.stdout


# ---------------------------------------------------------------- Streamlit

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
PAGES = [ROOT / "app" / "Home.py", *sorted((ROOT / "app" / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_renders_without_exceptions(page):
    at = streamlit_testing.AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_recommender_head_to_head_button():
    at = streamlit_testing.AppTest.from_file(str(ROOT / "app/pages/1_Recommender.py"),
                                             default_timeout=120).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("wins" in s.value for s in at.success)


def test_benchmark_lab_runs_both_experiments():
    at = streamlit_testing.AppTest.from_file(str(ROOT / "app/pages/7_Benchmark_Lab.py"),
                                             default_timeout=300).run()
    at.multiselect[2].set_value([25]).run()  # quality sizes: keep the test quick
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["bench_rows"]
    at.multiselect[3].set_value(["fcfs", "cpm"]).run()
    at.button[1].click().run()
    assert not at.exception
    assert at.session_state["scaling_rows"]


def test_cpu_simulator_reacts_to_parameters():
    at = streamlit_testing.AppTest.from_file(str(ROOT / "app/pages/2_CPU_Simulator.py"),
                                             default_timeout=120).run()
    at.slider[3].set_value(1.0).run()  # context-switch cost
    assert not at.exception
