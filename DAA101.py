"""Scheduling Algorithm Recommender (original entry point).

The recommender now lives in the ``schedlab`` package: it is explainable, scores
algorithms instead of plain keyword matching, and can check its picks on a
simulated workload. This script keeps the familiar ``python DAA101.py`` usage
working; see ``python -m schedlab --help`` for everything else.
"""

import sys

from schedlab.cli import main

if __name__ == "__main__":
    print("Welcome to the Scheduling Algorithm Recommender!")
    print("Describe your scheduling problem with details about:")
    print("- Type of resources being scheduled (CPU, machines, projects, etc.)")
    print("- Constraints (deadlines, priorities, dependencies, etc.)")
    print("- Optimization goals (minimizing time, maximizing throughput, energy efficiency, etc.)")
    print("- Environment characteristics (real-time, cloud, multiprocessor, etc.)\n")
    raise SystemExit(main(["recommend", "--evaluate", *sys.argv[1:]]))
