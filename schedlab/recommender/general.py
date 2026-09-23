"""General-purpose algorithm advisor (sorting, graphs, strings, optimisation ...).

This consolidates the four near-identical ``AlgorithmRecommenderChatbot`` classes and the
module-level ``recommend_algorithms`` function from the legacy file
``Alorithmic-Solution-to-all-life-problems.py``. The richest description of each
algorithm was kept, whole-word matching replaced raw substring checks, and nothing
prompts for input at import time.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneralAlgorithm:
    name: str
    description: str
    complexity: str = ""


def _a(name: str, description: str, complexity: str = "") -> GeneralAlgorithm:
    return GeneralAlgorithm(name, " ".join(description.split()), complexity)


MERGE_SORT = _a("Merge Sort", """Stable divide-and-conquer sort: split the array in half, sort
    each half recursively, then merge the sorted halves. Consistent O(n log n) time but
    O(n) extra space. Steps: divide into halves → recursively sort each → merge. Best for
    large datasets where stability matters, linked lists, and external sorting in databases.""",
    "O(n log n) time, O(n) space")
QUICK_SORT = _a("Quick Sort", """In-place sort that partitions around a pivot and recursively
    sorts the partitions. Average O(n log n), worst case O(n²) on already-sorted input with a
    poor pivot (use a random or median-of-three pivot), O(log n) stack. Usually the fastest
    in-memory sort in practice, but not stable.""", "O(n log n) average, O(n²) worst")
HEAP_SORT = _a("Heap Sort", """Builds a max-heap, then repeatedly swaps the maximum to the end
    and restores the heap. Guaranteed O(n log n) with O(1) extra space, so it is ideal when
    memory is tight (embedded systems). Not stable and slower than Quick Sort in practice.""",
    "O(n log n) time, O(1) space")
BINARY_SEARCH = _a("Binary Search", """Finds an item in a sorted array by repeatedly halving the
    search interval. Requires sorted input.""", "O(log n)")
DFS = _a("Depth-First Search (DFS)", """Explores as far as possible along each branch before
    backtracking. Used for connectivity, cycle detection, topological sort and strongly
    connected components.""", "O(V + E)")
BFS = _a("Breadth-First Search (BFS)", """Explores neighbours level by level; finds shortest paths
    in unweighted graphs.""", "O(V + E)")
DIJKSTRA = _a("Dijkstra's Algorithm", """Single-source shortest paths with non-negative edge
    weights. Greedily settles the closest unsettled vertex using a priority queue and relaxes
    its edges. Does not work with negative weights (use Bellman-Ford). Used in navigation
    systems and routing protocols.""", "O((V + E) log V)")
BELLMAN_FORD = _a("Bellman-Ford Algorithm", """Single-source shortest paths that tolerates
    negative edge weights: relax every edge V-1 times, then one more pass detects negative
    cycles. Slower than Dijkstra but used for arbitrage detection and distance-vector
    routing.""", "O(V·E)")
FLOYD_WARSHALL = _a("Floyd-Warshall Algorithm", """All-pairs shortest paths by dynamic
    programming over intermediate vertices k. Simple, detects negative cycles, best for dense
    graphs; too slow for very large sparse graphs.""", "O(V³)")
KRUSKAL = _a("Kruskal's Algorithm", """Minimum spanning tree: add edges in increasing weight
    order, skipping any that would form a cycle (union-find). Best for sparse graphs.""",
    "O(E log E)")
PRIM = _a("Prim's Algorithm", """Minimum spanning tree grown from a start vertex, always adding
    the cheapest edge leaving the tree (priority queue). Efficient for dense graphs.""",
    "O(E log V)")
KMP = _a("Knuth-Morris-Pratt (KMP)", """Exact string matching using a prefix (failure) function
    so the text is never re-scanned.""", "O(n + m)")
RABIN_KARP = _a("Rabin-Karp", """String matching with a rolling hash; great for searching many
    patterns at once (plagiarism detection).""", "O(n + m) expected")
BOYER_MOORE = _a("Boyer-Moore", """Matches the pattern right-to-left and skips ahead using
    bad-character and good-suffix rules; very fast on natural-language text.""",
    "sublinear on average")
DP = _a("Dynamic Programming", """Solves problems with overlapping subproblems and optimal
    substructure by storing sub-solutions (memoisation or tabulation): knapsack, LCS, edit
    distance, matrix-chain order.""", "problem-dependent")
GREEDY = _a("Greedy Algorithm", """Takes the locally best choice at each step. Optimal when the
    problem has the greedy-choice property (activity selection, fractional knapsack, Huffman
    coding).""", "often O(n log n)")
DP_KNAPSACK = _a("0/1 Knapsack (Dynamic Programming)", """Tabulates the best value for each
    item prefix and capacity. Exact for integer weights.""", "O(n·W) pseudo-polynomial")
BB_KNAPSACK = _a("Branch and Bound (Knapsack)", """Explores a state-space tree, pruning branches
    whose optimistic (fractional) bound cannot beat the best solution found so far.""",
    "exponential worst case, fast in practice")
SIEVE = _a("Sieve of Eratosthenes", """Lists every prime up to n by crossing out the multiples of
    each prime.""", "O(n log log n)")
MILLER_RABIN = _a("Miller-Rabin Primality Test", """Probabilistic primality test for huge numbers
    (cryptography); deterministic for 64-bit inputs with fixed bases.""", "O(k log³ n)")
STRASSEN = _a("Strassen's Matrix Multiplication", """Multiplies matrices with 7 recursive
    half-size products instead of 8.""", "O(n^2.81)")
TSP_APPROX = _a("Approximation Algorithms (TSP)", """Christofides (1.5-approx) or MST doubling
    (2-approx) for metric TSP when exact solutions are too slow.""", "polynomial")
SET_COVER = _a("Set Cover Approximation", """Greedy: repeatedly pick the set covering the most
    uncovered elements. An H(n) ≈ ln n approximation.""", "O(Σ|S|)")
A_STAR = _a("A* Algorithm", """Best-first search that orders nodes by g(n) + h(n); with an
    admissible heuristic it finds optimal paths and explores far fewer nodes than Dijkstra.
    Standard in games and robotics.""", "O(E) with a good heuristic")
REGEX = _a("Regular Expressions", """Pattern matching via finite automata; ideal for validation,
    tokenising and log search.""", "O(n) for DFA-based engines")
N_QUEENS = _a("N-Queens (Backtracking)", """Places queens row by row and backtracks on conflict:
    the classic backtracking example.""", "O(n!) worst case")
SUDOKU = _a("Sudoku Solver (Backtracking)", """Fills cells with candidate digits, backtracking on
    constraint violations (add constraint propagation for speed).""", "exponential worst case")
GRAPH_COLORING = _a("Graph Coloring (Backtracking)", """Assigns colours to vertices so that no two
    neighbours share one; backtracking with pruning (used for register allocation and
    timetabling).""", "exponential worst case")
FORD_FULKERSON = _a("Ford-Fulkerson", """Maximum flow via repeated augmenting paths in the residual
    graph.""", "O(E·|f*|)")
EDMONDS_KARP = _a("Edmonds-Karp", """Ford-Fulkerson with BFS augmenting paths: a polynomial
    max-flow algorithm, used for bipartite matching and network capacity.""", "O(V·E²)")

TOPICS: dict[str, tuple[tuple[str, ...], tuple[GeneralAlgorithm, ...]]] = {
    "sorting": ((r"sort(?:s|ing|ed)?", r"order(?:ing)? (?:the )?(?:data|items|numbers|records)"),
                (MERGE_SORT, QUICK_SORT, HEAP_SORT)),
    "searching": ((r"search(?:ing)?", r"find(?:ing)? an? (?:item|element|value)", r"lookup"),
                  (BINARY_SEARCH, DFS, BFS)),
    "shortest path": ((r"shortest paths?", r"routing", r"navigation", r"distances?"),
                      (DIJKSTRA, BELLMAN_FORD, FLOYD_WARSHALL)),
    "graph": ((r"graphs?", r"networks?", r"nodes and edges", r"connectivity", r"cycles?"),
              (DFS, BFS, KRUSKAL, PRIM)),
    "string matching": ((r"string matching", r"substrings?", r"text search", r"strings?"),
                        (KMP, RABIN_KARP, BOYER_MOORE)),
    "optimization": ((r"optimi[sz](?:ation|e)", r"maximi[sz]e", r"minimi[sz]e", r"best choice"),
                     (DP, GREEDY)),
    "knapsack": ((r"knapsack", r"capacity .{0,20}(?:weight|value)"), (DP_KNAPSACK, BB_KNAPSACK)),
    "primes": ((r"primes?", r"primality"), (SIEVE, MILLER_RABIN)),
    "matrix": ((r"matri(?:x|ces)", r"matrix multiplication"), (STRASSEN, DP)),
    "approximation": ((r"approximat(?:e|ion)", r"np[- ]hard", r"tsp", r"travell?ing salesman"),
                      (TSP_APPROX, SET_COVER)),
    "pathfinding": ((r"pathfinding", r"path finding", r"maze", r"grid path", r"game ai"),
                    (A_STAR, DIJKSTRA)),
    "minimum spanning tree": ((r"minimum spanning trees?", r"mst", r"spanning trees?",
                               r"cheapest network"), (KRUSKAL, PRIM)),
    "pattern": ((r"patterns?", r"regex", r"regular expressions?"), (DP, REGEX)),
    "backtracking": ((r"backtrack(?:ing)?", r"n[- ]queens?", r"sudoku", r"puzzles?",
                      r"constraint satisfaction", r"colou?ring"), (N_QUEENS, SUDOKU, GRAPH_COLORING)),
    "flow": ((r"max(?:imum)?[- ]flow", r"flow networks?", r"flows?", r"bipartite matching"),
             (FORD_FULKERSON, EDMONDS_KARP)),
}

_PATTERNS = {
    topic: re.compile(r"\b(?:" + "|".join(aliases) + r")\b", re.IGNORECASE)
    for topic, (aliases, _) in TOPICS.items()
}

_FOLLOW_UPS = [
    ("graph", "Got it! Is it related to shortest path, MST (minimum spanning tree), or "
              "finding strongly connected components?"),
    ("string matching", "Understood! Are you looking for exact string matching, pattern searching, "
                        "or substring finding?"),
    ("pattern", "Understood! Are you looking for exact string matching, pattern searching, "
                "or substring finding?"),
    ("optimization", "Optimization! Is it related to the knapsack problem, pathfinding, or another "
                     "optimization task?"),
]


class GeneralAlgorithmAdvisor:
    """Recommend classic algorithms from a free-text description."""

    def match_topics(self, text: str) -> list[str]:
        return [topic for topic, rx in _PATTERNS.items() if rx.search(text)]

    def recommend(self, text: str) -> list[tuple[str, GeneralAlgorithm]]:
        """(topic, algorithm) pairs; each algorithm appears once, under its first topic."""
        seen: set[str] = set()
        out = []
        for topic in self.match_topics(text):
            for algo in TOPICS[topic][1]:
                if algo.name not in seen:
                    seen.add(algo.name)
                    out.append((topic, algo))
        return out

    def follow_up_question(self, text: str) -> str:
        topics = set(self.match_topics(text))
        for topic, question in _FOLLOW_UPS:
            if topic in topics:
                return question
        return "Could you please share any more details that might help refine my suggestion?"

    def run_interactive(
        self, ask: Callable[[str], str] = input, say: Callable[[str], None] = print
    ) -> list[tuple[str, GeneralAlgorithm]]:
        say("Hello! I'm here to help you find the best algorithm for your problem.")
        first = ask(
            "Could you describe the main type of problem you are dealing with? "
            "For example, is it about sorting, searching, graphs, optimization, etc.?\n"
        )
        details = ask(self.follow_up_question(first) + "\n")
        recs = self.recommend(f"{first} {details}")
        if not recs:
            say("\nI'm sorry, I couldn't find a specific recommendation based on the provided details. "
                "You might try describing the problem differently.")
            return recs
        say("\nBased on your description, here are some algorithm recommendations:")
        for topic, algo in recs:
            suffix = f" [{algo.complexity}]" if algo.complexity else ""
            say(f"\n- {algo.name}{suffix} ({topic}):\n  {algo.description}")
        return recs
