"""Reach task benchmark: train across algorithms and robot/goal configurations.

Run from repo root:
    uv run python scripts/benchmark_reach.py
"""

from benchmark_base import AlgorithmSpec, BenchmarkRunner, REPO_ROOT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIMEOUT_S = 3 * 60 * 60  # per run

ALGORITHMS = [
    AlgorithmSpec("ppo",      "reach"),
    AlgorithmSpec("fast-sac", "reach-fast-sac"),
]

# Keys must match fields in reach_constants.py.
VARIANTS = [
    {"NUM_MASSPOINTS": 1, "NUM_UR10S": 0, "NUM_GOALS": 5},
    {"NUM_MASSPOINTS": 0, "NUM_UR10S": 1, "NUM_GOALS": 5},
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ReachBenchmark(BenchmarkRunner):
    task_name = "reach"
    constants_file = REPO_ROOT / "src" / "multi_robot_rl" / "configs" / "reach_constants.py"
    patch_targets = {
        "NUM_MASSPOINTS": r"^NUM_MASSPOINTS\s*=\s*\d+",
        "NUM_UR10S":      r"^NUM_UR10S\s*=\s*\d+",
        "NUM_GOALS":      r"^NUM_GOALS\s*=\s*\d+",
    }

    def _variant_label(self, algo_name: str, variant: dict) -> str:
        parts = [algo_name]
        if variant.get("NUM_MASSPOINTS", 0) > 0:
            parts.append(f"{variant['NUM_MASSPOINTS']}mp")
        if variant.get("NUM_UR10S", 0) > 0:
            parts.append(f"{variant['NUM_UR10S']}ur10")
        n = variant.get("NUM_GOALS", 1)
        parts.append(f"{n}goal{'s' if n > 1 else ''}")
        return "_".join(parts)


if __name__ == "__main__":
    ReachBenchmark().run(ALGORITHMS, VARIANTS, TIMEOUT_S)
