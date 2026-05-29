"""Push task benchmark: train across algorithms and robot/cuboid configurations.

Run from repo root:
    uv run python scripts/benchmark_push.py
"""

from benchmark_base import AlgorithmSpec, BenchmarkRunner, REPO_ROOT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIMEOUT_S = 3 * 60 * 60  # per run

ALGORITHMS = [
    AlgorithmSpec("ppo", "push"),
]

# Keys must match fields in push_constants.py.
VARIANTS = [
    {"NUM_MASSPOINTS": 2, "NUM_UR10S": 0, "NUM_CUBOIDS": 1},
    {"NUM_MASSPOINTS": 2, "NUM_UR10S": 0, "NUM_CUBOIDS": 2},
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PushBenchmark(BenchmarkRunner):
    task_name = "push"
    constants_file = REPO_ROOT / "src" / "multi_robot_rl" / "configs" / "push_constants.py"
    patch_targets = {
        "NUM_MASSPOINTS": r"^NUM_MASSPOINTS\s*=\s*\d+",
        "NUM_UR10S":      r"^NUM_UR10S\s*=\s*\d+",
        "NUM_CUBOIDS":    r"^NUM_CUBOIDS\s*=\s*\d+",
    }

    def _variant_label(self, algo_name: str, variant: dict) -> str:
        parts = [algo_name]
        if variant.get("NUM_MASSPOINTS", 0) > 0:
            parts.append(f"{variant['NUM_MASSPOINTS']}mp")
        if variant.get("NUM_UR10S", 0) > 0:
            parts.append(f"{variant['NUM_UR10S']}ur10")
        n = variant.get("NUM_CUBOIDS", 1)
        parts.append(f"{n}cube{'s' if n > 1 else ''}")
        return "_".join(parts)


if __name__ == "__main__":
    PushBenchmark().run(ALGORITHMS, VARIANTS, TIMEOUT_S)
