"""Type task benchmark: train across algorithms and robot/keyboard configurations.

Run from repo root:
    uv run python scripts/benchmark_type.py
"""

from benchmark_base import AlgorithmSpec, BenchmarkRunner, REPO_ROOT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIMEOUT_S = 60 * 120  # convervative timeout per run

ALGORITHMS = [
    AlgorithmSpec("ppo", "type"),
]

# Note: mjlab is not yet deterministic https://github.com/mujocolab/mjlab/issues/1023
SEEDS = [0, 1, 2, 3, 4]

VARIANTS = [
    {"NUM_MASSPOINTS": 2, "NUM_UR10S": 0, "NUM_ACTIVE_KEYS": 3},
    {"NUM_MASSPOINTS": 1, "NUM_UR10S": 0, "NUM_ACTIVE_KEYS": 3},
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TypeBenchmark(BenchmarkRunner):
    """Benchmark runner for the type (keyboard) task."""

    task_name = "type"

    def _variant_label(self, algo_name: str, variant: dict) -> str:
        parts = [algo_name]
        if variant.get("NUM_MASSPOINTS", 0) > 0:
            parts.append(f"{variant['NUM_MASSPOINTS']}mp")
        if variant.get("NUM_UR10S", 0) > 0:
            parts.append(f"{variant['NUM_UR10S']}ur10")
        parts.append(f"{variant.get('NUM_ACTIVE_KEYS', 3)}ak")
        return "_".join(parts)


if __name__ == "__main__":
    TypeBenchmark().run(ALGORITHMS, VARIANTS, TIMEOUT_S, SEEDS)
