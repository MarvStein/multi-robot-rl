"""Push task benchmark: train across algorithms and robot/cuboid configurations.

Run from repo root:
    uv run python scripts/benchmark_push.py
"""

from benchmark_base import AlgorithmSpec, BenchmarkRunner, REPO_ROOT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIMEOUT_S = 60 * 200  # convervative timeout per run

ALGORITHMS = [
    AlgorithmSpec("ppo", "push"),
]

# Note: mjlab is not yet deterministic https://github.com/mujocolab/mjlab/issues/1023
SEEDS = [0, 1, 2]

VARIANTS = [
    {"NUM_MASSPOINTS": 2, "NUM_UR10S": 0, "NUM_CUBOIDS": 1},
    {"NUM_MASSPOINTS": 1, "NUM_UR10S": 0, "NUM_CUBOIDS": 1},
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PushBenchmark(BenchmarkRunner):
    """Benchmark runner for the push task."""

    task_name = "push"

    def _variant_label(self, algo_name: str, variant: dict) -> str:
        """Build a human-readable run label from the algorithm name and variant config.

        Encodes the robot composition and cuboid count into a compact string
        used as a directory/experiment name.  Example outputs:
        ``"ppo_2mp_1cube"``, ``"ppo_4mp_3cubes"``.

        Args:
            algo_name: Short algorithm identifier (e.g. ``"ppo"``).
            variant: Mapping of constant names to their values for this run;
                expected keys are ``NUM_MASSPOINTS``, ``NUM_UR10S``, and
                ``NUM_CUBOIDS``.

        Returns:
            Underscore-joined label string encoding the algorithm, robot counts,
            and cuboid count.
        """
        parts = [algo_name]
        if variant.get("NUM_MASSPOINTS", 0) > 0:
            parts.append(f"{variant['NUM_MASSPOINTS']}mp")
        if variant.get("NUM_UR10S", 0) > 0:
            parts.append(f"{variant['NUM_UR10S']}ur10")
        n = variant.get("NUM_CUBOIDS", 1)
        parts.append(f"{n}cube{'s' if n > 1 else ''}")
        return "_".join(parts)


if __name__ == "__main__":
    PushBenchmark().run(ALGORITHMS, VARIANTS, TIMEOUT_S, SEEDS)
