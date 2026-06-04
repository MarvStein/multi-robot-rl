"""Reach task benchmark: train across algorithms and robot/goal configurations.

Run from repo root:
    uv run python scripts/benchmark_reach.py
"""

from benchmark_base import AlgorithmSpec, BenchmarkRunner, REPO_ROOT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIMEOUT_S = 60 * 180  # convervative timeout per run

ALGORITHMS = [
    AlgorithmSpec("ppo",      "reach"),
    # AlgorithmSpec("fast-sac", "reach-fast-sac"),
]

# Note: mjlab is not yet deterministic https://github.com/mujocolab/mjlab/issues/1023
SEEDS = [0, 1, 2, 3, 4]

# Keys must match fields in reach_constants.py.
VARIANTS = [
    # --- UR10s ---
    {"NUM_MASSPOINTS": 0, "NUM_UR10S": 2, "NUM_GOALS": 5},
    {"NUM_MASSPOINTS": 0, "NUM_UR10S": 1, "NUM_GOALS": 5},
    # --- masspoints ---
    {"NUM_MASSPOINTS": 2, "NUM_UR10S": 0, "NUM_GOALS": 5},
    {"NUM_MASSPOINTS": 1, "NUM_UR10S": 0, "NUM_GOALS": 5},
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ReachBenchmark(BenchmarkRunner):
    """Benchmark runner for the reach task.

    Trains several variants of the reach task. Each run
    patches ``reach_constants.py`` in-place to activate the selected variant
    before launching the training process.
    """

    task_name = "reach"
    constants_file = REPO_ROOT / "src" / "multi_robot_rl" / "configs" / "reach_constants.py"
    patch_targets = {
        "NUM_MASSPOINTS": r"^NUM_MASSPOINTS\s*=\s*\d+",
        "NUM_UR10S":      r"^NUM_UR10S\s*=\s*\d+",
        "NUM_GOALS":      r"^NUM_GOALS\s*=\s*\d+",
    }

    def _variant_label(self, algo_name: str, variant: dict) -> str:
        """Build a human-readable run label from the algorithm name and variant config.

        Encodes the robot composition and goal count into a compact string used
        as a directory/experiment name.  Example outputs: ``"ppo_2mp_5goals"``,
        ``"fast-sac_1ur10_5goals"``, ``"ppo_1mp_1ur10_5goals"``.

        Args:
            algo_name: Short algorithm identifier (e.g. ``"ppo"`` or ``"fast-sac"``).
            variant: Mapping of constant names to their values for this run;
                expected keys are ``NUM_MASSPOINTS``, ``NUM_UR10S``, and
                ``NUM_GOALS``.

        Returns:
            Underscore-joined label string encoding the algorithm, robot counts,
            and goal count.
        """
        parts = [algo_name]
        if variant.get("NUM_MASSPOINTS", 0) > 0:
            parts.append(f"{variant['NUM_MASSPOINTS']}mp")
        if variant.get("NUM_UR10S", 0) > 0:
            parts.append(f"{variant['NUM_UR10S']}ur10")
        n = variant.get("NUM_GOALS", 1)
        parts.append(f"{n}goal{'s' if n > 1 else ''}")
        return "_".join(parts)


if __name__ == "__main__":
    ReachBenchmark().run(ALGORITHMS, VARIANTS, TIMEOUT_S, SEEDS)
