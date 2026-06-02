"""Type task benchmark: train across algorithms and robot/keyboard configurations.

Run from repo root:
    uv run python scripts/benchmark_type.py
"""

from benchmark_base import AlgorithmSpec, BenchmarkRunner, REPO_ROOT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TIMEOUT_S = 3 * 60 * 60  # per run

ALGORITHMS = [
    AlgorithmSpec("ppo", "type"),
]

# Keys must match fields in type_constants.py.
VARIANTS = [
    {"NUM_MASSPOINTS": 1, "NUM_UR10S": 0, "NUM_ACTIVE_KEYS": 3, "NUM_COLS": 6, "NUM_ROWS": 3},
    {"NUM_MASSPOINTS": 3, "NUM_UR10S": 0, "NUM_ACTIVE_KEYS": 3, "NUM_COLS": 6, "NUM_ROWS": 3},
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TypeBenchmark(BenchmarkRunner):
    """Benchmark runner for the type (keyboard) task.

    Trains PPO across two robot count variants on a fixed 6x3 keyboard layout
    with 3 active keys: one masspoint and three masspoints.  Each run patches
    ``type_constants.py`` in-place to activate the selected variant before
    launching the training process.
    """

    task_name = "type"
    constants_file = REPO_ROOT / "src" / "multi_robot_rl" / "configs" / "type_constants.py"
    patch_targets = {
        "NUM_MASSPOINTS": r"^NUM_MASSPOINTS\s*=\s*\d+",
        "NUM_UR10S":      r"^NUM_UR10S\s*=\s*\d+",
        "NUM_ACTIVE_KEYS": r"^NUM_ACTIVE_KEYS\s*=\s*\d+",
        "NUM_COLS":       r"^NUM_COLS\s*=\s*\d+",
        "NUM_ROWS":       r"^NUM_ROWS\s*=\s*\d+",
    }

    def _variant_label(self, algo_name: str, variant: dict) -> str:
        """Build a human-readable run label from the algorithm name and variant config.

        Encodes the robot composition, keyboard grid dimensions, and active key
        count into a compact string used as a directory/experiment name.
        Example outputs: ``"ppo_1mp_6x3kb_3ak"``, ``"ppo_3mp_6x3kb_3ak"``.

        Args:
            algo_name: Short algorithm identifier (e.g. ``"ppo"``).
            variant: Mapping of constant names to their values for this run;
                expected keys are ``NUM_MASSPOINTS``, ``NUM_UR10S``,
                ``NUM_ACTIVE_KEYS``, ``NUM_COLS``, and ``NUM_ROWS``.

        Returns:
            Underscore-joined label string encoding the algorithm, robot counts,
            keyboard grid size, and active key count.
        """
        parts = [algo_name]
        if variant.get("NUM_MASSPOINTS", 0) > 0:
            parts.append(f"{variant['NUM_MASSPOINTS']}mp")
        if variant.get("NUM_UR10S", 0) > 0:
            parts.append(f"{variant['NUM_UR10S']}ur10")
        active_keys = variant.get("NUM_ACTIVE_KEYS", 3)
        cols = variant.get("NUM_COLS", 6)
        rows = variant.get("NUM_ROWS", 3)
        parts.append(f"{cols}x{rows}kb")
        parts.append(f"{active_keys}ak")
        return "_".join(parts)


if __name__ == "__main__":
    TypeBenchmark().run(ALGORITHMS, VARIANTS, TIMEOUT_S)
