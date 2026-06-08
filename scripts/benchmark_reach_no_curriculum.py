"""Reach task ablation: no curriculum (goals spawn at full workspace radius from start).

Used to demonstrate the effect of sparse rewards without curriculum scaffolding.
Run from repo root:
    uv run python scripts/benchmark_reach_no_curriculum.py
"""

from benchmark_base import AlgorithmSpec, BenchmarkRunner

TIMEOUT_S = 60 * 240  # conservative timeout per run
ALGORITHMS = [AlgorithmSpec("ppo", "reach-no-curriculum")]
SEEDS = [0, 1, 2]  # Note: mjlab is not yet deterministic https://github.com/mujocolab/mjlab/issues/1023
VARIANTS = [{"NUM_MASSPOINTS": 1, "NUM_UR10S": 0, "NUM_GOALS": 5}]


class ReachNoCurriculumBenchmark(BenchmarkRunner):
    task_name = "reach-no-curriculum"

    def _variant_label(self, algo_name: str, variant: dict) -> str:
        return f"{algo_name}_1mp_5goals_no_curriculum"


if __name__ == "__main__":
    ReachNoCurriculumBenchmark().run(ALGORITHMS, VARIANTS, TIMEOUT_S, SEEDS)
