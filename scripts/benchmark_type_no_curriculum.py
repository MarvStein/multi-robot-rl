"""Type task ablation: no curriculum (wrong_key_penalty = -0.3 from step 0).

Run from repo root:
    uv run python scripts/benchmark_type_no_curriculum.py
"""

from benchmark_base import AlgorithmSpec, BenchmarkRunner

TIMEOUT_S = 60 * 120  # convervative timeout per run
ALGORITHMS = [AlgorithmSpec("ppo", "type-no-curriculum")]
SEEDS = [0, 1, 2] # Note: mjlab is not yet deterministic https://github.com/mujocolab/mjlab/issues/1023
VARIANTS = [{"NUM_MASSPOINTS": 1, "NUM_UR10S": 0, "NUM_ACTIVE_KEYS": 3}]


class TypeNoCurriculumBenchmark(BenchmarkRunner):
    task_name = "type-no-curriculum"

    def _variant_label(self, algo_name: str, variant: dict) -> str:
        return f"{algo_name}_1mp_3ak_no_curriculum"


if __name__ == "__main__":
    TypeNoCurriculumBenchmark().run(ALGORITHMS, VARIANTS, TIMEOUT_S, SEEDS)
