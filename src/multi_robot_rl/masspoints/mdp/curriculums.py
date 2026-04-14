""" Custom curriculum to update reward term weights based on a metric value (e.g. success rate). Similar to Mjlab's built-in `reward_curriculum` but with support for more flexible stage definitions based on metric thresholds rather than fixed steps. """

import torch
from typing import TypedDict, Any, Sequence
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumTermCfg

class _MetricCurriculumStageOptional(TypedDict, total=False):
    weight: float
    params: dict[str, Any]

class MetricCurriculumStage(_MetricCurriculumStageOptional):
    metric_value: float

class metric_reward_curriculum:
    """Update a reward term's weight based on a metric value (exponential moving average)."""
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.reward_name: str = cfg.params["reward_name"]
        self.metric_name: str = cfg.params["metric_name"]
        self.stages: list[MetricCurriculumStage] = cfg.params["stages"]
        
        self.alpha: float = cfg.params.get("alpha", 0.01) # EMA smoothing factor
        self.ema_metric = 0.0

        self._term_cfg = env.reward_manager.get_term_cfg(self.reward_name)
        self.stages = sorted(self.stages, key=lambda x: x["metric_value"])

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        
        # We need the instantaneous metric value for this step across ALL environments
        # to update our running average.
        metric_idx = env.metrics_manager.active_terms.index(self.metric_name)
        # `env.metrics_manager._step_values[:, metric_idx]` holds the latest step value for each env
        latest_vals = env.metrics_manager._step_values[:, metric_idx]
        current_mean = latest_vals.mean().item()
        
        # Exponential moving average to avoid noisy sudden jumps
        self.ema_metric = (1.0 - self.alpha) * self.ema_metric + self.alpha * current_mean

        for stage in self.stages:
            if self.ema_metric >= stage["metric_value"]:
                if "weight" in stage:
                    self._term_cfg.weight = stage["weight"]
                if "params" in stage:
                    self._term_cfg.params.update(stage["params"])

        result = {"ema_value": torch.tensor(self.ema_metric)}
        result["alpha"] = torch.tensor(self.alpha)
        if hasattr(self._term_cfg, "weight"):
            result["weight"] = torch.tensor(self._term_cfg.weight)
            
        return result
