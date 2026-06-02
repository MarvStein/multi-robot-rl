"""Curriculum classes that advance training stages based on exponential moving averages of metrics.

Each curriculum class monitors a named metric via the environment's metrics_manager, maintains an
EMA of that metric, and updates either an event term's params or a reward term's weight/params
once the EMA crosses the threshold defined by each stage.
"""

import torch
from typing import TypedDict, Any
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumTermCfg

class _MetricCurriculumStageOptional(TypedDict, total=False):
    """Optional fields that a curriculum stage may provide to override term configuration.

    Attributes:
        weight: Replacement weight for the reward term at this stage.
        params: Partial parameter dict merged into the term's existing params at this stage.
    """

    weight: float
    params: dict[str, Any]

class MetricCurriculumStage(_MetricCurriculumStageOptional):
    """Full specification of one curriculum stage, including the metric threshold that unlocks it.

    Attributes:
        metric_value: EMA threshold the monitored metric must reach to activate this stage.
    """

    metric_value: float

class metric_event_curriculum:
    """Update an event term's params based on a metric value (exponential moving average)."""
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        """Initialise the curriculum from its term config and the live environment.

        Args:
            cfg: Curriculum term config whose ``params`` must contain ``event_name``
                (str), ``metric_name`` (str), and ``stages`` (list of
                MetricCurriculumStage); optionally ``alpha`` (float, default 0.01)
                for the EMA decay factor.
            env: The running environment used to resolve the event term config.

        Side Effects:
            - Resolves and caches the event term config from ``env.event_manager``.
            - Sorts ``self.stages`` in place by ascending ``metric_value``.
        """
        self.event_name: str = cfg.params["event_name"]
        self.metric_name: str = cfg.params["metric_name"]
        self.stages: list[MetricCurriculumStage] = cfg.params["stages"]

        self.alpha: float = cfg.params.get("alpha", 0.01)
        self.ema_metric = 0.0

        self._term_cfg = env.event_manager.get_term_cfg(self.event_name)
        self.stages = sorted(self.stages, key=lambda x: x["metric_value"])
        self._current_stage_idx = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the metric EMA and, if a new stage threshold is crossed, update the event term.

        Args:
            env: The running environment providing ``metrics_manager`` and ``event_manager``.
            env_ids: Indices of environments being stepped (unused; required by curriculum API).
            **kwargs: Additional keyword arguments forwarded by the curriculum manager (ignored).

        Returns:
            Dictionary with scalar diagnostic tensors:
            ``ema_value`` — current EMA of the monitored metric, and
            ``alpha`` — the EMA decay factor.

        Side Effects:
            - Updates ``self.ema_metric`` with the current step's mean metric value.
            - Advances ``self._current_stage_idx`` monotonically when thresholds are exceeded.
            - Merges the active stage's ``params`` into the event term config's ``params``.
        """
        metric_idx = env.metrics_manager.active_terms.index(self.metric_name)
        latest_vals = env.metrics_manager._step_values[:, metric_idx]
        current_mean = latest_vals.mean().item()

        self.ema_metric = (1.0 - self.alpha) * self.ema_metric + self.alpha * current_mean

        for i, stage in enumerate(self.stages):
            if self.ema_metric >= stage["metric_value"]:
                self._current_stage_idx = max(self._current_stage_idx, i)

        stage = self.stages[self._current_stage_idx]
        if "params" in stage:
            self._term_cfg.params.update(stage["params"])

        return {
            "ema_value": torch.tensor(self.ema_metric),
            "alpha": torch.tensor(self.alpha),
        }


class metric_reward_curriculum:
    """Update a reward term's weight based on a metric value (exponential moving average)."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        """Initialise the curriculum from its term config and the live environment.

        Args:
            cfg: Curriculum term config whose ``params`` must contain ``reward_name``
                (str), ``metric_name`` (str), and ``stages`` (list of
                MetricCurriculumStage); optionally ``alpha`` (float, default 0.01)
                for the EMA decay factor.
            env: The running environment used to resolve the reward term config.

        Side Effects:
            - Resolves and caches the reward term config from ``env.reward_manager``.
            - Sorts ``self.stages`` in place by ascending ``metric_value``.
        """
        self.reward_name: str = cfg.params["reward_name"]
        self.metric_name: str = cfg.params["metric_name"]
        self.stages: list[MetricCurriculumStage] = cfg.params["stages"]

        self.alpha: float = cfg.params.get("alpha", 0.01)
        self.ema_metric = 0.0

        self._term_cfg = env.reward_manager.get_term_cfg(self.reward_name)
        self.stages = sorted(self.stages, key=lambda x: x["metric_value"])
        self._current_stage_idx = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the metric EMA and, if a new stage threshold is crossed, update the reward term.

        Args:
            env: The running environment providing ``metrics_manager`` and ``reward_manager``.
            env_ids: Indices of environments being stepped (unused; required by curriculum API).
            **kwargs: Additional keyword arguments forwarded by the curriculum manager (ignored).

        Returns:
            Dictionary with scalar diagnostic tensors:
            ``ema_value`` — current EMA of the monitored metric,
            ``alpha`` — the EMA decay factor, and
            ``weight`` — the reward term's current weight (present only when the term has a weight
            attribute).

        Side Effects:
            - Updates ``self.ema_metric`` with the current step's mean metric value.
            - Advances ``self._current_stage_idx`` monotonically when thresholds are exceeded.
            - Overwrites the reward term config's ``weight`` if the active stage defines one.
            - Merges the active stage's ``params`` into the reward term config's ``params``.
        """
        metric_idx = env.metrics_manager.active_terms.index(self.metric_name)
        latest_vals = env.metrics_manager._step_values[:, metric_idx]
        current_mean = latest_vals.mean().item()

        self.ema_metric = (1.0 - self.alpha) * self.ema_metric + self.alpha * current_mean

        for i, stage in enumerate(self.stages):
            if self.ema_metric >= stage["metric_value"]:
                self._current_stage_idx = max(self._current_stage_idx, i)

        stage = self.stages[self._current_stage_idx]
        if "weight" in stage:
            self._term_cfg.weight = stage["weight"]
        if "params" in stage:
            self._term_cfg.params.update(stage["params"])

        result = {"ema_value": torch.tensor(self.ema_metric)}
        result["alpha"] = torch.tensor(self.alpha)
        if hasattr(self._term_cfg, "weight"):
            result["weight"] = torch.tensor(self._term_cfg.weight)

        return result
