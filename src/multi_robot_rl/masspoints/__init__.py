"""Masspoints module initialization and task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import masspoint_reach_env_cfg, multi_masspoint_reach_env_cfg
from .rl_cfg import masspoint_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Masspoint-Reach",
    env_cfg=masspoint_reach_env_cfg(),
    play_env_cfg=masspoint_reach_env_cfg(play=True),
    rl_cfg=masspoint_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id="Mjlab-Multi-Masspoint-Reach",
    env_cfg=multi_masspoint_reach_env_cfg(),
    play_env_cfg=multi_masspoint_reach_env_cfg(play=True),
    rl_cfg=masspoint_ppo_runner_cfg(),
)
