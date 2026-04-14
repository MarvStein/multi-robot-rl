"""Masspoints module initialization and task registration."""

import os

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import masspoint_keyboard_env_cfg, masspoint_multi_reach_env_cfg, masspoint_reach_env_cfg
from .rl_cfg import (
    ppo_runner_cfg_masspoint_reach,
    ppo_runner_cfg_masspoint_multi_reach,
    ppo_runner_cfg_masspoint_keyboard
)

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)

register_mjlab_task(
    task_id="Mjlab-Masspoint-Reach",
    env_cfg=masspoint_reach_env_cfg(),
    play_env_cfg=masspoint_reach_env_cfg(play=True),
    rl_cfg=ppo_runner_cfg_masspoint_reach(),
)

register_mjlab_task(
    task_id="Mjlab-Masspoint-MultiReach",
    env_cfg=masspoint_multi_reach_env_cfg(
        num_masspoints=_env_int("MRRL_NUM_MASSPOINTS", 2),
        num_goals=_env_int("MRRL_NUM_GOALS", 2),
        goal_reach_threshold=_env_float("MRRL_GOAL_REACH_THRESHOLD", 0.03),
        goal_respawn_delay_steps=_env_int("MRRL_GOAL_RESPAWN_DELAY_STEPS", 10),
    ),
    play_env_cfg=masspoint_multi_reach_env_cfg(
        play=True,
        num_masspoints=_env_int("MRRL_NUM_MASSPOINTS", 2),
        num_goals=_env_int("MRRL_NUM_GOALS", 2),
        goal_reach_threshold=_env_float("MRRL_GOAL_REACH_THRESHOLD", 0.03),
        goal_respawn_delay_steps=_env_int("MRRL_GOAL_RESPAWN_DELAY_STEPS", 10),
    ),
    rl_cfg=ppo_runner_cfg_masspoint_multi_reach(),
)

register_mjlab_task(
    task_id="Mjlab-Masspoint-Keyboard",
    env_cfg=masspoint_keyboard_env_cfg(),
    play_env_cfg=masspoint_keyboard_env_cfg(play=True),
    rl_cfg=ppo_runner_cfg_masspoint_keyboard(),
)
