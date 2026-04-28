"""Registers type task variants."""

from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import make_type_env
from multi_robot_rl.configs.rl_cfg import ppo_runner_cfg_type_task

register_mjlab_task(
    task_id="type",
    env_cfg=make_type_env(),
    play_env_cfg=make_type_env(play=True),
    rl_cfg=ppo_runner_cfg_type_task(),
)
