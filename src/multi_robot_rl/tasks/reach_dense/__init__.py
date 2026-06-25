"""Registers reach-dense task variants (sparse + PBRS dense reward, with and without curriculum)."""

from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import make_reach_dense_env
from multi_robot_rl.configs.rl_cfg import ppo_runner_cfg_reach_task

register_mjlab_task(
    task_id="reach-dense-curriculum",
    env_cfg=make_reach_dense_env(),
    play_env_cfg=make_reach_dense_env(play=True),
    rl_cfg=ppo_runner_cfg_reach_task(),
)

register_mjlab_task(
    task_id="reach-dense-no-curriculum",
    env_cfg=make_reach_dense_env(no_curriculum=True),
    play_env_cfg=make_reach_dense_env(play=True, no_curriculum=True),
    rl_cfg=ppo_runner_cfg_reach_task(),
)
