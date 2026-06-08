"""Registers reach task variants."""

from mjlab.rl.fast_sac import FastSACRunner
from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import make_reach_env
from multi_robot_rl.configs.rl_cfg import fast_sac_runner_cfg_reach_task, ppo_runner_cfg_reach_task

register_mjlab_task(
    task_id="reach",
    env_cfg=make_reach_env(),
    play_env_cfg=make_reach_env(play=True),
    rl_cfg=ppo_runner_cfg_reach_task(),
)

register_mjlab_task(
    task_id="reach-fast-sac",
    env_cfg=make_reach_env(),
    play_env_cfg=make_reach_env(play=True),
    rl_cfg=fast_sac_runner_cfg_reach_task(),
    runner_cls=FastSACRunner,
)

register_mjlab_task(
    task_id="reach-no-curriculum",
    env_cfg=make_reach_env(no_curriculum=True),
    play_env_cfg=make_reach_env(play=True, no_curriculum=True),
    rl_cfg=ppo_runner_cfg_reach_task(),
)
