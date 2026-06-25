"""Dense-reward MDP functions for the reach-dense task (PBRS formulation)."""

import torch
from typing import Any
from mjlab.envs import ManagerBasedRlEnv

from multi_robot_rl.assets.robots.base import RobotConfig
import multi_robot_rl.configs.reach_constants as reach_constants
from multi_robot_rl.common.mdp import get_ee_positions

from multi_robot_rl.tasks.reach.mdp import (
    _init_reach_state,
    goals_position_obs,
    goal_reached_mask_obs,
    all_goals_reached,
    out_of_bounds,
    goal_reached_fraction,
    robot_goal_reached_fraction,
    reset_goal_state as _sparse_reset_goal_state,
)

# Must match the RL algorithm's discount factor (0.99 in ppo_runner_cfg_reach_task).
_GAMMA = 0.99


def _init_dense_state(env: ManagerBasedRlEnv) -> None:
    if not hasattr(env, "_prev_potential"):
        env._prev_potential = torch.zeros(env.num_envs, device=env.device)


def _potential(distances: torch.Tensor, reached_mask: torch.Tensor) -> torch.Tensor:
    """Compute phi(s): the negative sum over unreached goals of the closest-robot distance.

    For each unreached goal the closest robot distance is taken, then those distances
    are summed and negated. The result is non-positive and increases toward zero as
    robots approach unreached goals.

    Args:
        distances:    (num_envs, num_robots, NUM_GOALS) pairwise EE-to-goal distances.
        reached_mask: (num_envs, NUM_GOALS) bool, True for already-reached goals.

    Returns:
        (num_envs,) potential values, non-positive.
    """
    min_dist_per_goal, _ = distances.min(dim=1)          # (num_envs, NUM_GOALS)
    unreached = ~reached_mask                             # (num_envs, NUM_GOALS)
    return -(min_dist_per_goal * unreached.float()).sum(dim=1)  # (num_envs,)


def reset_goal_state_dense(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    play: bool,
    radius: float = reach_constants.GOAL_WORKSPACE_RADIUS,
    dz: float = reach_constants.GOAL_WORKSPACE_HEIGHT / 2.0,
    **kwargs: Any,
) -> None:
    """Reset goal state and clear the cached PBRS potential for reset environments.

    Delegates to the sparse reset_goal_state for all goal-position and mask logic,
    then zeros env._prev_potential for the affected environments so the first-step
    shaping is gamma * phi(s_0) rather than gamma * phi(s_0) - phi(end of last episode).
    """
    _sparse_reset_goal_state(env, env_ids, play, radius=radius, dz=dz, **kwargs)
    _init_dense_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env._prev_potential[env_ids] = 0.0


def pbrs_dense_reward(
    env: ManagerBasedRlEnv,
    robots: list[RobotConfig],
    gamma: float = _GAMMA,
    **kwargs: Any,
) -> torch.Tensor:
    """PBRS shaping term: F(s, s') = gamma * phi(s') - phi(s).

    phi(s) is the negative sum over unreached goals of the minimum end-effector
    distance from any robot to that goal. It is non-positive and increases toward
    zero as robots get closer to unreached goals.

    Intended to be used alongside goal_reached_reward, which owns the mask-update
    side-effects. The effective mask for the potential is computed as the union of
    the stored mask and any goals reached in this step, making the result
    ordering-independent with respect to goal_reached_reward.

    Args:
        env:    The environment instance.
        robots: Robot configurations used to compute end-effector positions.
        gamma:  Discount factor; must match the RL algorithm (default 0.99).

    Returns:
        Tensor of shape (num_envs,) with the PBRS shaping reward for each environment.
    """
    _init_reach_state(env)
    _init_dense_state(env)

    ee_positions = get_ee_positions(env, robots)                          # (num_envs, num_robots, 3)
    distances = torch.cdist(ee_positions, env.goal_positions)             # (num_envs, num_robots, NUM_GOALS)

    # Effective mask: union of the stored mask and goals reached this step.
    # Ordering-independent: correct whether goal_reached_reward ran before or after.
    any_robot_reached = (distances < reach_constants.GOAL_REACH_THRESHOLD).any(dim=1)
    effective_mask = env._goal_reached_mask | any_robot_reached           # (num_envs, NUM_GOALS)

    current_potential = _potential(distances, effective_mask)             # (num_envs,)
    shaping = gamma * current_potential - env._prev_potential
    # Zero shaping on the first step of each episode: _prev_potential was reset to 0
    # (not φ(s₀)), so the signal would be spuriously large. Step 2 onward is correct
    # because _prev_potential is properly set to φ(s₁) after this call.
    shaping = shaping.masked_fill(env.episode_length_buf == 1, 0.0)
    env._prev_potential = current_potential.clone()

    return shaping  # (num_envs,)
