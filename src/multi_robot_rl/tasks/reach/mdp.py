"""Reward, observation, termination, metric, and reset functions for the multi-robot reach task."""

import torch
from typing import Any
from mjlab.envs import ManagerBasedRlEnv

from multi_robot_rl.assets.robots.base import RobotConfig
import multi_robot_rl.configs.reach_constants as reach_constants
import multi_robot_rl.common.quat_helpers as quat_helpers
from multi_robot_rl.common.mdp import get_ee_positions

def _init_reach_state(env: ManagerBasedRlEnv) -> None:
    """Initialize per-environment reach state tensors on the env object if not already present.

    Args:
        env: The environment instance to initialize.

    Side Effects:
        - Sets env.goal_positions to a zero tensor of shape (num_envs, NUM_GOALS, 3) if absent.
        - Sets env._goal_reached_mask to a boolean zero tensor of shape (num_envs, NUM_GOALS) if absent.
        - Sets env._final_goal_reached_fraction to a zero tensor of shape (num_envs,) if absent.
    """
    if not hasattr(env, "goal_positions"):
        env.goal_positions = torch.zeros(
            (env.num_envs, reach_constants.NUM_GOALS, 3), device=env.device
        )
        env._goal_reached_mask = torch.zeros(
            (env.num_envs, reach_constants.NUM_GOALS), dtype=torch.bool, device=env.device
        )
        env._final_goal_reached_fraction = torch.zeros(env.num_envs, device=env.device)


def _init_robot_contribution_state(env: ManagerBasedRlEnv, num_robots: int) -> None:
    """Initialize per-robot goal contribution tensors on the env object if not already present.

    Args:
        env: The environment instance to initialize.
        num_robots: Number of robots in the environment.

    Side Effects:
        - Sets env._robot_goal_counts to a zero tensor of shape (num_envs, num_robots) if absent.
        - Sets env._final_robot_goal_fractions to a zero tensor of shape (num_envs, num_robots) if absent.
    """
    if not hasattr(env, "_robot_goal_counts"):
        env._robot_goal_counts = torch.zeros((env.num_envs, num_robots), device=env.device)
        env._final_robot_goal_fractions = torch.zeros((env.num_envs, num_robots), device=env.device)

# =========================================================
# OBSERVATIONS
# =========================================================

def goals_position_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return concatenated goal positions as a flat observation vector.

    Args:
        env: The environment instance.

    Returns:
        Tensor of shape (num_envs, NUM_GOALS * 3) containing the (x, y, z) position
        of every goal, concatenated along the last dimension.
    """
    _init_reach_state(env)
    return env.goal_positions.reshape(env.num_envs, -1)

def goal_reached_mask_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return the goal-reached mask cast to float as an observation.

    Args:
        env: The environment instance.

    Returns:
        Tensor of shape (num_envs, NUM_GOALS) with 1.0 for each goal that has been
        reached in the current episode and 0.0 otherwise.
    """
    _init_reach_state(env)
    return env._goal_reached_mask.float()

# =========================================================
# TERMINATIONS
# =========================================================

def all_goals_reached(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return a boolean tensor indicating which environments have reached every goal.

    Args:
        env: The environment instance.

    Returns:
        Boolean tensor of shape (num_envs,) that is True for each environment in
        which all goals have been reached.
    """
    _init_reach_state(env)
    return env._goal_reached_mask.all(dim=1)

def out_of_bounds(env: ManagerBasedRlEnv, robots: list[RobotConfig], **kwargs) -> torch.Tensor:
    """Return a boolean tensor indicating which environments have a robot end-effector outside the allowed cylindrical workspace.

    Args:
        env: The environment instance.
        robots: List of robot configurations whose end-effector positions are checked.

    Returns:
        Boolean tensor of shape (num_envs,) that is True for each environment in
        which at least one robot's end-effector exceeds the workspace radius or
        height bounds.
    """
    ee_positions = get_ee_positions(env, robots)  # (num_envs, num_robots, 3)
    r = torch.norm(ee_positions[:, :, :2], dim=-1)  # (num_envs, num_robots)
    z = ee_positions[:, :, 2]
    out = (r > reach_constants.OUT_OF_BOUNDS_RADIUS) | (z > reach_constants.OUT_OF_BOUNDS_HEIGHT) | (z < 0.0)
    return out.any(dim=1)

# =========================================================
# REWARDS
# =========================================================

def goal_reached_reward(
    env: ManagerBasedRlEnv,
    robots: list[RobotConfig],
    play: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Return a sparse one-time reward for each goal newly reached in this step.

    Each newly reached goal contributes 1/NUM_GOALS to the reward, so the maximum
    reward per step is 1.0 when all remaining goals are reached simultaneously.

    Args:
        env: The environment instance.
        robots: List of robot configurations used to compute end-effector positions.
        play: When True, markers for newly reached goals are hidden by moving them
            below the floor for visual feedback during interactive evaluation.

    Returns:
        Tensor of shape (num_envs,) containing the reward for each environment.

    Side Effects:
        - Updates env._goal_reached_mask with newly reached goals.
        - In play mode, writes hidden mocap poses to the scene for reached goal markers.
    """
    ee_positions = get_ee_positions(env, robots)  # (num_envs, num_robots, 3)
    distances = torch.cdist(ee_positions, env.goal_positions)  # (num_envs, num_robots, NUM_GOALS)
    any_robot_reached = (distances < reach_constants.GOAL_REACH_THRESHOLD).any(dim=1)  # (num_envs, NUM_GOALS)

    newly_reached = any_robot_reached & ~env._goal_reached_mask
    env._goal_reached_mask |= newly_reached

    _init_robot_contribution_state(env, len(robots))
    if newly_reached.any():
        closest_robot = distances.argmin(dim=1)  # (num_envs, NUM_GOALS)
        env._robot_goal_counts.scatter_add_(1, closest_robot, newly_reached.float())

    if play and newly_reached.any():
        # hide markers of (newly) reached goals by moving them below the floor (purely visual)
        for i in range(reach_constants.NUM_GOALS):
            env_ids = newly_reached[:, i].nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                n = env_ids.numel()
                hidden_poses = quat_helpers.position_to_pose(
                    torch.zeros(n, device=env.device),
                    torch.zeros(n, device=env.device),
                    torch.full((n,), -100.0, device=env.device),
                )
                env.scene[f"goal_{i}"].write_mocap_pose_to_sim(mocap_pose=hidden_poses, env_ids=env_ids)

    return newly_reached.float().sum(dim=1) / reach_constants.NUM_GOALS  # (num_envs,)

# =========================================================
# METRICS
# =========================================================

def goal_reached_fraction(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return the fraction of goals that were reached at the end of the previous episode.

    The value is taken from env._final_goal_reached_fraction (snapshotted at reset)
    rather than computed live from env._goal_reached_mask. If the live mask were used,
    mjlab's per-step time-averaging of metrics would blend in partial episode progress,
    effectively measuring "speed of reaching goals" instead of final coverage.

    Args:
        env: The environment instance.

    Returns:
        Tensor of shape (num_envs,) in [0.0, 1.0] representing the fraction of goals
        reached by the end of each environment's most recently completed episode.
    """
    _init_reach_state(env)
    return env._final_goal_reached_fraction

def robot_goal_reached_fraction(env: ManagerBasedRlEnv, robot_index: int, **kwargs) -> torch.Tensor:
    """Return the fraction of goals attributed to a specific robot at the end of the previous episode.

    Per-robot fractions sum to goal_reached_fraction across all robots by construction.

    Args:
        env: The environment instance.
        robot_index: Zero-based index of the robot whose contribution to return.

    Returns:
        Tensor of shape (num_envs,) in [0.0, 1.0].
    """
    _init_reach_state(env)
    if not hasattr(env, "_final_robot_goal_fractions"):
        return torch.zeros(env.num_envs, device=env.device)
    return env._final_robot_goal_fractions[:, robot_index]

# =========================================================
# EVENTS & RESETS
# =========================================================

def reset_goal_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    play: bool,
    radius: float = reach_constants.GOAL_WORKSPACE_RADIUS,
    dz: float = reach_constants.GOAL_WORKSPACE_HEIGHT / 2.0,
    **kwargs: Any,
) -> None:
    """Sample new goal positions in a cylinder centered on the workspace and update mocap markers.

    Args:
        env: The environment instance.
        env_ids: Indices of the environments to reset. If None, all environments are reset.
        play: When True, radius and dz are overridden to 1.0 to disable curriculum
            scaling and sample goals across the full workspace during evaluation.
        radius: Fractional radius in [0, 1] of the sampling cylinder relative to
            GOAL_WORKSPACE_RADIUS.
        dz: Fractional half-height in [0, 1] of the sampling cylinder relative to
            (GOAL_WORKSPACE_HEIGHT - GOAL_WORKSPACE_MIN_HEIGHT) / 2.

    Side Effects:
        - Snapshots the current goal_reached_mask into env._final_goal_reached_fraction for the reset environments.
        - Clears env._goal_reached_mask for the reset environments.
        - Updates env.goal_positions with newly sampled positions.
        - Writes new mocap poses to all goal marker scene entities for the reset environments.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    _init_reach_state(env)
    env._final_goal_reached_fraction[env_ids] = env._goal_reached_mask[env_ids].float().mean(dim=1)
    if hasattr(env, "_robot_goal_counts"):
        env._final_robot_goal_fractions[env_ids] = env._robot_goal_counts[env_ids] / reach_constants.NUM_GOALS
        env._robot_goal_counts[env_ids] = 0.0
    env._goal_reached_mask[env_ids] = False

    num_envs = len(env_ids)
    num_goals = reach_constants.NUM_GOALS

    if play:
        # override radius and dz to disable curriculum in evaluation (sample goals in the full workspace from the start)
        radius = 1.0
        dz = 1.0

    # Uniform distribution in a disk: r = R * sqrt(U), theta = 2*pi*U
    r = radius * reach_constants.GOAL_WORKSPACE_RADIUS * torch.sqrt(torch.rand(num_envs, num_goals, device=env.device))
    theta = 2.0 * torch.pi * torch.rand(num_envs, num_goals, device=env.device)
    z_half = (reach_constants.GOAL_WORKSPACE_HEIGHT - reach_constants.GOAL_WORKSPACE_MIN_HEIGHT) / 2.0
    z_offset = (2.0 * torch.rand(num_envs, num_goals, device=env.device) - 1.0) * dz * z_half

    center_z = reach_constants.GOAL_WORKSPACE_MIN_HEIGHT + z_half
    env.goal_positions[env_ids, :, 0] = r * torch.cos(theta)
    env.goal_positions[env_ids, :, 1] = r * torch.sin(theta)
    env.goal_positions[env_ids, :, 2] = center_z + z_offset

    for i in range(reach_constants.NUM_GOALS):
        poses = quat_helpers.position_to_pose(
            env.goal_positions[env_ids, i, 0],
            env.goal_positions[env_ids, i, 1],
            env.goal_positions[env_ids, i, 2],
        )
        env.scene[f"goal_{i}"].write_mocap_pose_to_sim(mocap_pose=poses, env_ids=env_ids)