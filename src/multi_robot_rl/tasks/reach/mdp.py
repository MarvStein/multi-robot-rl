import torch
from mjlab.envs import ManagerBasedRlEnv

from multi_robot_rl.assets.robots.base import RobotConfig
import multi_robot_rl.configs.reach_constants as reach_constants
import multi_robot_rl.common.quat_helpers as quat_helpers
from multi_robot_rl.common.mdp import get_ee_positions, get_ee_centroid_positions

def _init_reach_state(env: ManagerBasedRlEnv) -> None:
    if not hasattr(env, "goal_positions"):
        env.goal_positions = torch.zeros(
            (env.num_envs, reach_constants.NUM_GOALS, 3), device=env.device
        )
        env._goal_reached_mask = torch.zeros(
            (env.num_envs, reach_constants.NUM_GOALS), dtype=torch.bool, device=env.device
        )
        env._final_goal_reached_fraction = torch.zeros(env.num_envs, device=env.device)

# =========================================================
# OBSERVATIONS
# =========================================================

def goals_position_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Returns concatenated goal positions: (num_envs, NUM_GOALS * 3)"""
    _init_reach_state(env)
    return env.goal_positions.reshape(env.num_envs, -1)

def goal_reached_mask_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Returns the goal_reached_mask as float: (num_envs, NUM_GOALS)"""
    _init_reach_state(env)
    return env._goal_reached_mask.float()

# =========================================================
# TERMINATIONS
# =========================================================

def all_goals_reached(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return a tensor indicating in which envs all goals have been reached."""
    _init_reach_state(env)
    return env._goal_reached_mask.all(dim=1)

def out_of_bounds(env: ManagerBasedRlEnv, robots: list[RobotConfig], **kwargs) -> torch.Tensor:
    """Terminate if any robot's EE leaves the allowed cylindrical workspace."""
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
    **kwargs,
) -> torch.Tensor:
    """Sparse one-time reward when any robot newly reaches a goal."""
    ee_positions = get_ee_positions(env, robots)  # (num_envs, num_robots, 3)
    distances = torch.cdist(ee_positions, env.goal_positions)  # (num_envs, num_robots, NUM_GOALS)
    any_robot_reached = (distances < reach_constants.GOAL_REACH_THRESHOLD).any(dim=1)  # (num_envs, NUM_GOALS)

    newly_reached = any_robot_reached & ~env._goal_reached_mask
    env._goal_reached_mask |= newly_reached

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

    return newly_reached.float().sum(dim=1)  # (num_envs,)

# =========================================================
# METRICS
# =========================================================

def goal_reached_fraction(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Returns the fraction of goals reached at the end of the previous episode."""
    # Note: if we were to return env._goal_reached_mask.float().mean(dim=1),
    # then because mjlab time-averages metrics over the episode length,
    # it would reflect "the speed of reaching goals" as well.
    _init_reach_state(env)
    return env._final_goal_reached_fraction

# =========================================================
# EVENTS & RESETS
# =========================================================

def reset_goal_state(
    env: ManagerBasedRlEnv,
    env_ids,
    robots: list[RobotConfig],
    radius: float = reach_constants.GOAL_WORKSPACE_RADIUS,
    dz: float = reach_constants.GOAL_WORKSPACE_HEIGHT / 2.0,
    **kwargs,
) -> None:
    """Sample new goal positions in a cylinder centered on the EE centroid and update mocap markers.

    Goals are drawn uniformly within a cylinder of radius `radius` and half-height `dz` centered
    on the mean end-effector position, clipped to the workspace bounds.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    _init_reach_state(env)
    env._final_goal_reached_fraction[env_ids] = env._goal_reached_mask[env_ids].float().mean(dim=1)
    env._goal_reached_mask[env_ids] = False

    num_envs = len(env_ids)
    num_goals = reach_constants.NUM_GOALS

    # Uniform distribution in a disk: r = R * sqrt(U), theta = 2*pi*U
    r = radius * torch.sqrt(torch.rand(num_envs, num_goals, device=env.device))
    theta = 2.0 * torch.pi * torch.rand(num_envs, num_goals, device=env.device)
    z_offset = (2.0 * torch.rand(num_envs, num_goals, device=env.device) - 1.0) * dz

    centroids = get_ee_centroid_positions(env, robots, env_ids)  # (num_envs, 3)
    xy = torch.stack([
        centroids[:, 0:1] + r * torch.cos(theta),
        centroids[:, 1:2] + r * torch.sin(theta),
    ], dim=-1)  # (num_envs, num_goals, 2)
    xy_norm = torch.norm(xy, dim=-1, keepdim=True).clamp(min=1e-6)
    xy = xy * (xy_norm.clamp(max=reach_constants.GOAL_WORKSPACE_RADIUS) / xy_norm)

    env.goal_positions[env_ids, :, 0] = xy[:, :, 0]
    env.goal_positions[env_ids, :, 1] = xy[:, :, 1]
    env.goal_positions[env_ids, :, 2] = (centroids[:, 2:3] + z_offset).clamp(0.0, reach_constants.GOAL_WORKSPACE_HEIGHT)

    for i in range(reach_constants.NUM_GOALS):
        poses = quat_helpers.position_to_pose(
            env.goal_positions[env_ids, i, 0],
            env.goal_positions[env_ids, i, 1],
            env.goal_positions[env_ids, i, 2],
        )
        env.scene[f"goal_{i}"].write_mocap_pose_to_sim(mocap_pose=poses, env_ids=env_ids)