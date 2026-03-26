"""Observation functions."""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

def distance_to_goal(env, asset_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance between the asset and the goal."""
    asset_pos = env.scene[asset_cfg.name].data.joint_pos  # Joint positions for the two sliders gives actual XY
    goal_pos = env.scene[goal_cfg.name].data.root_link_pos_w
    # asset_pos: [num_envs, 2], goal_pos: [num_envs, 3]
    return torch.norm(asset_pos[:, :2] - goal_pos[:, :2], dim=-1).unsqueeze(-1)

def relative_goal_pos(env, asset_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg) -> torch.Tensor:
    """Relative 2D vector pointing from the asset to the goal."""
    asset_pos = env.scene[asset_cfg.name].data.joint_pos
    goal_pos = env.scene[goal_cfg.name].data.root_link_pos_w
    return goal_pos[:, :2] - asset_pos[:, :2]

def root_lin_vel_w_2d(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the root linear velocity in 2D."""
    # Since it's attached via slide joints, its velocity is its joint velocities
    return env.scene[asset_cfg.name].data.joint_vel[:, :2]

# ---------------------------------------------------------------------------
# Multi-entity observation helpers
# ---------------------------------------------------------------------------

def multi_masspoint_vel(env, masspoint_cfgs: list[SceneEntityCfg]) -> torch.Tensor:
    """Concatenated 2D velocities for N masspoints: [num_envs, N*2].

    The observations are ordered as [vel_0_x, vel_0_y, vel_1_x, vel_1_y, ...].
    """
    return torch.cat([root_lin_vel_w_2d(env, cfg) for cfg in masspoint_cfgs], dim=-1)

def multi_masspoint_relative_goal_pos(
    env,
    masspoint_cfgs: list[SceneEntityCfg],
    goal_cfgs: list[SceneEntityCfg],
) -> torch.Tensor:
    """Relative 2D vectors from each masspoint to each goal: [num_envs, N*M*2].

    Ordered as: [mp_0→goal_0, mp_0→goal_1, ..., mp_1→goal_0, ...].
    """
    parts = [
        relative_goal_pos(env, mp_cfg, goal_cfg)
        for mp_cfg in masspoint_cfgs
        for goal_cfg in goal_cfgs
    ]
    return torch.cat(parts, dim=-1)

def multi_masspoint_distance_to_goals(
    env,
    masspoint_cfgs: list[SceneEntityCfg],
    goal_cfgs: list[SceneEntityCfg],
) -> torch.Tensor:
    """Distances from each masspoint to each goal: [num_envs, N*M].

    Ordered as: [dist(mp_0, goal_0), dist(mp_0, goal_1), ..., dist(mp_1, goal_0), ...].
    """
    parts = [
        distance_to_goal(env, mp_cfg, goal_cfg)
        for mp_cfg in masspoint_cfgs
        for goal_cfg in goal_cfgs
    ]
    return torch.cat(parts, dim=-1)
