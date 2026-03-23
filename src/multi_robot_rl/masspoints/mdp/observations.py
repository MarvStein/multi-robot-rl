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
