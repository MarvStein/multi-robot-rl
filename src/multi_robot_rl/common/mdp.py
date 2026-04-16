import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnv

def action_magnitude_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Action magnitude generic penalty using L2 norm."""
    return torch.norm(env.action_manager.action, dim=-1)

def action_change_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions (L2 norm)."""
    # TODO: check mjlab.envs.mdp.rewards for built-in
    raise NotImplementedError()

def robot_collision_penalty(env: ManagerBasedRlEnv, robot_prefix: str) -> torch.Tensor:
    """Generic penalty if there's a collision on the specific robot"""
    # TODO implement robot_collision_penalty
    raise NotImplementedError()

def joint_pos_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the absolute joint positions of the asset."""
    # Assuming joint_ids are either not provided (all) or slice
    if asset_cfg.joint_ids is None:
        return env.scene[asset_cfg.name].data.joint_pos
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

def joint_vel_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the absolute joint velocities of the asset."""
    if asset_cfg.joint_ids is None:
        return env.scene[asset_cfg.name].data.joint_vel
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]