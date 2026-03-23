"""Event and reset functions."""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

def reset_goal_position(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, pos_range: tuple):
    """Randomize the goal position."""
    goal = env.scene[asset_cfg.name]
    num_envs_to_reset = len(env_ids)
    
    x = torch.empty(num_envs_to_reset, device=env.device).uniform_(pos_range[0][0], pos_range[0][1])
    y = torch.empty(num_envs_to_reset, device=env.device).uniform_(pos_range[1][0], pos_range[1][1])
    z = torch.ones(num_envs_to_reset, device=env.device) * pos_range[2][0] # Fixed Z for 2D
    new_pos = torch.stack([x, y, z], dim=-1)
    
    # Generate random positions + unrotated quaternions (since write_mocap_pose_to_sim expects 7D pose: pos + quat)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs_to_reset, 1)
    new_pose = torch.cat([new_pos, quats], dim=-1)
    goal.write_mocap_pose_to_sim(mocap_pose=new_pose, env_ids=env_ids)
