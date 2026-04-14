"""Observation functions."""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
import multi_robot_rl.masspoints.keyboard_constants as kc


def _stack_masspoint_state(env, masspoint_names: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stacked masspoint 2D joint positions and velocities."""
    pos_2d = [env.scene[name].data.joint_pos[:, :2] for name in masspoint_names]
    vel_2d = [env.scene[name].data.joint_vel[:, :2] for name in masspoint_names]
    return torch.stack(pos_2d, dim=1), torch.stack(vel_2d, dim=1)


def _stack_goal_pos(env, goal_names: tuple[str, ...]) -> torch.Tensor:
    """Return stacked goal 2D positions."""
    goal_pos_2d = [env.scene[name].data.root_link_pos_w[:, :2] for name in goal_names]
    return torch.stack(goal_pos_2d, dim=1)


def centralized_state(
    env,
    masspoint_names: tuple[str, ...],
    goal_names: tuple[str, ...],
    include_goal_activity: bool = True,
) -> torch.Tensor:
    """Flattened centralized observation with fixed order for all masspoints and goals."""
    mp_pos, mp_vel = _stack_masspoint_state(env, masspoint_names)
    goal_pos = _stack_goal_pos(env, goal_names)

    terms = [
        mp_pos.reshape(env.num_envs, -1),
        mp_vel.reshape(env.num_envs, -1),
        goal_pos.reshape(env.num_envs, -1),
    ]
    if include_goal_activity:
        active = getattr(env, "_multi_goal_active", None)
        if active is None:
            active = torch.ones(env.num_envs, len(goal_names), device=env.device, dtype=torch.float32)
        terms.append(active.float())
    return torch.cat(terms, dim=-1)

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

def keyboard_state_obs(env, masspoint_names: tuple[str, ...], **kwargs):
    qpos_n = env.scene["keyboard"].data.joint_pos[:, :kc.TOTAL_KEYS]
    if not hasattr(env, "active_key"):
        active_targets = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.long)
        next_targets = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.long)
    else:
        active_targets = env.active_key.unsqueeze(-1)
        next_targets = env.next_key.unsqueeze(-1)
        
    mp_pos = torch.stack([env.scene[name].data.joint_pos[:, :3] for name in masspoint_names], dim=1)
    mp_vel_list = []
    for name in masspoint_names:
        if hasattr(env.scene[name].data, "joint_vel"):
            mp_vel_list.append(env.scene[name].data.joint_vel[:, :3])
        else:
            mp_vel_list.append(torch.zeros_like(env.scene[name].data.joint_pos[:, :3]))
    mp_vel = torch.stack(mp_vel_list, dim=1)

    terms = [
        qpos_n,
        active_targets,
        next_targets,
        mp_pos.reshape(env.num_envs, -1),
        mp_vel.reshape(env.num_envs, -1),
    ]
        
    return torch.cat(terms, dim=-1)
