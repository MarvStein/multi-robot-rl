import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from multi_robot_rl.assets.robots.base import RobotConfig
import multi_robot_rl.configs.push_constants as push_constants

# =========================================================
# HELPERS
# =========================================================

def _get_ee_configs(env: ManagerBasedRlEnv, robots: list[RobotConfig]) -> list[SceneEntityCfg]:
    if not hasattr(env, "_push_ee_site_cfgs"):
        cfgs = [SceneEntityCfg(r.name, site_names=(r.end_effector_site,)) for r in robots]
        for cfg in cfgs:
            cfg.resolve(env.scene)
        env._push_ee_site_cfgs = list(cfgs)
    return env._push_ee_site_cfgs


def _get_cuboid_joint_cfgs(env: ManagerBasedRlEnv) -> list[SceneEntityCfg]:
    if not hasattr(env, "_cuboid_joint_cfgs"):
        cfgs = [
            SceneEntityCfg(f"cuboid_{i}", joint_names=("cuboid_x", "cuboid_y", "cuboid_yaw"))
            for i in range(push_constants.NUM_CUBOIDS)
        ]
        for cfg in cfgs:
            cfg.resolve(env.scene)
        env._cuboid_joint_cfgs = list(cfgs)
    return env._cuboid_joint_cfgs


# =========================================================
# OBSERVATIONS
# =========================================================

def cuboid_states_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Returns (num_envs, num_cuboids * 5): [x, y, z, sin_yaw, cos_yaw] per cuboid in world frame."""
    cfgs = _get_cuboid_joint_cfgs(env)
    parts = []
    for cfg in cfgs:
        qpos = env.scene[cfg.name].data.joint_pos[:, cfg.joint_ids]  # (num_envs, 3): x, y, yaw
        x = qpos[:, 0:1]
        y = qpos[:, 1:2]
        z = torch.full((env.num_envs, 1), push_constants.CUBOID_HZ, device=env.device)
        yaw = qpos[:, 2:3]
        parts.append(torch.cat([x, y, z, torch.sin(yaw), torch.cos(yaw)], dim=-1))
    return torch.cat(parts, dim=-1)

def target_poses_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Returns (num_envs, num_targets * 5): [x, y, z, sin_yaw, cos_yaw] per target."""
    if not hasattr(env, "target_poses"):
        num_targets = push_constants.NUM_CUBOIDS
        return torch.zeros(env.num_envs, num_targets * 5, device=env.device)

    poses = env.target_poses  # (num_envs, num_targets, 4): x, y, z, yaw — yaw→sin/cos expands to 5
    x   = poses[:, :, 0:1]
    y   = poses[:, :, 1:2]
    z   = poses[:, :, 2:3]
    yaw = poses[:, :, 3:4]
    flat = torch.cat([x, y, z, torch.sin(yaw), torch.cos(yaw)], dim=-1)  # (num_envs, num_targets, 5)
    return flat.flatten(start_dim=1)


# =========================================================
# TERMINATIONS
# =========================================================

def out_of_bounds(env: ManagerBasedRlEnv, robots: list[RobotConfig], **kwargs) -> torch.Tensor:
    """Terminate if any robot's EE leaves the allowed workspace."""
    ee_cfgs = _get_ee_configs(env, robots)
    out_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in ee_cfgs:
        ee_pos = env.scene[cfg.name].data.site_pos_w[:, cfg.site_ids, :].squeeze(1)  # (num_envs, 3)
        r = torch.norm(ee_pos[:, :2], dim=-1)
        z = ee_pos[:, 2]
        out_mask |= r > push_constants.OUT_OF_BOUNDS_RADIUS
        out_mask |= z > push_constants.OUT_OF_BOUNDS_HEIGHT
        out_mask |= z < 0.0
    return out_mask


# =========================================================
# REWARDS
# =========================================================

def cuboid_placed_reward(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """
    Sparse one-time reward when any cuboid is placed at a target pose, where "placed" means within both the XY
    POSITION_THRESHOLD (XY) and YAW_THRESHOLD. Any cuboid can satisfy any target.
    """
    if not hasattr(env, "target_poses"):
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_target_satisfied_mask"):
        _init_target_satisfied_mask(env, push_constants.NUM_CUBOIDS)

    cfgs = _get_cuboid_joint_cfgs(env)
    # Collect cuboid (x, y, yaw) — shape (num_envs, num_cuboids, 3)
    cuboid_states = torch.stack(
        [env.scene[cfg.name].data.joint_pos[:, cfg.joint_ids] for cfg in cfgs],
        dim=1,
    )  # (num_envs, N, 3): col 0=x, 1=y, 2=yaw

    cuboid_xy  = cuboid_states[:, :, :2]   # (num_envs, N, 2)
    cuboid_yaw = cuboid_states[:, :, 2]    # (num_envs, N)

    target_xy  = env.target_poses[:, :, :2]  # (num_envs, N, 2)
    target_yaw = env.target_poses[:, :, 3]   # (num_envs, N)

    # XY distances: (num_envs, num_cuboids, num_targets)
    pos_dists = torch.cdist(cuboid_xy, target_xy)

    # Angular differences: (num_envs, num_cuboids, num_targets)
    # Broadcast: cuboid_yaw (N,N_c) vs target_yaw (N,N_t)
    # TODO: double check angular distance calculation
    d_yaw = cuboid_yaw.unsqueeze(2) - target_yaw.unsqueeze(1)
    ang_dists = torch.abs((d_yaw + torch.pi) % (2 * torch.pi) - torch.pi)

    pos_ok = pos_dists < push_constants.POSITION_THRESHOLD   # (num_envs, N_c, N_t)
    yaw_ok = ang_dists < push_constants.YAW_THRESHOLD        # (num_envs, N_c, N_t)

    # A target is satisfied if any cuboid meets both criteria
    any_cuboid_satisfies = (pos_ok & yaw_ok).any(dim=1)      # (num_envs, N_t)

    newly_satisfied = any_cuboid_satisfies & ~env._target_satisfied_mask
    env._target_satisfied_mask |= newly_satisfied

    return newly_satisfied.float().sum(dim=1)


# =========================================================
# METRICS
# =========================================================

def targets_reached_fraction(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Mean fraction of targets currently satisfied across envs."""
    if not hasattr(env, "_target_satisfied_mask"):
        return torch.zeros(env.num_envs, device=env.device)
    return env._target_satisfied_mask.float().mean(dim=1)

# =========================================================
# EVENTS & RESETS
# =========================================================

def _init_target_poses(env: ManagerBasedRlEnv) -> None:
    if not hasattr(env, "target_poses"):
        env.target_poses = torch.zeros((env.num_envs, push_constants.NUM_CUBOIDS, 4), device=env.device)


def _init_target_satisfied_mask(env: ManagerBasedRlEnv) -> None:
    if not hasattr(env, "_target_satisfied_mask"):
        env._target_satisfied_mask = torch.zeros(
            (env.num_envs, push_constants.NUM_CUBOIDS), dtype=torch.bool, device=env.device
        )


def reset_cuboids_and_targets(
    env: ManagerBasedRlEnv,
    env_ids,
    **kwargs,
) -> None:
    """Reset cuboid positions/yaw and target poses for the given env_ids."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    num_cuboids = push_constants.NUM_CUBOIDS

    _init_target_poses(env)
    _init_target_satisfied_mask(env)
    env._target_satisfied_mask[env_ids] = False
    num = len(env_ids)

    # --- Reset cuboid joint states ---
    cfgs = _get_cuboid_joint_cfgs(env)

    r_c     = push_constants.CUBOID_SPAWN_RADIUS * torch.sqrt(torch.rand(num, num_cuboids, device=env.device))
    theta_c = 2.0 * torch.pi * torch.rand(num, num_cuboids, device=env.device)
    yaw_low, yaw_high = push_constants.CUBOID_YAW_SPAWN_RANGE
    yaw_c   = yaw_low + (yaw_high - yaw_low) * torch.rand(num, num_cuboids, device=env.device)
    x_c     = r_c * torch.cos(theta_c)
    y_c     = r_c * torch.sin(theta_c)

    for i, cfg in enumerate(cfgs):
        joint_pos = torch.stack([x_c[:, i], y_c[:, i], yaw_c[:, i]], dim=-1)
        joint_vel = torch.zeros_like(joint_pos)
        joint_ids = cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)
        env.scene[cfg.name].write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids)

    # --- Sample new target poses ---
    r_t     = push_constants.TARGET_SPAWN_RADIUS * torch.sqrt(torch.rand(num, num_cuboids, device=env.device))
    theta_t = 2.0 * torch.pi * torch.rand(num, num_cuboids, device=env.device)
    tyaw_low, tyaw_high = push_constants.TARGET_YAW_RANGE
    tyaw    = tyaw_low + (tyaw_high - tyaw_low) * torch.rand(num, num_cuboids, device=env.device)
    tx      = r_t * torch.cos(theta_t)
    ty      = r_t * torch.sin(theta_t)
    tz      = 0

    env.target_poses[env_ids, :, 0] = tx
    env.target_poses[env_ids, :, 1] = ty
    env.target_poses[env_ids, :, 2] = tz
    env.target_poses[env_ids, :, 3] = tyaw

    for t in range(num_cuboids):
        half_yaw = tyaw[:, t] * 0.5
        poses = torch.zeros((num, 7), device=env.device)
        poses[:, 0] = tx[:, t]
        poses[:, 1] = ty[:, t]
        poses[:, 2] = tz
        poses[:, 3] = torch.cos(half_yaw)   # qw
        poses[:, 6] = torch.sin(half_yaw)   # qz
        env.scene[f"push_target_{t}"].write_mocap_pose_to_sim(mocap_pose=poses, env_ids=env_ids)
