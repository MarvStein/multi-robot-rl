"""Reward, observation, termination, metric, and reset functions for the multi-robot push task."""

import torch
from typing import Any
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from multi_robot_rl.assets.robots.base import RobotConfig
import multi_robot_rl.configs.push_constants as push_constants
from multi_robot_rl.common.mdp import get_ee_positions

# Maximum rejection-sampling attempts per object per reset.
# If exhausted, the pre-seeded fallback position (a valid in-workspace sample) is kept,
# which may violate separation constraints but prevents infinite loops.
_MAX_SPAWN_TRIES = 30


def _get_cuboid_joint_cfgs(env: ManagerBasedRlEnv) -> list[SceneEntityCfg]:
    """Lazily create and cache resolved SceneEntityCfg objects for all cuboid joints.

    Args:
        env: The environment instance used to resolve scene entities.

    Returns:
        List of resolved SceneEntityCfg objects, one per cuboid, covering the
        (cuboid_x, cuboid_y, cuboid_yaw) joints.

    Side Effects:
        - Sets ``env._cuboid_joint_cfgs`` on first call; subsequent calls return
          the cached list without re-resolving.
    """
    if not hasattr(env, "_cuboid_joint_cfgs"):
        cfgs = [
            SceneEntityCfg(f"cuboid_{i}", joint_names=("cuboid_x", "cuboid_y", "cuboid_yaw"))
            for i in range(push_constants.NUM_CUBOIDS)
        ]
        for cfg in cfgs:
            cfg.resolve(env.scene)
        env._cuboid_joint_cfgs = list(cfgs)
    return env._cuboid_joint_cfgs


def _init_push_state(env: ManagerBasedRlEnv) -> None:
    """Initialize push-task state tensors on the environment if not already present.

    Args:
        env: The environment instance on which state tensors are initialised.

    Side Effects:
        - Sets ``env.target_poses`` to a zero tensor of shape
          (num_envs, NUM_CUBOIDS, 4) if absent.
        - Sets ``env._target_satisfied_mask`` to a boolean zero tensor of shape
          (num_envs, NUM_CUBOIDS) if absent.
        - Sets ``env._final_target_reached_fraction`` to a zero tensor of shape
          (num_envs,) if absent.
    """
    if not hasattr(env, "_target_satisfied_mask"):
        env.target_poses = torch.zeros(
            (env.num_envs, push_constants.NUM_CUBOIDS, 4), device=env.device
        )
        env._target_satisfied_mask = torch.zeros(
            (env.num_envs, push_constants.NUM_CUBOIDS), dtype=torch.bool, device=env.device
        )
        env._final_target_reached_fraction = torch.zeros(env.num_envs, device=env.device)
        env._final_episode_length = torch.zeros(env.num_envs, device=env.device)


# =========================================================
# OBSERVATIONS
# =========================================================

def cuboid_states_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return the pose of every cuboid encoded as [x, y, z, sin_yaw, cos_yaw] in the world frame.

    Args:
        env: The environment instance providing scene data.
        **kwargs: Unused; accepted for compatibility with the observation manager interface.

    Returns:
        Tensor of shape (num_envs, num_cuboids * 5) containing the flattened
        per-cuboid pose features.
    """
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
    """Return every target pose encoded as [x, y, z, sin_yaw, cos_yaw].

    Args:
        env: The environment instance providing target pose state.
        **kwargs: Unused; accepted for compatibility with the observation manager interface.

    Returns:
        Tensor of shape (num_envs, num_targets * 5) containing the flattened
        per-target pose features.
    """
    _init_push_state(env)
    poses = env.target_poses  # (num_envs, num_targets, 4): x, y, z, yaw
    x   = poses[:, :, 0:1]
    y   = poses[:, :, 1:2]
    z   = poses[:, :, 2:3]
    yaw = poses[:, :, 3:4]
    flat = torch.cat([x, y, z, torch.sin(yaw), torch.cos(yaw)], dim=-1)  # (num_envs, num_targets, 5)
    return flat.flatten(start_dim=1)

def target_satisfied_mask_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return the per-target satisfaction mask cast to float.

    Args:
        env: The environment instance providing target satisfaction state.
        **kwargs: Unused; accepted for compatibility with the observation manager interface.

    Returns:
        Float tensor of shape (num_envs, NUM_CUBOIDS) where 1.0 means the
        corresponding target has been satisfied and 0.0 means it has not.
    """
    _init_push_state(env)
    return env._target_satisfied_mask.float()


# =========================================================
# TERMINATIONS
# =========================================================

def out_of_bounds(env: ManagerBasedRlEnv, robots: list[RobotConfig], **kwargs) -> torch.Tensor:
    """Return a mask indicating which environments have a robot end-effector outside the allowed workspace.

    An end-effector is considered out of bounds when its XY distance from the
    origin exceeds OUT_OF_BOUNDS_RADIUS, its height exceeds OUT_OF_BOUNDS_HEIGHT,
    or its height drops below zero.

    Args:
        env: The environment instance providing scene and simulation state.
        robots: List of robot configurations used to locate end-effector sites.
        **kwargs: Unused; accepted for compatibility with termination/reward manager interfaces.

    Returns:
        Boolean tensor of shape (num_envs,) that is True for every environment
        in which at least one robot's end-effector violates the workspace bounds.
    """
    ee_positions = get_ee_positions(env, robots)  # (num_envs, num_robots, 3)
    r = torch.norm(ee_positions[:, :, :2], dim=-1)
    z = ee_positions[:, :, 2]
    out_mask = (r > push_constants.OUT_OF_BOUNDS_RADIUS) | (z > push_constants.OUT_OF_BOUNDS_HEIGHT) | (z < 0.0)
    return out_mask.any(dim=1)

def all_targets_reached(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return a mask indicating which environments have all push targets satisfied.

    Args:
        env: The environment instance providing target satisfaction state.
        **kwargs: Unused; accepted for compatibility with the termination manager interface.

    Returns:
        Boolean tensor of shape (num_envs,) that is True for every environment
        in which every target in ``_target_satisfied_mask`` is True.
    """
    _init_push_state(env)
    return env._target_satisfied_mask.all(dim=1)


# =========================================================
# REWARDS
# =========================================================

def cuboid_placed_reward(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return a sparse one-time reward for each newly satisfied push target.

    A target is considered satisfied when any cuboid is within POSITION_THRESHOLD
    (XY distance) and YAW_THRESHOLD (angular distance) of it.  Each newly satisfied
    target contributes 1/NUM_CUBOIDS to the reward, so the total reward reaches 1.0
    when all targets are satisfied.  Any cuboid can satisfy any target.

    Args:
        env: The environment instance providing scene and push-task state.
        **kwargs: Unused; accepted for compatibility with the reward manager interface.

    Returns:
        Float tensor of shape (num_envs,) with the per-environment reward for
        targets newly satisfied on the current step.

    Side Effects:
        - Updates ``env._target_satisfied_mask`` in-place by OR-ing in any targets
          newly satisfied on this step.
    """
    _init_push_state(env)

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
    d_yaw = cuboid_yaw.unsqueeze(2) - target_yaw.unsqueeze(1)
    ang_dists = torch.abs((d_yaw + torch.pi) % (2 * torch.pi) - torch.pi)

    pos_ok = pos_dists < push_constants.POSITION_THRESHOLD   # (num_envs, N_c, N_t)
    yaw_ok = ang_dists < push_constants.YAW_THRESHOLD        # (num_envs, N_c, N_t)

    # A target is satisfied if any cuboid meets both criteria
    any_cuboid_satisfies = (pos_ok & yaw_ok).any(dim=1)      # (num_envs, N_t)

    newly_satisfied = any_cuboid_satisfies & ~env._target_satisfied_mask
    env._target_satisfied_mask |= newly_satisfied

    return newly_satisfied.float().sum(dim=1) / push_constants.NUM_CUBOIDS  # (num_envs,)


# =========================================================
# METRICS
# =========================================================

def targets_reached_fraction(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return the fraction of targets that were satisfied at the end of the previous episode.

    The value is frozen at episode end by ``reset_cuboids_and_targets`` and held
    constant until the next reset, making it suitable for use as a curriculum metric.

    Args:
        env: The environment instance providing push-task state.
        **kwargs: Unused; accepted for compatibility with the metrics manager interface.

    Returns:
        Float tensor of shape (num_envs,) with values in [0, 1] representing the
        per-environment fraction of targets satisfied in the episode that just ended.
    """
    _init_push_state(env)
    return env._final_target_reached_fraction


def targets_per_second(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Return total targets satisfied per second at the end of the previous episode.

    Args:
        env: The environment instance providing push-task state.
        **kwargs: Unused; accepted for compatibility with the metrics manager interface.

    Returns:
        Float tensor of shape (num_envs,) with targets/second for the most recently
        completed episode.
    """
    _init_push_state(env)
    episode_seconds = (env._final_episode_length * env.step_dt).clamp(min=env.step_dt)
    targets_reached = env._final_target_reached_fraction * push_constants.NUM_CUBOIDS
    return targets_reached / episode_seconds

# =========================================================
# EVENTS & RESETS
# =========================================================

def reset_cuboids_and_targets(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    play: bool = False,
    cuboid_distance_fraction: float = 1.0,
    **kwargs: Any,
) -> None:
    """Reset cuboid positions/yaw and target poses for the given environments.

    Targets are always sampled uniformly over the full workspace (TARGET_SPAWN_RADIUS),
    with pairwise rejection to keep targets at least TARGET_MIN_SEPARATION apart
    (= 2 * POSITION_THRESHOLD, so no single cuboid can simultaneously satisfy two targets).
    Each cuboid i is then sampled uniformly within the workspace disk, rejecting positions
    that are (a) further than ``cuboid_distance_fraction * 2 * TARGET_SPAWN_RADIUS`` from
    paired target i, or (b) closer than CUBOID_MIN_SEPARATION (bounding-circle diameter)
    to any already-placed cuboid.  At fraction=1 constraint (a) equals the workspace
    diameter and never rejects, giving purely uniform sampling.
    The cuboid–target pairing is only used for initialisation — during the episode any
    cuboid can satisfy any target.

    Args:
        env: The environment instance to reset.
        env_ids: Indices of the environments to reset.
        play: When True, overrides ``cuboid_distance_fraction`` to 1.0 to disable
            curriculum constraints during interactive play.
        cuboid_distance_fraction: Value in [0, 1] controlling the maximum initial
            cuboid-to-target distance as a fraction of the workspace diameter.
        **kwargs: Unused; accepted for compatibility with the event manager interface.

    Side Effects:
        - Snapshots ``env._target_satisfied_mask`` into ``env._final_target_reached_fraction``
          for the reset environments before clearing the mask.
        - Clears ``env._target_satisfied_mask`` to False for all reset environments.
        - Writes sampled target poses into ``env.target_poses`` for the reset environments.
        - Writes mocap poses for all push-target markers to the simulation via
          ``write_mocap_pose_to_sim``.
        - Writes joint positions and velocities for all cuboids to the simulation via
          ``write_joint_state_to_sim``.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if play:
        cuboid_distance_fraction = 1.0

    # At fraction=1 the threshold equals the workspace diameter, so constraint (a)
    # never rejects and sampling is purely uniform in the workspace.
    max_push_distance = cuboid_distance_fraction * 2.0 * push_constants.TARGET_SPAWN_RADIUS

    _init_push_state(env)
    env._final_target_reached_fraction[env_ids] = env._target_satisfied_mask[env_ids].float().mean(dim=1)
    env._final_episode_length[env_ids] = env.episode_length_buf[env_ids].float()
    env._target_satisfied_mask[env_ids] = False
    num = len(env_ids)

    # --- Sample target poses: uniform in workspace, pairwise separation enforced ---
    tyaw_low, tyaw_high = push_constants.TARGET_YAW_RANGE
    tyaw = tyaw_low + (tyaw_high - tyaw_low) * torch.rand(num, push_constants.NUM_CUBOIDS, device=env.device)
    tx   = torch.zeros(num, push_constants.NUM_CUBOIDS, device=env.device)
    ty   = torch.zeros(num, push_constants.NUM_CUBOIDS, device=env.device)

    for i in range(push_constants.NUM_CUBOIDS):
        # Pre-seed with an unconditional in-workspace sample as fallback.
        r      = push_constants.TARGET_SPAWN_RADIUS * torch.sqrt(torch.rand(num, device=env.device))
        theta  = 2.0 * torch.pi * torch.rand(num, device=env.device)
        tx[:, i] = r * torch.cos(theta)
        ty[:, i] = r * torch.sin(theta)

        needs_resample = torch.ones(num, dtype=torch.bool, device=env.device)
        for _ in range(_MAX_SPAWN_TRIES):
            r      = push_constants.TARGET_SPAWN_RADIUS * torch.sqrt(torch.rand(num, device=env.device))
            theta  = 2.0 * torch.pi * torch.rand(num, device=env.device)
            cand_x = r * torch.cos(theta)
            cand_y = r * torch.sin(theta)
            if i > 0:
                cand_x_exp = cand_x.unsqueeze(1)
                cand_y_exp = cand_y.unsqueeze(1)
                dists = torch.sqrt((cand_x_exp - tx[:, :i]) ** 2 + (cand_y_exp - ty[:, :i]) ** 2)
                valid = (dists >= push_constants.TARGET_MIN_SEPARATION).all(dim=1)
            else:
                valid = torch.ones(num, dtype=torch.bool, device=env.device)
            accepted  = needs_resample & valid
            tx[:, i]  = torch.where(accepted, cand_x, tx[:, i])
            ty[:, i]  = torch.where(accepted, cand_y, ty[:, i])
            needs_resample = needs_resample & ~accepted

    tz = 0
    env.target_poses[env_ids, :, 0] = tx
    env.target_poses[env_ids, :, 1] = ty
    env.target_poses[env_ids, :, 2] = tz
    env.target_poses[env_ids, :, 3] = tyaw

    for t in range(push_constants.NUM_CUBOIDS):
        half_yaw = tyaw[:, t] * 0.5
        poses = torch.zeros((num, 7), device=env.device)
        poses[:, 0] = tx[:, t]
        poses[:, 1] = ty[:, t]
        poses[:, 2] = tz
        poses[:, 3] = torch.cos(half_yaw)   # qw
        poses[:, 6] = torch.sin(half_yaw)   # qz
        env.scene[f"push_target_{t}"].write_mocap_pose_to_sim(mocap_pose=poses, env_ids=env_ids)

    # --- Sample cuboids: uniform in workspace, cuboid-to-target and pairwise separation enforced ---
    cfgs = _get_cuboid_joint_cfgs(env)
    yaw_low, yaw_high = push_constants.CUBOID_YAW_SPAWN_RANGE
    yaw_c = yaw_low + (yaw_high - yaw_low) * torch.rand(num, push_constants.NUM_CUBOIDS, device=env.device)
    x_c   = torch.zeros(num, push_constants.NUM_CUBOIDS, device=env.device)
    y_c   = torch.zeros(num, push_constants.NUM_CUBOIDS, device=env.device)

    for i in range(push_constants.NUM_CUBOIDS):
        # Pre-seed with an unconditional in-workspace sample as fallback.
        r      = push_constants.CUBOID_SPAWN_RADIUS * torch.sqrt(torch.rand(num, device=env.device))
        theta  = 2.0 * torch.pi * torch.rand(num, device=env.device)
        x_c[:, i] = r * torch.cos(theta)
        y_c[:, i] = r * torch.sin(theta)

        needs_resample = torch.ones(num, dtype=torch.bool, device=env.device)
        for _ in range(_MAX_SPAWN_TRIES):
            r      = push_constants.CUBOID_SPAWN_RADIUS * torch.sqrt(torch.rand(num, device=env.device))
            theta  = 2.0 * torch.pi * torch.rand(num, device=env.device)
            cand_x = r * torch.cos(theta)
            cand_y = r * torch.sin(theta)
            dist_target = torch.sqrt((cand_x - tx[:, i]) ** 2 + (cand_y - ty[:, i]) ** 2)
            valid  = dist_target <= max_push_distance
            if i > 0:
                cand_x_exp = cand_x.unsqueeze(1)
                cand_y_exp = cand_y.unsqueeze(1)
                dists_c = torch.sqrt((cand_x_exp - x_c[:, :i]) ** 2 + (cand_y_exp - y_c[:, :i]) ** 2)
                valid &= (dists_c >= push_constants.CUBOID_MIN_SEPARATION).all(dim=1)
            accepted  = needs_resample & valid
            x_c[:, i] = torch.where(accepted, cand_x, x_c[:, i])
            y_c[:, i] = torch.where(accepted, cand_y, y_c[:, i])
            needs_resample = needs_resample & ~accepted

    for i, cfg in enumerate(cfgs):
        joint_pos = torch.stack([x_c[:, i], y_c[:, i], yaw_c[:, i]], dim=-1)
        joint_vel = torch.zeros_like(joint_pos)
        joint_ids = cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)
        env.scene[cfg.name].write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=joint_ids)
