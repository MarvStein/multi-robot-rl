"""Event and reset functions."""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _build_pose(pos_2d: torch.Tensor, z: float) -> torch.Tensor:
    """Create mocap pose tensor [x, y, z, qw, qx, qy, qz] from XY positions."""
    z_col = torch.full((pos_2d.shape[0], 1), z, device=pos_2d.device)
    pos_3d = torch.cat([pos_2d, z_col], dim=-1)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=pos_2d.device).repeat(pos_2d.shape[0], 1)
    return torch.cat([pos_3d, quat], dim=-1)


def _sample_positions_with_rejection(
    env,
    env_ids: torch.Tensor,
    pos_range: tuple,
    masspoint_names: tuple[str, ...],
    min_dist_to_masspoints: float,
    max_tries: int,
) -> torch.Tensor:
    """Sample 2D positions and keep those sufficiently far from masspoints when possible."""
    n = env_ids.shape[0]
    x_min, x_max = pos_range[0]
    y_min, y_max = pos_range[1]

    best = torch.empty(n, 2, device=env.device)
    accepted = torch.zeros(n, dtype=torch.bool, device=env.device)

    if min_dist_to_masspoints <= 0.0 or len(masspoint_names) == 0:
        best[:, 0].uniform_(x_min, x_max)
        best[:, 1].uniform_(y_min, y_max)
        return best

    masspoint_pos = torch.stack(
        [env.scene[name].data.joint_pos[env_ids, :2] for name in masspoint_names], dim=1
    )

    for _ in range(max_tries):
        unresolved = ~accepted
        if not unresolved.any():
            break

        sample = torch.empty(n, 2, device=env.device)
        sample[:, 0].uniform_(x_min, x_max)
        sample[:, 1].uniform_(y_min, y_max)

        sample_to_mp = torch.norm(sample.unsqueeze(1) - masspoint_pos, dim=-1)
        valid = torch.all(sample_to_mp >= min_dist_to_masspoints, dim=1)

        newly_accepted = unresolved & valid
        best[newly_accepted] = sample[newly_accepted]
        accepted |= newly_accepted

    fallback = ~accepted
    if fallback.any():
        best[fallback, 0].uniform_(x_min, x_max)
        best[fallback, 1].uniform_(y_min, y_max)

    return best


def _ensure_multi_goal_buffers(env, num_goals: int):
    """Allocate per-env goal activity, cooldown, and reached-this-step buffers."""
    if not hasattr(env, "_multi_goal_active") or env._multi_goal_active.shape[1] != num_goals:
        env._multi_goal_active = torch.ones(env.num_envs, num_goals, device=env.device, dtype=torch.bool)
        env._multi_goal_cooldown = torch.zeros(env.num_envs, num_goals, device=env.device, dtype=torch.long)
        env._multi_goal_reached_this_step = torch.zeros(
            env.num_envs, num_goals, device=env.device, dtype=torch.bool
        )

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


def reset_multi_goals(
    env,
    env_ids: torch.Tensor,
    goal_names: tuple[str, ...],
    masspoint_names: tuple[str, ...],
    pos_range: tuple,
    min_dist_to_masspoints: float,
    rejection_max_tries: int,
):
    """Reset all goals and lifecycle state for selected environments."""
    _ensure_multi_goal_buffers(env, len(goal_names))
    env._multi_goal_active[env_ids] = True
    env._multi_goal_cooldown[env_ids] = 0
    env._multi_goal_reached_this_step[env_ids] = False

    z = pos_range[2][0]
    for goal_name in goal_names:
        goal_xy = _sample_positions_with_rejection(
            env=env,
            env_ids=env_ids,
            pos_range=pos_range,
            masspoint_names=masspoint_names,
            min_dist_to_masspoints=min_dist_to_masspoints,
            max_tries=rejection_max_tries,
        )
        env.scene[goal_name].write_mocap_pose_to_sim(mocap_pose=_build_pose(goal_xy, z), env_ids=env_ids)


def update_multi_goals_lifecycle(
    env,
    goal_names: tuple[str, ...],
    masspoint_names: tuple[str, ...],
    reach_threshold: float,
    respawn_delay_steps: int,
    pos_range: tuple,
    min_dist_to_masspoints: float,
    rejection_max_tries: int,
):
    """Detect reached goals, keep them inactive for delay, then respawn and reactivate."""
    _ensure_multi_goal_buffers(env, len(goal_names))
    env._multi_goal_reached_this_step[:] = False

    active = env._multi_goal_active
    cooldown = env._multi_goal_cooldown

    # Decrement cooldown for inactive goals.
    inactive_with_timer = (~active) & (cooldown > 0)
    cooldown[inactive_with_timer] -= 1

    # Respawn goals whose cooldown expired.
    ready_to_respawn = (~active) & (cooldown <= 0)
    if ready_to_respawn.any():
        z = pos_range[2][0]
        for goal_idx, goal_name in enumerate(goal_names):
            env_ids = torch.where(ready_to_respawn[:, goal_idx])[0]
            if env_ids.numel() == 0:
                continue
            goal_xy = _sample_positions_with_rejection(
                env=env,
                env_ids=env_ids,
                pos_range=pos_range,
                masspoint_names=masspoint_names,
                min_dist_to_masspoints=min_dist_to_masspoints,
                max_tries=rejection_max_tries,
            )
            env.scene[goal_name].write_mocap_pose_to_sim(mocap_pose=_build_pose(goal_xy, z), env_ids=env_ids)
            active[env_ids, goal_idx] = True
            cooldown[env_ids, goal_idx] = 0

    # Detect reached goals (both newly reached active ones and continuously touched inactive ones).
    mp_pos = torch.stack([env.scene[name].data.joint_pos[:, :2] for name in masspoint_names], dim=1)
    goal_pos = torch.stack([env.scene[name].data.root_link_pos_w[:, :2] for name in goal_names], dim=1)
    dist = torch.cdist(mp_pos, goal_pos, p=2)
    reached_any = torch.any(dist <= reach_threshold, dim=1)
    newly_reached = active & reached_any

    # Record any touched goal so that masspoints get continuous rewards while staying near them.
    env._multi_goal_reached_this_step[:] = reached_any

    if newly_reached.any():
        active[newly_reached] = False
        cooldown[newly_reached] = max(respawn_delay_steps, 0)
