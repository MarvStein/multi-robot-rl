"""Event and reset functions."""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .. import keyboard_constants as kc


def _build_pose(pos_2d: torch.Tensor, z: float) -> torch.Tensor:
    """Create mocap pose tensor [x, y, z, qw, qx, qy, qz] from XY positions."""
    poses = torch.zeros((pos_2d.shape[0], 7), device=pos_2d.device)
    poses[:, :2] = pos_2d
    poses[:, 2] = z
    poses[:, 3] = 1.0 # w=1 quaternion
    return poses


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

def _init_keyboard_state(env, num_masspoints: int):
    if not hasattr(env, "agent_freeze_ticks"):
        env.agent_freeze_ticks = torch.zeros((env.num_envs, num_masspoints), device=env.device, dtype=torch.long)
    if not hasattr(env, "active_key"):
        env.active_key = torch.randint(0, kc.TOTAL_KEYS, (env.num_envs,), device=env.device)
        env.next_key = torch.randint(0, kc.TOTAL_KEYS, (env.num_envs,), device=env.device)
        env.key_pressed = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        env.wrong_key_pressed = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

def reset_keyboard_state(env, env_ids, masspoint_names: tuple[str, ...], **kwargs):
    """Event to completely reset the keyboard state for the given env_ids (e.g. episode reset)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
        
    num_masspoints = len(masspoint_names)
    _init_keyboard_state(env, num_masspoints)
    
    # Strictly randomize the keys for the resetting environments
    env.active_key[env_ids] = torch.randint(0, kc.TOTAL_KEYS, (len(env_ids),), device=env.device)
    env.next_key[env_ids] = torch.randint(0, kc.TOTAL_KEYS, (len(env_ids),), device=env.device)
    env.agent_freeze_ticks[env_ids] = 0
    env.key_pressed[env_ids] = False
    env.wrong_key_pressed[env_ids] = False
    
    # We don't need to manually update markers here because update_keyboard_state 
    # will run during the very next `step` event before rendering!

def _detect_key_presses(env, env_ids):
    """Detects which keys are currently pressed beyond the threshold."""
    key_qpos = env.scene["keyboard"].data.joint_pos[env_ids]
    is_pressed = key_qpos < kc.KEY_PRESS_THRESHOLD
    return key_qpos, is_pressed

def update_keyboard_state(env, env_ids, masspoint_names: tuple[str, ...], **kwargs):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
        
    num_masspoints = len(masspoint_names)
    _init_keyboard_state(env, num_masspoints)
        
    key_qpos_subset, is_pressed = _detect_key_presses(env, env_ids)

    # Determine the status of the active key to register a successful press
    active_keys = env.active_key[env_ids]
    env_success = is_pressed[torch.arange(len(env_ids)), active_keys]
    env.key_pressed[env_ids] = env_success
    
    # Determine if any wrong key is pressed
    active_mask = torch.zeros_like(is_pressed)
    active_mask[torch.arange(len(env_ids)), active_keys] = True
    wrong_pressed = is_pressed & (~active_mask)
    env.wrong_key_pressed[env_ids] = wrong_pressed.any(dim=1)
    
    # 1. EXPENSIVE LOGIC: Only sample new keys for environments that ACTUALLY succeeded
    success_indices = env_success.nonzero(as_tuple=False).flatten()
    if len(success_indices) > 0:
        success_env_ids = env_ids[success_indices]
        
        # Calculate global coordinates of active keys
        col_active = env.active_key[success_env_ids] % kc.NUM_COLS
        row_active = env.active_key[success_env_ids] // kc.NUM_COLS
        x_active, y_active = kc.get_key_pos_2d(col_active, row_active)
        active_key_pos = torch.stack([x_active, y_active], dim=-1)

        # Calculate distance from each masspoint to active key center
        mp_pos_list = [env.scene[name].data.joint_pos[success_env_ids, :2] for name in masspoint_names]
        mp_pos_stack = torch.stack(mp_pos_list, dim=1) # [num_success, num_masspoints, 2]
        
        dists = torch.norm(mp_pos_stack - active_key_pos.unsqueeze(1), dim=-1) # [num_success, num_masspoints]
        closest_idx = torch.argmin(dists, dim=1)
        
        # Assign freeze ticks
        env.agent_freeze_ticks[success_env_ids, closest_idx] = kc.FREEZE_STEPS

        # sample new targets only for successes
        new_keys = torch.randint(0, kc.TOTAL_KEYS, (len(success_env_ids),), device=env.device)
        
        # advance targets
        env.active_key[success_env_ids] = env.next_key[success_env_ids]
        env.next_key[success_env_ids] = new_keys
        
    # Decrement freeze ticks
    frozen_mask = env.agent_freeze_ticks > 0
    env.agent_freeze_ticks[frozen_mask] -= 1
    
    # Freeze velocities for those frozen agents
    for idx, name in enumerate(masspoint_names):
        if hasattr(env.scene[name].data, "joint_vel"):
            agent_frozen_mask = env.agent_freeze_ticks[:, idx] > 0
            if agent_frozen_mask.any():
                env.scene[name].data.joint_vel[agent_frozen_mask] = 0.0
    # 2. VISUAL LOGIC: Make the markers physically track the keys downward every step
    # We add the negative qpos (depression distance) to the static MARKER_Z altitude
    active_key_qpos = key_qpos_subset[torch.arange(len(env_ids)), env.active_key[env_ids]]
    next_key_qpos = key_qpos_subset[torch.arange(len(env_ids)), env.next_key[env_ids]]
    
    z_active = kc.MARKER_Z + active_key_qpos
    z_next = kc.MARKER_Z + next_key_qpos
    
    # Calculate global coordinates using the constants, accounting for board position
    col_active = env.active_key[env_ids] % kc.NUM_COLS
    row_active = env.active_key[env_ids] // kc.NUM_COLS
    x_active, y_active = kc.get_key_pos_2d(col_active, row_active)
    env.scene["active_key_marker"].write_mocap_pose_to_sim(
        mocap_pose=_build_pose(torch.stack([x_active, y_active], dim=-1), z_active), env_ids=env_ids
    )

    col_next = env.next_key[env_ids] % kc.NUM_COLS
    row_next = env.next_key[env_ids] // kc.NUM_COLS
    x_next, y_next = kc.get_key_pos_2d(col_next, row_next)
    env.scene["next_key_marker"].write_mocap_pose_to_sim(
        mocap_pose=_build_pose(torch.stack([x_next, y_next], dim=-1), z_next), env_ids=env_ids
    )

def reset_masspoint_3d(
    env,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    z_height: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    default_joint_vel = asset.data.default_joint_vel

    joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    
    # Randomize x and y
    joint_pos[:, 0] += torch.empty(len(env_ids), device=env.device).uniform_(*position_range)
    joint_pos[:, 1] += torch.empty(len(env_ids), device=env.device).uniform_(*position_range)
    # Set exact z height
    joint_pos[:, 2] = z_height

    joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel += torch.empty_like(joint_vel).uniform_(*velocity_range)

    joint_ids_tensor = asset_cfg.joint_ids
    if isinstance(joint_ids_tensor, (list, tuple)):
        joint_ids_tensor = torch.tensor(joint_ids_tensor, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids_tensor,
    )
