import torch
from mjlab.envs import ManagerBasedRlEnv

import multi_robot_rl.configs.type_constants as type_constants
import multi_robot_rl.common.quat_helpers as quat_helpers
from multi_robot_rl.assets.robots.base import RobotConfig
from multi_robot_rl.common.mdp import get_ee_positions


def _init_type_state(env: ManagerBasedRlEnv) -> None:
    if not hasattr(env, "active_keys"):
        env.active_keys = torch.zeros(
            (env.num_envs, type_constants.NUM_ACTIVE_KEYS), dtype=torch.long, device=env.device
        )
        env.newly_pressed_count = torch.zeros(env.num_envs, device=env.device)
        env.wrong_key_pressed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._total_keys_pressed = torch.zeros(env.num_envs, device=env.device)
        env._final_throughput = torch.zeros(env.num_envs, device=env.device)


# =========================================================
# OBSERVATIONS
# =========================================================

def keyboard_state_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Key joint positions + active key 2D positions: (num_envs, TOTAL_KEYS + NUM_ACTIVE_KEYS*2)"""
    _init_type_state(env)
    qpos = env.scene["keyboard"].data.joint_pos[:, :type_constants.TOTAL_KEYS]

    cols = env.active_keys % type_constants.NUM_COLS   # (num_envs, NUM_ACTIVE_KEYS)
    rows = env.active_keys // type_constants.NUM_COLS
    x, y = type_constants.get_key_pos_2d(cols, rows)  # (num_envs, NUM_ACTIVE_KEYS) each
    active_key_pos = torch.stack([x, y], dim=-1).reshape(env.num_envs, -1).float()

    return torch.cat([qpos, active_key_pos], dim=-1)


# =========================================================
# TERMINATIONS
# =========================================================

def out_of_bounds(env: ManagerBasedRlEnv, robots: list[RobotConfig], **kwargs) -> torch.Tensor:
    """Terminate if any robot strays too far from the keyboard in XY or out of z-bounds."""
    ee_positions = get_ee_positions(env, robots)  # (num_envs, num_robots, 3)
    out_x = torch.abs(ee_positions[:, :, 0] - type_constants.CENTER_POS[0]) > (type_constants.KEYBOARD_SIZE[0] + type_constants.OUT_OF_BOUNDS_MARGIN)
    out_y = torch.abs(ee_positions[:, :, 1] - type_constants.CENTER_POS[1]) > (type_constants.KEYBOARD_SIZE[1] + type_constants.OUT_OF_BOUNDS_MARGIN)
    out_z = (ee_positions[:, :, 2] > type_constants.EE_Z_MAX) | (ee_positions[:, :, 2] < type_constants.EE_Z_MIN)
    return (out_x | out_y | out_z).any(dim=1)

# =========================================================
# REWARDS
# =========================================================

def key_pressed_reward(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Sparse +1.0 per active key correctly pressed this step."""
    _init_type_state(env)
    return env.newly_pressed_count

def wrong_key_penalty(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """1.0 if any non-active key is pressed this step."""
    _init_type_state(env)
    return env.wrong_key_pressed.float()


# =========================================================
# METRICS
# =========================================================

def throughput(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Total correctly pressed keys in the previous episode."""
    _init_type_state(env)
    return env._final_throughput


# =========================================================
# EVENTS & RESETS
# =========================================================

def _sample_unique_active_keys(n: int, device: torch.device) -> torch.Tensor:
    """Sample NUM_ACTIVE_KEYS unique key indices per env using topk of uniform noise."""
    noise = torch.rand(n, type_constants.TOTAL_KEYS, device=device)
    _, indices = torch.topk(noise, type_constants.NUM_ACTIVE_KEYS, dim=1)
    return indices  # (n, NUM_ACTIVE_KEYS)


def reset_keyboard_state(env: ManagerBasedRlEnv, env_ids, **kwargs) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    _init_type_state(env)

    env._final_throughput[env_ids] = env._total_keys_pressed[env_ids]
    env._total_keys_pressed[env_ids] = 0.0
    env.newly_pressed_count[env_ids] = 0.0
    env.wrong_key_pressed[env_ids] = False
    env.active_keys[env_ids] = _sample_unique_active_keys(len(env_ids), env.device)


def _detect_key_presses(env: ManagerBasedRlEnv, env_ids: torch.Tensor):
    """Returns (key_qpos, is_pressed) for the given env_ids."""
    key_qpos = env.scene["keyboard"].data.joint_pos[env_ids, :type_constants.TOTAL_KEYS]
    is_pressed = key_qpos < type_constants.KEY_PRESS_THRESHOLD
    return key_qpos, is_pressed


def _evaluate_key_presses(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    is_pressed: torch.Tensor,
) -> torch.Tensor:
    """
    Determines which active key slots are pressed and whether any wrong key is pressed.

    Side effects:
    - Updates env.wrong_key_pressed for env_ids.

    Returns:
        slot_pressed: (n, NUM_ACTIVE_KEYS) bool — which slots had their active key pressed.
    """
    n = len(env_ids)
    active_keys_local = env.active_keys[env_ids]  # (n, NUM_ACTIVE_KEYS)

    # Which active slots are pressed
    row_idx = torch.arange(n, device=env.device).unsqueeze(1).expand(n, type_constants.NUM_ACTIVE_KEYS)
    slot_pressed = is_pressed[row_idx, active_keys_local]  # (n, NUM_ACTIVE_KEYS)

    # Wrong key: any pressed key not in active set
    active_mask = torch.zeros(n, type_constants.TOTAL_KEYS, dtype=torch.bool, device=env.device)
    active_mask.scatter_(1, active_keys_local, True)
    env.wrong_key_pressed[env_ids] = (is_pressed & ~active_mask).any(dim=1)

    return slot_pressed


def _handle_successful_presses(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    slot_pressed: torch.Tensor,
) -> None:
    """
    For each pressed slot: increments the press counter and replaces the key with a
    new unique key (not already in the active set for that env).
    """
    if not slot_pressed.any():
        return

    counts = slot_pressed.float().sum(dim=1)
    env._total_keys_pressed[env_ids] += counts
    env.newly_pressed_count[env_ids] = counts

    current_active = env.active_keys[env_ids].clone()  # (n, NUM_ACTIVE_KEYS)

    for slot_i in range(type_constants.NUM_ACTIVE_KEYS):
        pressed_in_slot = slot_pressed[:, slot_i]
        if not pressed_in_slot.any():
            continue
        for local_idx in pressed_in_slot.nonzero(as_tuple=False).flatten():
            occupied = set(current_active[local_idx].tolist())
            occupied.discard(current_active[local_idx, slot_i].item())
            candidates = [k for k in range(type_constants.TOTAL_KEYS) if k not in occupied]
            new_key = candidates[torch.randint(len(candidates), (1,), device=env.device).item()]
            current_active[local_idx, slot_i] = new_key

    env.active_keys[env_ids] = current_active


def _update_marker_poses(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    key_qpos: torch.Tensor,
) -> None:
    """Moves each active_key_i marker to the current position of its assigned key."""
    n = len(env_ids)
    for slot_i in range(type_constants.NUM_ACTIVE_KEYS):
        key_idx = env.active_keys[env_ids, slot_i]  # (n,)
        z = type_constants.MARKER_Z + key_qpos[torch.arange(n, device=env.device), key_idx]
        col = key_idx % type_constants.NUM_COLS
        row = key_idx // type_constants.NUM_COLS
        x, y = type_constants.get_key_pos_2d(col, row)
        env.scene[f"active_key_{slot_i}"].write_mocap_pose_to_sim(
            mocap_pose=quat_helpers.position_to_pose(x, y, z),
            env_ids=env_ids,
        )


def update_keyboard_state(env: ManagerBasedRlEnv, env_ids, **kwargs) -> None:
    """Step event: detect presses, update state, replace pressed keys, move markers."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    _init_type_state(env)

    env.newly_pressed_count[env_ids] = 0.0

    key_qpos, is_pressed = _detect_key_presses(env, env_ids)
    slot_pressed = _evaluate_key_presses(env, env_ids, is_pressed)
    _handle_successful_presses(env, env_ids, slot_pressed)
    _update_marker_poses(env, env_ids, key_qpos)
