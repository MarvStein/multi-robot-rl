"""Reward, observation, termination, metric, and reset functions for the multi-robot keyboard typing task."""
import torch
from typing import Any
from mjlab.envs import ManagerBasedRlEnv

import multi_robot_rl.configs.type_constants as type_constants
import multi_robot_rl.common.quat_helpers as quat_helpers
from multi_robot_rl.assets.robots.base import RobotConfig
from multi_robot_rl.common.mdp import get_ee_positions


def _init_type_state(env: ManagerBasedRlEnv) -> None:
    """Initialize all typing-task state attributes on the environment if they are not yet present.

    Args:
        env: The managed RL environment to initialize.

    Side Effects:
        - Sets env.active_keys, env.newly_pressed_count, env.newly_wrong_count,
          env._total_keys_pressed, env._final_throughput,
          env._total_wrong_keys_pressed, env._final_wrong_keys_per_episode, and
          env._prev_is_pressed as zero-valued tensors on the environment.
    """
    if not hasattr(env, "active_keys"):
        env.active_keys = torch.zeros(
            (env.num_envs, type_constants.NUM_ACTIVE_KEYS), dtype=torch.long, device=env.device
        )
        env.newly_pressed_count = torch.zeros(env.num_envs, device=env.device)
        env.newly_wrong_count = torch.zeros(env.num_envs, device=env.device)
        env._total_keys_pressed = torch.zeros(env.num_envs, device=env.device)
        env._final_throughput = torch.zeros(env.num_envs, device=env.device)
        env._total_wrong_keys_pressed = torch.zeros(env.num_envs, device=env.device)
        env._final_wrong_keys_per_episode = torch.zeros(env.num_envs, device=env.device)
        env._prev_is_pressed = torch.zeros(
            (env.num_envs, type_constants.TOTAL_KEYS), dtype=torch.bool, device=env.device
        )


# =========================================================
# OBSERVATIONS
# =========================================================

def keyboard_state_obs(env: ManagerBasedRlEnv, **kwargs: Any) -> torch.Tensor:
    """Return the keyboard joint positions concatenated with the 2D grid positions of the active keys.

    Args:
        env: The managed RL environment.
        **kwargs: Unused keyword arguments passed by the observation manager.

    Returns:
        Tensor of shape (num_envs, TOTAL_KEYS + NUM_ACTIVE_KEYS * 2) containing
        the TOTAL_KEYS key joint positions followed by the flattened 2D (x, y)
        positions of the NUM_ACTIVE_KEYS currently active keys.
    """
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

def out_of_bounds(env: ManagerBasedRlEnv, robots: list[RobotConfig], **kwargs: Any) -> torch.Tensor:
    """Return a per-environment bool indicating whether any robot end-effector is out of bounds.

    An environment is flagged when any robot's end-effector exceeds the keyboard
    footprint (plus OUT_OF_BOUNDS_MARGIN) in X or Y, or falls outside [EE_Z_MIN, EE_Z_MAX].

    Args:
        env: The managed RL environment.
        robots: List of RobotConfig objects whose end-effector positions are checked.
        **kwargs: Unused keyword arguments passed by the termination manager.

    Returns:
        Boolean tensor of shape (num_envs,); True where at least one robot is out of bounds.
    """
    ee_positions = get_ee_positions(env, robots)  # (num_envs, num_robots, 3)
    out_x = torch.abs(ee_positions[:, :, 0] - type_constants.CENTER_POS[0]) > (type_constants.KEYBOARD_SIZE[0] + type_constants.OUT_OF_BOUNDS_MARGIN)
    out_y = torch.abs(ee_positions[:, :, 1] - type_constants.CENTER_POS[1]) > (type_constants.KEYBOARD_SIZE[1] + type_constants.OUT_OF_BOUNDS_MARGIN)
    out_z = (ee_positions[:, :, 2] > type_constants.EE_Z_MAX) | (ee_positions[:, :, 2] < type_constants.EE_Z_MIN)
    return (out_x | out_y | out_z).any(dim=1)

# =========================================================
# REWARDS
# =========================================================

def key_pressed_reward(env: ManagerBasedRlEnv, **kwargs: Any) -> torch.Tensor:
    """Return the count of active keys correctly pressed this step as a sparse reward signal.

    The returned value is env.newly_pressed_count, which is populated each step
    by update_keyboard_state.

    Args:
        env: The managed RL environment.
        **kwargs: Unused keyword arguments passed by the reward manager.

    Returns:
        Float tensor of shape (num_envs,) with the number of correct key presses this step.
    """
    _init_type_state(env)
    return env.newly_pressed_count

def wrong_key_penalty(env: ManagerBasedRlEnv, **kwargs: Any) -> torch.Tensor:
    """Return the rising-edge count of non-active keys pressed this step as a penalty signal.

    Only new presses (rising-edge) are counted so that holding a wrong key down
    does not accumulate further penalty. The value matches the wrong_keys_per_episode metric.

    Args:
        env: The managed RL environment.
        **kwargs: Unused keyword arguments passed by the reward manager.

    Returns:
        Float tensor of shape (num_envs,) with the number of newly pressed wrong keys this step.
    """
    _init_type_state(env)
    return env.newly_wrong_count


# =========================================================
# METRICS
# =========================================================

def throughput(env: ManagerBasedRlEnv, **kwargs: Any) -> torch.Tensor:
    """Return the total number of correctly pressed keys recorded in the previous episode.

    The value is snapped at episode end by reset_keyboard_state and stored in
    env._final_throughput so that it is stable throughout the current episode.

    Args:
        env: The managed RL environment.
        **kwargs: Unused keyword arguments passed by the metrics manager.

    Returns:
        Float tensor of shape (num_envs,) with the per-environment throughput
        from the most recently completed episode.
    """
    _init_type_state(env)
    return env._final_throughput

def wrong_keys_per_episode(env: ManagerBasedRlEnv, **kwargs: Any) -> torch.Tensor:
    """Return the total number of wrong key presses recorded in the previous episode.

    The value is snapped at episode end by reset_keyboard_state and stored in
    env._final_wrong_keys_per_episode so that it is stable throughout the current episode.

    Args:
        env: The managed RL environment.
        **kwargs: Unused keyword arguments passed by the metrics manager.

    Returns:
        Float tensor of shape (num_envs,) with the per-environment wrong-key count
        from the most recently completed episode.
    """
    _init_type_state(env)
    return env._final_wrong_keys_per_episode


# =========================================================
# EVENTS & RESETS
# =========================================================

def _sample_unique_active_keys(n: int, device: torch.device) -> torch.Tensor:
    """Sample NUM_ACTIVE_KEYS unique key indices for each of n environments.

    Uniqueness per environment is guaranteed by taking the top-k indices of
    per-row uniform noise, so no key index appears twice in the same row.

    Args:
        n: Number of environments to sample for.
        device: Torch device on which to allocate the result tensor.

    Returns:
        Long tensor of shape (n, NUM_ACTIVE_KEYS) containing unique key indices
        drawn from [0, TOTAL_KEYS) for each environment.
    """
    noise = torch.rand(n, type_constants.TOTAL_KEYS, device=device)
    _, indices = torch.topk(noise, type_constants.NUM_ACTIVE_KEYS, dim=1)
    return indices  # (n, NUM_ACTIVE_KEYS)


def reset_keyboard_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **kwargs: Any) -> None:
    """Snap final metrics, zero per-episode counters, and sample new active keys for reset environments.

    Args:
        env: The managed RL environment.
        env_ids: Indices of environments being reset; defaults to all environments if None.
        **kwargs: Unused keyword arguments passed by the event manager.

    Side Effects:
        - Copies env._total_keys_pressed into env._final_throughput for env_ids, then zeros it.
        - Copies env._total_wrong_keys_pressed into env._final_wrong_keys_per_episode for env_ids, then zeros it.
        - Zeros env.newly_pressed_count and env.newly_wrong_count for env_ids.
        - Resets env._prev_is_pressed to False for env_ids.
        - Samples new unique active keys into env.active_keys for env_ids.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    _init_type_state(env)

    env._final_throughput[env_ids] = env._total_keys_pressed[env_ids]
    env._total_keys_pressed[env_ids] = 0.0
    env._final_wrong_keys_per_episode[env_ids] = env._total_wrong_keys_pressed[env_ids]
    env._total_wrong_keys_pressed[env_ids] = 0.0
    env.newly_pressed_count[env_ids] = 0.0
    env.newly_wrong_count[env_ids] = 0.0
    env._prev_is_pressed[env_ids] = False
    env.active_keys[env_ids] = _sample_unique_active_keys(len(env_ids), env.device)


def _detect_key_presses(env: ManagerBasedRlEnv, env_ids: torch.Tensor):
    """Read keyboard joint positions and determine which keys are currently pressed.

    A key is considered pressed when its joint position falls below KEY_PRESS_THRESHOLD.

    Args:
        env: The managed RL environment containing the keyboard scene entity.
        env_ids: Indices of environments to query.

    Returns:
        key_qpos: Float tensor of shape (len(env_ids), TOTAL_KEYS) with joint positions.
        is_pressed: Bool tensor of shape (len(env_ids), TOTAL_KEYS); True where a key is pressed.
    """
    key_qpos = env.scene["keyboard"].data.joint_pos[env_ids, :type_constants.TOTAL_KEYS]
    is_pressed = key_qpos < type_constants.KEY_PRESS_THRESHOLD
    return key_qpos, is_pressed


def _evaluate_key_presses(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    is_pressed: torch.Tensor,
    prev_is_pressed: torch.Tensor,
) -> torch.Tensor:
    """Determine which active key slots are pressed and count newly pressed wrong keys.

    Args:
        env: The managed RL environment.
        env_ids: Indices of environments to evaluate.
        is_pressed: Bool tensor of shape (n, TOTAL_KEYS) indicating currently pressed keys.
        prev_is_pressed: Bool tensor of shape (n, TOTAL_KEYS) indicating keys pressed in the previous step.

    Returns:
        slot_pressed: Bool tensor of shape (n, NUM_ACTIVE_KEYS); True where the active key
        in that slot is currently pressed.

    Side Effects:
        - Sets env.newly_wrong_count[env_ids] to the rising-edge count of wrong key presses
          (holding a wrong key down does not accumulate further).
        - Increments env._total_wrong_keys_pressed[env_ids] by the same rising-edge count.
    """
    n = len(env_ids)
    active_keys_local = env.active_keys[env_ids]  # (n, NUM_ACTIVE_KEYS)

    # Which active slots are pressed
    row_idx = torch.arange(n, device=env.device).unsqueeze(1).expand(n, type_constants.NUM_ACTIVE_KEYS)
    slot_pressed = is_pressed[row_idx, active_keys_local]  # (n, NUM_ACTIVE_KEYS)

    # Wrong key: any pressed key not in active set
    active_mask = torch.zeros(n, type_constants.TOTAL_KEYS, dtype=torch.bool, device=env.device)
    active_mask.scatter_(1, active_keys_local, True)
    # Rising-edge wrong key count: used for both the penalty reward and the metric
    newly_wrong = (is_pressed & ~active_mask) & ~prev_is_pressed  # (n, TOTAL_KEYS)
    env.newly_wrong_count[env_ids] = newly_wrong.sum(dim=1).float()
    env._total_wrong_keys_pressed[env_ids] += env.newly_wrong_count[env_ids]

    return slot_pressed


def _handle_successful_presses(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    slot_pressed: torch.Tensor,
) -> None:
    """Increment press counters and replace each successfully pressed active key with a new unique key.

    For each pressed slot the pressed key is replaced by a randomly sampled key
    that is not already in the active set for that environment, preserving the
    uniqueness invariant across slots within the same environment.

    Args:
        env: The managed RL environment.
        env_ids: Indices of environments to update.
        slot_pressed: Bool tensor of shape (n, NUM_ACTIVE_KEYS) indicating which slots were pressed.

    Side Effects:
        - Increments env._total_keys_pressed[env_ids] by the per-environment press count.
        - Sets env.newly_pressed_count[env_ids] to the per-environment press count.
        - Updates env.active_keys[env_ids] so that pressed slots contain new unique key indices.
    """
    if not slot_pressed.any():
        return

    counts = slot_pressed.float().sum(dim=1)
    env._total_keys_pressed[env_ids] += counts
    env.newly_pressed_count[env_ids] = counts

    n = len(env_ids)
    current_active = env.active_keys[env_ids].clone()  # (n, NUM_ACTIVE_KEYS)

    # Process slots sequentially so that a replacement written to slot i is
    # already excluded from the candidate pool when we sample for slot i+1
    # (maintains the uniqueness invariant across slots within the same env).
    for slot_i in range(type_constants.NUM_ACTIVE_KEYS):
        pressed_in_slot = slot_pressed[:, slot_i]  # (n,) bool
        if not pressed_in_slot.any():
            continue

        # Occupation mask: keys already taken by other slots — (n, TOTAL_KEYS).
        occupied = torch.zeros(n, type_constants.TOTAL_KEYS, dtype=torch.bool, device=env.device)
        for other in range(type_constants.NUM_ACTIVE_KEYS):
            if other == slot_i:
                continue
            occupied.scatter_(1, current_active[:, other:other + 1], True)

        # Single batched sample for all n envs; masked positions can't win argmax.
        noise = torch.rand(n, type_constants.TOTAL_KEYS, device=env.device)
        noise.masked_fill_(occupied, -1.0)
        new_keys = torch.argmax(noise, dim=1)  # (n,)

        current_active[:, slot_i] = torch.where(pressed_in_slot, new_keys, current_active[:, slot_i])

    env.active_keys[env_ids] = current_active


def _update_marker_poses(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    key_qpos: torch.Tensor,
) -> None:
    """Move each active_key_i marker to the current 3-D position of its assigned key.

    Args:
        env: The managed RL environment containing the active_key_* scene entities.
        env_ids: Indices of environments to update.
        key_qpos: Float tensor of shape (len(env_ids), TOTAL_KEYS) with current key joint positions.

    Side Effects:
        - Writes mocap poses to the simulation for each active_key_i marker,
          positioning it at the grid (x, y) of its assigned key plus the key's current z offset.
    """
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


def update_keyboard_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **kwargs: Any) -> None:
    """Step event that advances the full keyboard state for the given environments.

    Args:
        env: The managed RL environment.
        env_ids: Indices of environments to update; defaults to all environments if None.
        **kwargs: Unused keyword arguments passed by the event manager.

    Side Effects:
        - Zeros env.newly_pressed_count and env.newly_wrong_count for env_ids.
        - Calls _detect_key_presses to read current key positions.
        - Updates env._prev_is_pressed with the current press state.
        - Calls _evaluate_key_presses to compute slot_pressed and update wrong-key counts.
        - Calls _handle_successful_presses to update total/newly pressed counts and active keys.
        - Calls _update_marker_poses to reposition active key markers in the simulation.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    _init_type_state(env)

    env.newly_pressed_count[env_ids] = 0.0
    env.newly_wrong_count[env_ids] = 0.0

    prev_is_pressed = env._prev_is_pressed[env_ids].clone()
    key_qpos, is_pressed = _detect_key_presses(env, env_ids)
    env._prev_is_pressed[env_ids] = is_pressed
    slot_pressed = _evaluate_key_presses(env, env_ids, is_pressed, prev_is_pressed)
    _handle_successful_presses(env, env_ids, slot_pressed)
    _update_marker_poses(env, env_ids, key_qpos)
