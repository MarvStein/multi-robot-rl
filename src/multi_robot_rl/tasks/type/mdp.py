import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from multi_robot_rl.assets.robots.base import RobotConfig
import multi_robot_rl.configs.type_constants as type_constants
import multi_robot_rl.common.quat_helpers as quat_helpers

# =========================================================
# OBSERVATIONS
# =========================================================

def keyboard_state_obs(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Observation grouping key states and targets."""
    qpos_n = env.scene["keyboard"].data.joint_pos[:, :type_constants.TOTAL_KEYS]
    if not hasattr(env, "active_key"):
        active_targets = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.long)
        next_targets = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.long)
    else:
        active_targets = env.active_key.unsqueeze(-1)
        next_targets = env.next_key.unsqueeze(-1)

    terms = [
        qpos_n,
        active_targets,
        next_targets,
    ]
        
    return torch.cat(terms, dim=-1)

# =========================================================
# TERMINATIONS
# =========================================================

def _get_ee_configs(env: ManagerBasedRlEnv, robots: list[RobotConfig]) -> list[SceneEntityCfg]:
    """Helper to lazily create and resolve SceneEntityCfg for all robot end-effectors."""
    if not hasattr(env, "_ee_site_cfgs"):
        cfgs = [SceneEntityCfg(r.name, site_names=(r.end_effector_site,)) for r in robots]
        for cfg in cfgs:
            cfg.resolve(env.scene)
        env._ee_site_cfgs = list(cfgs)
    return env._ee_site_cfgs

def _get_joint_configs(env: ManagerBasedRlEnv, robots: list[RobotConfig]) -> list[SceneEntityCfg]:
    """Helper to lazily create and resolve SceneEntityCfg for all actuated joints."""
    if not hasattr(env, "_joint_cfgs"):
        cfgs = [SceneEntityCfg(r.name, joint_names=tuple(r.joint_names)) for r in robots]
        for cfg in cfgs:
            cfg.resolve(env.scene)
        env._joint_cfgs = list(cfgs)
    return env._joint_cfgs

def out_of_bounds(env: ManagerBasedRlEnv, robots: list[RobotConfig], **kwargs):
    """Terminate if any robot strays too far from the keyboard in XY or out of z-bounds."""
    ee_cfgs = _get_ee_configs(env, robots)
    out_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in ee_cfgs:
        pos = env.scene[cfg.name].data.site_pos_w[:, cfg.site_ids, :].squeeze(1)
        out_x = torch.abs(pos[:, 0] - type_constants.CENTER_POS[0]) > (type_constants.KEYBOARD_SIZE[0] + type_constants.OUT_OF_BOUNDS_MARGIN)
        out_y = torch.abs(pos[:, 1] - type_constants.CENTER_POS[1]) > (type_constants.KEYBOARD_SIZE[1] + type_constants.OUT_OF_BOUNDS_MARGIN)
        out_z = (pos[:, 2] > type_constants.EE_Z_MAX) | (pos[:, 2] < type_constants.EE_Z_MIN)
        out_mask = out_mask | out_x | out_y | out_z
    return out_mask

# =========================================================
# REWARDS
# =========================================================

def z_depression_reward(env: ManagerBasedRlEnv, **kwargs):
    """Reward when the correct key is actively pressed."""
    if hasattr(env, "key_pressed"):
        return env.key_pressed.float() * 1.0
    return torch.zeros(env.num_envs, device=env.device)

def wrong_key_penalty(env: ManagerBasedRlEnv, **kwargs):
    """Penalty for pressing the wrong key."""
    if hasattr(env, "wrong_key_pressed"):
        return env.wrong_key_pressed.float() * 1.0
    return torch.zeros(env.num_envs, device=env.device)

# =========================================================
# METRICS
# =========================================================

def key_press_fraction(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    """Returns 1.0 if a correct key was pressed this step."""
    key_pressed = getattr(env, "key_pressed", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float()
    return key_pressed

# =========================================================
# EVENTS & RESETS
# =========================================================

def _init_keyboard_state(env, num_robots: int):
    """
    Initializes the the following tensors:
    - env.agent_freeze_ticks: (num_envs, num_robots) tensor counting down freeze duration for each agent
    - env.active_key: (num_envs,) tensor of currently active key indices
    - env.next_key: (num_envs,) tensor of next key indices
    - env.key_pressed: (num_envs,) boolean tensor indicating if the correct key is currently pressed
    - env.wrong_key_pressed: (num_envs,) boolean tensor indicating if any wrong key is currently pressed

    Args:
        env: The environment.
        num_robots: The number of robots in the environment, used to size the agent_freeze_ticks tensor.
    """
    if not hasattr(env, "agent_freeze_ticks"):
        env.agent_freeze_ticks = torch.zeros((env.num_envs, num_robots), device=env.device, dtype=torch.long)
    if not hasattr(env, "active_key"):
        env.active_key = torch.randint(0, type_constants.TOTAL_KEYS, (env.num_envs,), device=env.device)
        env.next_key = torch.randint(0, type_constants.TOTAL_KEYS, (env.num_envs,), device=env.device)
        env.key_pressed = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        env.wrong_key_pressed = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

def reset_keyboard_state(env: ManagerBasedRlEnv, env_ids, robots: list[RobotConfig], **kwargs):
    """
    Event to completely reset the keyboard state for the given env_ids.
    This includes:
    - Randomly sampling new active and next keys
    - Resetting agent freeze ticks to 0
    - Clearing key_pressed and wrong_key_pressed buffers

    Args:
        env: The environment
        env_ids: handled by mjlab
        robots: A tuple of robot configs.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
        
    num_robots = len(robots)
    _init_keyboard_state(env, num_robots)
    
    env.active_key[env_ids] = torch.randint(0, type_constants.TOTAL_KEYS, (len(env_ids),), device=env.device)
    env.next_key[env_ids] = torch.randint(0, type_constants.TOTAL_KEYS, (len(env_ids),), device=env.device)
    env.agent_freeze_ticks[env_ids] = 0
    env.key_pressed[env_ids] = False
    env.wrong_key_pressed[env_ids] = False

def _detect_key_presses(env: ManagerBasedRlEnv, env_ids):
    """
    Detects which keys are currently pressed beyond the threshold.
    Args:
        env: The environment
        env_ids: handled by mjlab

    Returns:
        (key_qpos, is_pressed): A tuple where key_qpos is the joint positions of the keys for the given env_ids, and is_pressed is a boolean tensor indicating which keys are pressed.
    """
    key_qpos = env.scene["keyboard"].data.joint_pos[env_ids]
    is_pressed = key_qpos < type_constants.KEY_PRESS_THRESHOLD
    return key_qpos, is_pressed

def _evaluate_key_presses(env: ManagerBasedRlEnv, env_ids: torch.Tensor, is_pressed: torch.Tensor) -> torch.Tensor:
    """
    Evaluates which keys were pressed correctly and incorrectly.
    
    Side effects:
    - Updates env.key_pressed to indicate which envs had the correct key pressed.
    - Updates env.wrong_key_pressed to indicate which envs had any wrong key pressed.

    Args:
        env: The environment containing the keyboard state.
        env_ids: The indices of the environments to evaluate.
        is_pressed: A boolean tensor indicating which keys are currently pressed for the given env_ids

    Returns:
        env_success: A boolean tensor indicating which environments had the correct key pressed.
    """
    active_keys = env.active_key[env_ids]
    
    # Check if the active key is pressed
    env_success = is_pressed[torch.arange(len(env_ids)), active_keys]
    env.key_pressed[env_ids] = env_success
    
    # Check if any wrong key is pressed
    active_mask = torch.zeros_like(is_pressed)
    active_mask[torch.arange(len(env_ids)), active_keys] = True
    wrong_pressed = is_pressed & (~active_mask)
    env.wrong_key_pressed[env_ids] = wrong_pressed.any(dim=1)
    
    return env_success

def _handle_successful_presses(env: ManagerBasedRlEnv, success_env_ids: torch.Tensor, robots: list[RobotConfig]):
    """
    Handles the logic when a correct key is actively pressed.
    This includes:
    - Freezing the nearest agent to the active key for a configured number of steps
    - Advancing the active key to the next key, and sampling a new next key

    Args:
        env: The environment
        success_env_ids: The indices of the environments where the correct key was pressed.
        robots: A tuple of robot configs.
    """
    if len(success_env_ids) == 0:
        return

    # Get the 2D position of the active key for the successful environments
    col_active = env.active_key[success_env_ids] % type_constants.NUM_COLS
    row_active = env.active_key[success_env_ids] // type_constants.NUM_COLS
    x_active, y_active = type_constants.get_key_pos_2d(col_active, row_active)
    active_key_pos = torch.stack([x_active, y_active], dim=-1)

    # Collect all generic robot end-effector positions
    ee_cfgs = _get_ee_configs(env, robots)
    ee_pos_list = []
    for cfg in ee_cfgs:
        # mjlab standard: Extract all environment EEs securely first, squeeze, AND THEN index env IDs:
        # Avoids advanced index broadcasting clashes between success_env_ids tensor and site_ids list.
        all_ee_pos = env.scene[cfg.name].data.site_pos_w[:, cfg.site_ids, :2].squeeze(1)
        tmp_pos = all_ee_pos[success_env_ids]
        ee_pos_list.append(tmp_pos)
    
    ee_pos_stack = torch.stack(ee_pos_list, dim=1)
    
    # Find the nearest end-effector
    dists = torch.norm(ee_pos_stack - active_key_pos.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(dists, dim=1)
    
    # Freeze the nearest robot
    env.agent_freeze_ticks[success_env_ids, closest_idx] = type_constants.FREEZE_STEPS

    # Advance the targets
    new_keys = torch.randint(0, type_constants.TOTAL_KEYS, (len(success_env_ids),), device=env.device)
    env.active_key[success_env_ids] = env.next_key[success_env_ids]
    env.next_key[success_env_ids] = new_keys

def _apply_agent_freezing(env: ManagerBasedRlEnv, robots: list[RobotConfig]):
    """Decrements freeze ticks and zeroes out velocities of actuated joints for frozen agents."""
    frozen_mask = env.agent_freeze_ticks > 0
    env.agent_freeze_ticks[frozen_mask] -= 1
    
    joint_cfgs = _get_joint_configs(env, robots)
    for idx, cfg in enumerate(joint_cfgs):
        if hasattr(env.scene[cfg.name].data, "joint_vel"):
            agent_frozen_mask = env.agent_freeze_ticks[:, idx] > 0
            if agent_frozen_mask.any():
                env.scene[cfg.name].data.joint_vel[agent_frozen_mask, cfg.joint_ids] = 0.0

def _update_marker_poses(env: ManagerBasedRlEnv, env_ids: torch.Tensor, key_qpos_subset: torch.Tensor):
    """Updates the graphical markers indicating the active and next keys."""
    # Active key marker
    active_key_qpos = key_qpos_subset[torch.arange(len(env_ids)), env.active_key[env_ids]]
    z_active = type_constants.MARKER_Z + active_key_qpos
    col_active = env.active_key[env_ids] % type_constants.NUM_COLS
    row_active = env.active_key[env_ids] // type_constants.NUM_COLS
    x_active, y_active = type_constants.get_key_pos_2d(col_active, row_active)
    
    env.scene["active_key"].write_mocap_pose_to_sim(
        mocap_pose=quat_helpers.position_to_pose(x_active, y_active, z_active), 
        env_ids=env_ids
    )

    # Next key marker
    next_key_qpos = key_qpos_subset[torch.arange(len(env_ids)), env.next_key[env_ids]]
    z_next = type_constants.MARKER_Z + next_key_qpos
    col_next = env.next_key[env_ids] % type_constants.NUM_COLS
    row_next = env.next_key[env_ids] // type_constants.NUM_COLS
    x_next, y_next = type_constants.get_key_pos_2d(col_next, row_next)

    env.scene["next_key"].write_mocap_pose_to_sim(
        mocap_pose=quat_helpers.position_to_pose(x_next, y_next, z_next), 
        env_ids=env_ids
    )

def update_keyboard_state(env: ManagerBasedRlEnv, env_ids, robots: list[RobotConfig], **kwargs):
    """Step event tracking key touches and advancing targets."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
        
    num_robots = len(robots)
    _init_keyboard_state(env, num_robots) # num_robots is needed to initialize agent_freeze_ticks
        
    key_qpos_subset, is_pressed = _detect_key_presses(env, env_ids)

    env_success = _evaluate_key_presses(env, env_ids, is_pressed)
    
    success_indices = env_success.nonzero(as_tuple=False).flatten()
    if len(success_indices) > 0:
        success_env_ids = env_ids[success_indices]
        _handle_successful_presses(env, success_env_ids, robots)
        
    _apply_agent_freezing(env, robots)
    _update_marker_poses(env, env_ids, key_qpos_subset)
    