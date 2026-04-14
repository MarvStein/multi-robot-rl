"""Termination functions."""
import torch
import multi_robot_rl.masspoints.keyboard_constants as kc

def time_out(env) -> torch.Tensor:
    """Terminate episode after a hard time limit."""
    return env.episode_length_buf >= env.max_episode_length

def out_of_bounds(env, masspoint_names: tuple[str, ...], **kwargs):
    # incorporates a margin around the board, and relative to the board's true center.
    out_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in masspoint_names:
        pos = env.scene[name].data.joint_pos[:, :2] # Assuming x, y are first two
        out_x = torch.abs(pos[:, 0] - kc.CENTER_POS[0]) > (kc.KEYBOARD_SIZE[0] + kc.OUT_OF_BOUNDS_MARGIN)
        out_y = torch.abs(pos[:, 1] - kc.CENTER_POS[1]) > (kc.KEYBOARD_SIZE[1] + kc.OUT_OF_BOUNDS_MARGIN)
        out_mask = out_mask | out_x | out_y
    return out_mask
