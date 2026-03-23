"""Termination functions."""
import torch

def time_out(env) -> torch.Tensor:
    """Terminate episode after a hard time limit."""
    return env.episode_length_buf >= env.max_episode_length
