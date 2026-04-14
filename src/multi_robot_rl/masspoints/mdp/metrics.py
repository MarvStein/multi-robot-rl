import torch

def key_press_fraction(env, **kwargs) -> torch.Tensor:
    """Returns 1.0 if a key is pressed this step.
    
    The MetricsManager automatically averages this over the actual episode length
    (including early terminations), resulting in the fraction of steps where a key was successfully pressed.
    """
    # The active key press status
    key_pressed = getattr(env, "key_pressed", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float()
    return key_pressed
