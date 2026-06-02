"""Quaternion and pose utility functions for multi-robot environments.

Provides helpers for converting raw position data into full pose representations
compatible with MuJoCo conventions (position + unit quaternion).
"""

import torch

def position_to_pose(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Build a pose tensor with an identity quaternion from per-axis position tensors.

    Args:
        x: X coordinates of shape ``(N,)``.
        y: Y coordinates of shape ``(N,)``.
        z: Z coordinates of shape ``(N,)``.

    Returns:
        Float tensor of shape ``(N, 7)`` where each row is
        ``[x, y, z, qw, qx, qy, qz]`` with the identity quaternion
        ``qw=1, qx=0, qy=0, qz=0``.
    """
    poses = torch.zeros((x.shape[0], 7), device=x.device)
    poses[:, 0] = x
    poses[:, 1] = y
    poses[:, 2] = z
    poses[:, 3] = 1.0 # w=1 quaternion
    return poses