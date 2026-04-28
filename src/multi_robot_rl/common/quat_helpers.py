import torch

def position_to_pose(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Returns a pose tensor [x, y, z, qw, qx, qy, qz] with unit quaternion from positions."""
    poses = torch.zeros((x.shape[0], 7), device=x.device)
    poses[:, 0] = x
    poses[:, 1] = y
    poses[:, 2] = z
    poses[:, 3] = 1.0 # w=1 quaternion
    return poses