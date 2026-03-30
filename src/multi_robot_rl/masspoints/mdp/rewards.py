"""Reward functions."""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .observations import distance_to_goal


def cooperative_goal_completion_reward(env) -> torch.Tensor:
    """Reward globally for touching goals (provides continuous rewards during cooldown until respawn)."""
    reached = getattr(env, "_multi_goal_reached_this_step", None)
    if reached is None:
        return torch.zeros(env.num_envs, device=env.device)
    return reached.float().sum(dim=1)


def nearest_goal_progress_reward(
    env,
    masspoint_names: tuple[str, ...],
    goal_names: tuple[str, ...],
) -> torch.Tensor:
    """Unassigned shaping term using distance to nearest active goal for each masspoint."""
    mp_pos = torch.stack([env.scene[name].data.joint_pos[:, :2] for name in masspoint_names], dim=1)
    goal_pos = torch.stack([env.scene[name].data.root_link_pos_w[:, :2] for name in goal_names], dim=1)
    active = getattr(env, "_multi_goal_active", None)
    if active is None:
        active = torch.ones(env.num_envs, len(goal_names), device=env.device, dtype=torch.bool)

    dists = torch.cdist(mp_pos, goal_pos, p=2)
    inactive_mask = ~active.unsqueeze(1)
    dists = torch.where(inactive_mask, torch.full_like(dists, 1.0e6), dists)

    nearest = torch.min(dists, dim=-1).values
    nearest = torch.where(nearest >= 1.0e5, torch.zeros_like(nearest), nearest)
    return torch.exp(-2.0 * nearest).mean(dim=1)


def proximity_penalty(env, masspoint_names: tuple[str, ...], min_distance: float) -> torch.Tensor:
    """Penalty when masspoints get closer than a configured distance."""
    if len(masspoint_names) <= 1:
        return torch.zeros(env.num_envs, device=env.device)

    mp_pos = torch.stack([env.scene[name].data.joint_pos[:, :2] for name in masspoint_names], dim=1)
    pairwise_dist = torch.cdist(mp_pos, mp_pos, p=2)

    num_agents = len(masspoint_names)
    tri_i, tri_j = torch.triu_indices(num_agents, num_agents, offset=1, device=env.device)
    dist_pairs = pairwise_dist[:, tri_i, tri_j]
    violations = torch.relu(min_distance - dist_pairs)
    return violations.mean(dim=1)

def tolerance(x: torch.Tensor, bounds: tuple[float, float], margin: float) -> torch.Tensor:
    """
    Mimics `tolerance` function from `mujoco_playground/mujoco_playground/_src/reward.py`
    Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: A tensor containing the values to be evaluated for tolerance.
        bounds:
            A tuple of floats specifying inclusive `(lower, upper)` bounds for
            the target interval. These can be infinite if the interval is unbounded at
            one or both ends, or they can be equal to one another if the target value
            is exact.
        margin:
            Float. Parameter that controls how steeply the output decreases as
            `x` moves out-of-bounds. * If `margin == 0` then the output will be 0 for
            all values of `x` outside of `bounds`. * If `margin > 0` then the output
            will decrease sigmoidally with increasing distance from the nearest bound.
            The output value when the distance from `x` to the nearest bound is equal
            to `margin` is expected to be a float between 0 and 1. Ignored if
            `margin == 0`.

    Returns:
        A tensor with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    def reciprocal_sigmoid(x: torch.Tensor, value_at_1: float = 0.1) -> torch.Tensor:
        """Returns a symmetric sigmoid that maps all real numbers to the range (0, 1), with `x=0` mapping to 1 and `x=±1` mapping to `value_at_1`."""
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be nonnegative and smaller than 1, got "
                f"{value_at_1}."
            )
        else:
            if not 0 < value_at_1 < 1:
                raise ValueError(
                    f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}."
                )
        scale = 1 / value_at_1 - 1
        return 1 / (torch.abs(x) * scale + 1)

    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("`margin` must be non-negative.")

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0.0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, reciprocal_sigmoid(d))

    return value

def goal_distance_reward(env, asset_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg) -> torch.Tensor:
    """Exponential reward based on the distance to the goal, with dense bonuses for being within 5 cm and 5 mm."""
    dist_2d = distance_to_goal(env, asset_cfg, goal_cfg).squeeze(-1)
    is_close = (dist_2d < 0.05).float()
    is_perfect = (dist_2d < 0.005).float()
    return torch.exp(-2.0 * dist_2d) + (10 * is_close) + (20 * is_perfect)

def action_magnitude_penalty(env) -> torch.Tensor:
    """Penalize the magnitude of the actions."""
    return torch.norm(env.action_manager.action, dim=-1)

def action_change_penalty(env) -> torch.Tensor:
    """Penalize the rate of change of the actions (L2 norm)."""
    return torch.norm(env.action_manager.action - env.action_manager.prev_action, dim=-1)
