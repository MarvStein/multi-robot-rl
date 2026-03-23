"""Masspoint environment configurations."""
import math
from pathlib import Path

import mujoco
from shapely import bounds
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import actions
from mjlab.envs.mdp import events as mjlab_events
from mjlab.envs.mdp import observations as mjlab_observations
from mjlab.envs.mdp import rewards as mjlab_rewards
from mjlab.scene import SceneCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.actuator import XmlVelocityActuatorCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sim import MujocoCfg, SimulationCfg

def tolerance(x: torch.Tensor, bounds: tuple[float, float], margin: float) -> torch.Tensor:
    """
    Mimics `tolerance` function from `mujoco_playground/mujoco_playground/_src/reward.py`
    Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: A tensor containing the values to be evaluated for tolerance.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded at
        one or both ends, or they can be equal to one another if the target value
        is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds. * If `margin == 0` then the output will be 0 for
        all values of `x` outside of `bounds`. * If `margin > 0` then the output
        will decrease sigmoidally with increasing distance from the nearest bound.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

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

_XML_DIR = Path(__file__).parent / "xmls"

def _get_masspoint_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "masspoint.xml"))

def _get_goal_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "goal.xml"))

def distance_to_goal(env, asset_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance between the asset and the goal."""
    asset_pos = env.scene[asset_cfg.name].data.joint_pos  # Joint positions for the two sliders gives actual XY
    goal_pos = env.scene[goal_cfg.name].data.root_link_pos_w
    # asset_pos: [num_envs, 2], goal_pos: [num_envs, 3]
    return torch.norm(asset_pos[:, :2] - goal_pos[:, :2], dim=-1).unsqueeze(-1)

def relative_goal_pos(env, asset_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg) -> torch.Tensor:
    """Relative 2D vector pointing from the asset to the goal."""
    asset_pos = env.scene[asset_cfg.name].data.joint_pos
    goal_pos = env.scene[goal_cfg.name].data.root_link_pos_w
    return goal_pos[:, :2] - asset_pos[:, :2]


def goal_distance_reward(env, asset_cfg: SceneEntityCfg, goal_cfg: SceneEntityCfg) -> torch.Tensor:
    """Exponential reward based on the distance to the goal, with dense bonuses for being within 5 cm and 5 mm."""
    # We strip the trailing dimension since the reward manager expects exactly [num_envs]
    dist_2d = distance_to_goal(env, asset_cfg, goal_cfg).squeeze(-1)
    # return tolerance(dist_2d, bounds=(0.0, 0.005), margin=0.4)
    is_close = (dist_2d < 0.05).float()  # Binary reward for being within 5 cm of the goal
    is_perfect = (dist_2d < 0.005).float()  # Bonus for being within 5 mm of the goal
    return torch.exp(-2.0 * dist_2d) + (10 * is_close) + (20 * is_perfect)

def action_magnitude_penalty(env) -> torch.Tensor:
    """Penalize the magnitude of the actions."""
    return torch.norm(env.action_manager.action, dim=-1)

def action_change_penalty(env) -> torch.Tensor:
    """Penalize the rate of change of the actions (L2 norm)."""
    return torch.norm(env.action_manager.action - env.action_manager.prev_action, dim=-1)

def reset_goal_position(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, pos_range: tuple):
    """Randomize the goal position."""
    goal = env.scene[asset_cfg.name]
    num_envs_to_reset = len(env_ids)
    
    # Generate random positions only for the environments undergoing reset
    x = torch.empty(num_envs_to_reset, device=env.device).uniform_(pos_range[0][0], pos_range[0][1])
    y = torch.empty(num_envs_to_reset, device=env.device).uniform_(pos_range[1][0], pos_range[1][1])
    z = torch.ones(num_envs_to_reset, device=env.device) * pos_range[2][0]  # Fixed Z for 2D
    new_pos = torch.stack([x, y, z], dim=-1)
    
    # Generate random positions + unrotated quaternions (since write_mocap_pose_to_sim expects 7D pose: pos + quat)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs_to_reset, 1)
    new_pose = torch.cat([new_pos, quats], dim=-1)
    # Entity data has write_mocap_pose_to_sim
    goal.write_mocap_pose_to_sim(mocap_pose=new_pose, env_ids=env_ids)

def time_out(env) -> torch.Tensor:
    """Terminate episode after a hard time limit."""
    return env.episode_length_buf >= env.max_episode_length

def get_masspoint_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_masspoint_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlVelocityActuatorCfg(target_names_expr=("mp_x", "mp_y")),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={"mp_x": 0.0, "mp_y": 0.0},
            joint_vel={".*": 0.0},
        ),
    )

def get_goal_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_goal_spec,
    )

def root_lin_vel_w_2d(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the root linear velocity in 2D."""
    # Since it's attached via slide joints, its velocity is its joint velocities
    return env.scene[asset_cfg.name].data.joint_vel[:, :2]

def masspoint_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Configuration for the single masspoint reach task."""
    
    # 1. Scene setup
    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "masspoint": get_masspoint_cfg(),
            "goal": get_goal_cfg(),
        },
        num_envs=1 if play else 2048,
        env_spacing=2.0,
    )
    
    # Scene Entity Configs
    mp_cfg = SceneEntityCfg("masspoint")
    goal_cfg = SceneEntityCfg("goal")

    # 2. Observations
    obs_terms = {
        "masspoint_vel": ObservationTermCfg(
            func=root_lin_vel_w_2d, 
            params={"asset_cfg": mp_cfg}
        ),
        "relative_goal_pos": ObservationTermCfg(
            func=relative_goal_pos,
            params={"asset_cfg": mp_cfg, "goal_cfg": goal_cfg}
        ),
        "distance": ObservationTermCfg(
            func=distance_to_goal,
            params={"asset_cfg": mp_cfg, "goal_cfg": goal_cfg}
        )
    }

    observations = {
        "actor": ObservationGroupCfg(obs_terms),
        "critic": ObservationGroupCfg(obs_terms),
    }

    # 3. Actions
    _actions = {
        "velocity": actions.JointVelocityActionCfg(
            entity_name="masspoint",
            actuator_names=("mp_x", "mp_y"),
            scale=1.0, # Policy output of 1.0 = target velocity of 1.0 m/s
        )
    }

    # 4. Rewards
    rewards = {
        "goal_distance": RewardTermCfg(
            func=goal_distance_reward,
            weight=1.0,
            params={"asset_cfg": mp_cfg, "goal_cfg": goal_cfg},
        ),
        "total_command": RewardTermCfg(
            func=action_magnitude_penalty,
            weight=-0.1,
        ),
        "action_rate": RewardTermCfg(
            func=action_change_penalty,
            weight=-0.1,
        )
    }

    # 5. Terminations
    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
    }

    # 6. Events (Resets)
    events = {
        "reset_masspoint": EventTermCfg(
            func=mjlab_events.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.01, 0.01),
                "asset_cfg": SceneEntityCfg("masspoint", joint_names=("mp_x", "mp_y")),
            },
        ),
        "reset_goal": EventTermCfg(
            func=reset_goal_position,
            mode="reset",
            params={
                "asset_cfg": goal_cfg,
                "pos_range": ((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.0)),
            }
        )
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=_actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.01)
        ),
        decimation=5,
        episode_length_s=5.0,
    )
