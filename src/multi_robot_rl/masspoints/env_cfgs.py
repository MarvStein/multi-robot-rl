"""Masspoint environment configurations."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import actions
from mjlab.envs.mdp import events as mjlab_events
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sim import MujocoCfg, SimulationCfg

# Import local modules
from multi_robot_rl.masspoints.assets import get_masspoint_cfg, get_goal_cfg
import multi_robot_rl.masspoints.mdp as mdp


def _validate_multi_config(num_masspoints: int, num_goals: int):
    if num_masspoints <= 0:
        raise ValueError(f"num_masspoints must be > 0, got {num_masspoints}.")
    if num_goals <= 0:
        raise ValueError(f"num_goals must be > 0, got {num_goals}.")
    if num_goals < num_masspoints:
        raise ValueError(
            "num_goals must be greater than or equal to num_masspoints "
            f"(got num_goals={num_goals}, num_masspoints={num_masspoints})."
        )

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
            func=mdp.observations.root_lin_vel_w_2d, 
            params={"asset_cfg": mp_cfg}
        ),
        "relative_goal_pos": ObservationTermCfg(
            func=mdp.observations.relative_goal_pos,
            params={"asset_cfg": mp_cfg, "goal_cfg": goal_cfg}
        ),
        "distance": ObservationTermCfg(
            func=mdp.observations.distance_to_goal,
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
            func=mdp.rewards.goal_distance_reward,
            weight=1.0,
            params={"asset_cfg": mp_cfg, "goal_cfg": goal_cfg},
        ),
        "total_command": RewardTermCfg(
            func=mdp.rewards.action_magnitude_penalty,
            weight=-0.1,
        ),
        "action_rate": RewardTermCfg(
            func=mdp.rewards.action_change_penalty,
            weight=-0.1,
        )
    }

    # 5. Terminations
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.terminations.time_out, time_out=True),
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
            func=mdp.events.reset_goal_position,
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


def masspoint_multi_reach_env_cfg(
    play: bool = False,
    num_masspoints: int = 2,
    num_goals: int = 2,
    goal_reach_threshold: float = 0.03,
    goal_respawn_delay_steps: int = 10,
    goal_spawn_range_xy: tuple[float, float] = (-0.5, 0.5),
    min_agent_separation_dist: float = 0.06,
    min_goal_to_masspoint_spawn_dist: float = 0.08,
    spawn_rejection_max_tries: int = 12,
) -> ManagerBasedRlEnvCfg:
    """Configuration for centralized multi-masspoint/multi-goal cooperative reach."""
    _validate_multi_config(num_masspoints, num_goals)

    masspoint_names = tuple(f"masspoint_{idx}" for idx in range(num_masspoints))
    goal_names = tuple(f"goal_{idx}" for idx in range(num_goals))
    pos_range = (
        (goal_spawn_range_xy[0], goal_spawn_range_xy[1]),
        (goal_spawn_range_xy[0], goal_spawn_range_xy[1]),
        (0.0, 0.0),
    )

    entities = {name: get_masspoint_cfg() for name in masspoint_names}
    entities.update({name: get_goal_cfg() for name in goal_names})

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities=entities,
        num_envs=1 if play else 2048,
        env_spacing=2.0,
    )

    obs_terms = {
        "centralized_state": ObservationTermCfg(
            func=mdp.observations.centralized_state,
            params={
                "masspoint_names": masspoint_names,
                "goal_names": goal_names,
                "include_goal_activity": True,
            },
        )
    }
    observations = {
        "actor": ObservationGroupCfg(obs_terms),
        "critic": ObservationGroupCfg(obs_terms),
    }

    _actions = {
        f"velocity_{name}": actions.JointVelocityActionCfg(
            entity_name=name,
            actuator_names=("mp_x", "mp_y"),
            scale=1.0,
        )
        for name in masspoint_names
    }

    rewards = {
        "goal_completion": RewardTermCfg(
            func=mdp.rewards.cooperative_goal_completion_reward,
            weight=20.0,
        ),
        "goal_progress": RewardTermCfg(
            func=mdp.rewards.nearest_goal_progress_reward,
            weight=1.0,
            params={
                "masspoint_names": masspoint_names,
                "goal_names": goal_names,
            },
        ),
        "close_agents": RewardTermCfg(
            func=mdp.rewards.proximity_penalty,
            weight=-2.0,
            params={
                "masspoint_names": masspoint_names,
                "min_distance": min_agent_separation_dist,
            },
        ),
        "total_command": RewardTermCfg(
            func=mdp.rewards.action_magnitude_penalty,
            weight=-0.05,
        ),
        "action_rate": RewardTermCfg(
            func=mdp.rewards.action_change_penalty,
            weight=-0.05,
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.terminations.time_out, time_out=True),
    }

    events = {
        "reset_goals": EventTermCfg(
            func=mdp.events.reset_multi_goals,
            mode="reset",
            params={
                "goal_names": goal_names,
                "masspoint_names": masspoint_names,
                "pos_range": pos_range,
                "min_dist_to_masspoints": min_goal_to_masspoint_spawn_dist,
                "rejection_max_tries": spawn_rejection_max_tries,
            },
        ),
        "update_goal_lifecycle": EventTermCfg(
            func=mdp.events.update_multi_goals_lifecycle,
            mode="post_physics",
            params={
                "goal_names": goal_names,
                "masspoint_names": masspoint_names,
                "reach_threshold": goal_reach_threshold,
                "respawn_delay_steps": goal_respawn_delay_steps,
                "pos_range": pos_range,
                "min_dist_to_masspoints": min_goal_to_masspoint_spawn_dist,
                "rejection_max_tries": spawn_rejection_max_tries,
            },
        ),
    }

    for name in masspoint_names:
        events[f"reset_{name}"] = EventTermCfg(
            func=mjlab_events.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.01, 0.01),
                "asset_cfg": SceneEntityCfg(name, joint_names=("mp_x", "mp_y")),
            },
        )

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=_actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.01),
        ),
        decimation=5,
        episode_length_s=5.0,
    )
