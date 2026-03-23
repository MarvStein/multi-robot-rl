"""Masspoint environment configurations."""
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
