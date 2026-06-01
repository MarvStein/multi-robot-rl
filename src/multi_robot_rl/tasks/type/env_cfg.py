""" Environment configuration for the type task. """
# mjlab imports
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.viewer.viewer_config import ViewerConfig
from mjlab.envs.mdp import terminations as mjlab_terminations
from mjlab.envs.mdp import rewards as mjlab_rewards

# Assets and Robots imports
from multi_robot_rl.assets.objects import (
    get_keyboard_entity_cfg,
    get_active_key_marker_entity_cfg,
)
from multi_robot_rl.assets.robots import masspoint, ur10

# custom MDP imports and constants
from multi_robot_rl.tasks.type import mdp as type_mdp
from multi_robot_rl.common import mdp as common_mdp
import multi_robot_rl.configs.type_constants as type_constants


def make_type_env(play: bool = False) -> ManagerBasedRlEnvCfg:
    """
    Factory for the type task environment.

    Args:
        play: Single-env interactive mode when True.

    Returns:
        ManagerBasedRlEnvCfg: The configuration for the type task environment.
    """
    # UR10 base positions and rotations equally spaced around a circle, facing inward
    _ur10_poses = ur10.get_ur10_base_poses(type_constants.NUM_UR10S)
    _UR10_BASE_POSITIONS = [pos for pos, _ in _ur10_poses]
    _UR10_BASE_ROTATIONS = [rot for _, rot in _ur10_poses]

    robots = [masspoint.get_masspoint_robot_config_type_task(f"masspoint_{i}") for i in range(type_constants.NUM_MASSPOINTS)]
    robots += [
        ur10.get_ur10_robot_config_type_task(
            f"ur10_{i}",
            pos=_UR10_BASE_POSITIONS[i],
            rot=_UR10_BASE_ROTATIONS[i],
        )
        for i in range(type_constants.NUM_UR10S)
    ]

    entities = {robot.name: robot.entity_cfg for robot in robots}
    entities["keyboard"] = get_keyboard_entity_cfg()
    for i in range(type_constants.NUM_ACTIVE_KEYS):
        entities[f"active_key_{i}"] = get_active_key_marker_entity_cfg(i)

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities=entities,
        sensors=common_mdp.make_inter_robot_contact_sensors(robots),
        num_envs=1 if play else 2048,
        env_spacing=3.0,
    )

    obs_terms = {}
    for robot in robots:
        obs_terms.update(robot.obs_terms)
    obs_terms["keyboard_state_obs"] = ObservationTermCfg(func=type_mdp.keyboard_state_obs)
    observations = {
        "actor": ObservationGroupCfg(obs_terms),
        "critic": ObservationGroupCfg(obs_terms),
    }

    actions = {}
    for robot in robots:
        actions.update(robot.action_terms)

    rewards = {
        "key_pressed_reward": RewardTermCfg(
            func=type_mdp.key_pressed_reward,
            weight=1.0,
        ),
        "wrong_key_penalty": RewardTermCfg(
            func=type_mdp.wrong_key_penalty,
            weight=-0.001,
        ),
        "action_rate_penalty": RewardTermCfg(
            func=mjlab_rewards.action_rate_l2,
            weight=-0.01,
        ),
        "out_of_bounds_penalty": RewardTermCfg(
            func=type_mdp.out_of_bounds,
            params={"robots": robots},
            weight=-10.0,
        ),
        "collision_penalty": RewardTermCfg(
            func=common_mdp.robot_collision_penalty,
            params={"robots": robots},
            weight=-1.0,
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mjlab_terminations.time_out, time_out=True),
        "out_of_bounds": TerminationTermCfg(func=type_mdp.out_of_bounds, params={"robots": robots}),
    }

    events = {
        "reset_keyboard": EventTermCfg(
            func=type_mdp.reset_keyboard_state,
            mode="reset",
        ),
        "update_keyboard": EventTermCfg(
            func=type_mdp.update_keyboard_state,
            mode="step",
        ),
    }
    for robot in robots:
        events.update(robot.reset_terms)

    metrics = {
        "throughput": MetricsTermCfg(
            func=type_mdp.throughput,
        ),
        "wrong_keys_per_episode": MetricsTermCfg(
            func=type_mdp.wrong_keys_per_episode,
        ),
    }

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=type_constants.CENTER_POS,
        azimuth=90.0,
        elevation=-23.8,
        distance=1.479,
    ) if play else ViewerConfig()

    return ManagerBasedRlEnvCfg(
        viewer=viewer,
        scene=scene,
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.01),
            njmax=300,
        ),
        decimation=5,
        episode_length_s=5.0,
        scale_rewards_by_dt=False,
    )
