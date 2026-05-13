""" Environment configuration for the push task. """
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
    get_cuboid_entity_cfg,
    get_push_target_marker_entity_cfg,
)
from multi_robot_rl.assets.robots import masspoint, ur10

# custom MDP imports and constants
from multi_robot_rl.tasks.push import mdp as push_mdp
from multi_robot_rl.common import mdp as common_mdp
import multi_robot_rl.configs.push_constants as push_constants


def make_push_env(play: bool = False) -> ManagerBasedRlEnvCfg:
    """
    Factory for the push task environment.

    Args:
        play: Single-env interactive mode when True.

    Returns:
        ManagerBasedRlEnvCfg: The configuration for the push task environment.
    """
    # UR10 base positions and rotations equally spaced around a circle, facing inward
    _ur10_poses = ur10.get_ur10_base_poses(push_constants.NUM_UR10S)
    _UR10_BASE_POSITIONS = [pos for pos, _ in _ur10_poses]
    _UR10_BASE_ROTATIONS = [rot for _, rot in _ur10_poses]

    robots = [masspoint.get_masspoint_robot_config_push_task(f"masspoint_{i}") for i in range(push_constants.NUM_MASSPOINTS)]
    robots += [
        ur10.get_ur10_robot_config_push_task(
            f"ur10_{i}",
            pos=_UR10_BASE_POSITIONS[i],
            rot=_UR10_BASE_ROTATIONS[i],
        )
        for i in range(push_constants.NUM_UR10S)
    ]

    entities = {robot.name: robot.entity_cfg for robot in robots}
    for i in range(push_constants.NUM_CUBOIDS):
        entities[f"cuboid_{i}"]      = get_cuboid_entity_cfg()
        entities[f"push_target_{i}"] = get_push_target_marker_entity_cfg()

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
    obs_terms.update({
        "cuboid_states_obs": ObservationTermCfg(
            func=push_mdp.cuboid_states_obs,
        ),
        "target_poses": ObservationTermCfg(
            func=push_mdp.target_poses_obs,
        ),
    })
    observations = {
        "actor":  ObservationGroupCfg(obs_terms),
        "critic": ObservationGroupCfg(obs_terms),
    }

    actions = {}
    for robot in robots:
        actions.update(robot.action_terms)

    rewards = {
        "cuboids_at_targets": RewardTermCfg(
            func=push_mdp.cuboid_placed_reward,
            weight=1.0,
        ),
                "action_magnitude": RewardTermCfg(
            func=common_mdp.action_magnitude_penalty,
            weight=-0.01,
        ),
        "out_of_bounds_penalty": RewardTermCfg(
            func=mjlab_rewards.is_terminated,
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
        "out_of_bounds": TerminationTermCfg(
            func=push_mdp.out_of_bounds,
            params={"robots": robots},
        ),
    }

    events = {
        "reset_cuboids_and_targets": EventTermCfg(
            func=push_mdp.reset_cuboids_and_targets,
            mode="reset",
            params={"num_cuboids": push_constants.NUM_CUBOIDS},
        ),
    }
    for robot in robots:
        events.update(robot.reset_terms)

    metrics = {
        "targets_reached_fraction": MetricsTermCfg(
            func=push_mdp.targets_reached_fraction,
        ),
    }

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.0, 0.0, 0.1),
        distance=2.5,
        elevation=-50.0,
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
            njmax=500,
        ),
        decimation=5,
        episode_length_s=5.0,
    )
