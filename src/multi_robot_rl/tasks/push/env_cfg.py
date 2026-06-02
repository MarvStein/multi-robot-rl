""" Environment configuration for the push task. """
# mjlab imports
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
from multi_robot_rl.common import curriculum as common_curriculum
import multi_robot_rl.configs.push_constants as push_constants


def make_push_env(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build and return the configuration for the push task environment.

    Args:
        play: When True, creates a single-environment interactive setup with
            curriculum constraints disabled and an adjusted viewer.

    Returns:
        Fully populated ManagerBasedRlEnvCfg covering scene, observations,
        actions, rewards, terminations, events, metrics, and curriculum.
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
        "cuboid_states_obs": ObservationTermCfg(func=push_mdp.cuboid_states_obs),
        "target_poses_obs": ObservationTermCfg(func=push_mdp.target_poses_obs),
        "target_satisfied_mask_obs": ObservationTermCfg(func=push_mdp.target_satisfied_mask_obs),
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
        "action_rate_penalty": RewardTermCfg(
            func=mjlab_rewards.action_rate_l2,
            weight=-0.01,
        ),
        "out_of_bounds_penalty": RewardTermCfg(
            func=push_mdp.out_of_bounds,
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
        "out_of_bounds": TerminationTermCfg(
            func=push_mdp.out_of_bounds,
            params={"robots": robots},
        ),
        "all_targets_reached": TerminationTermCfg(func=push_mdp.all_targets_reached),
    }

    events = {
        "reset_cuboids_and_targets": EventTermCfg(
            func=push_mdp.reset_cuboids_and_targets,
            mode="reset",
            params={
                "play": play,
                "cuboid_distance_fraction": 0.2,
            },
        ),
    }
    for robot in robots:
        events.update(robot.reset_terms)

    metrics = {
        "targets_reached_fraction": MetricsTermCfg(
            func=push_mdp.targets_reached_fraction,
        ),
    }

    curriculum = {
        "target_spawn_curriculum": CurriculumTermCfg(
            func=common_curriculum.metric_event_curriculum,
            params={
                "event_name": "reset_cuboids_and_targets",
                "metric_name": "targets_reached_fraction",
                "alpha": 1e-3,
                "stages": [
                    {"metric_value": 0.0, "params": {"cuboid_distance_fraction": 0.2}},
                    {"metric_value": 0.2, "params": {"cuboid_distance_fraction": 0.4}},
                    {"metric_value": 0.4, "params": {"cuboid_distance_fraction": 0.6}},
                    {"metric_value": 0.6, "params": {"cuboid_distance_fraction": 0.8}},
                    {"metric_value": 0.8, "params": {"cuboid_distance_fraction": 1.0}},
                ],
            },
        ),
    }

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.0, 0.0, 0.2),
        azimuth=136.3,
        elevation=-26.1,
        distance=2.21,
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
        curriculum=curriculum,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.01),
            njmax=500,
        ),
        decimation=5,
        episode_length_s=5.0,
        scale_rewards_by_dt=False,
    )
