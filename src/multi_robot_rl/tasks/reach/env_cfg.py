""" Environment configuration for the reach task. """
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
from multi_robot_rl.assets.objects import get_reach_goal_marker_entity_cfg
from multi_robot_rl.assets.robots import masspoint, ur10

# custom MDP imports and constants
from multi_robot_rl.tasks.reach import mdp as reach_mdp
from multi_robot_rl.common import mdp as common_mdp
from multi_robot_rl.common import curriculum as common_curriculum
import multi_robot_rl.configs.reach_constants as reach_constants


def make_reach_env(play: bool = False) -> ManagerBasedRlEnvCfg:
    """
    Factory for the reach task environment.

    Args:
        play: Single-env interactive mode when True.

    Returns:
        ManagerBasedRlEnvCfg: The configuration for the reach task environment.
    """
    # UR10 base positions and rotations equally spaced around a circle, facing inward
    _ur10_poses = ur10.get_ur10_base_poses(reach_constants.NUM_UR10S)
    _UR10_BASE_POSITIONS = [pos for pos, _ in _ur10_poses]
    _UR10_BASE_ROTATIONS = [rot for _, rot in _ur10_poses]
    
    robots = [masspoint.get_masspoint_robot_config_reach_task(f"masspoint_{i}") for i in range(reach_constants.NUM_MASSPOINTS)]
    robots += [
        ur10.get_ur10_robot_config_reach_task(
            f"ur10_{i}",
            pos=_UR10_BASE_POSITIONS[i],
            rot=_UR10_BASE_ROTATIONS[i],
        )
        for i in range(reach_constants.NUM_UR10S)
    ]
    entities = {robot.name: robot.entity_cfg for robot in robots}
    for i in range(reach_constants.NUM_GOALS):
        entities[f"goal_{i}"] = get_reach_goal_marker_entity_cfg()

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
        "goal_states_obs": ObservationTermCfg(func=reach_mdp.goals_position_obs),
        "goal_reached_mask_obs": ObservationTermCfg(func=reach_mdp.goal_reached_mask_obs),
    })
    observations = {
        "actor": ObservationGroupCfg(obs_terms),
        "critic": ObservationGroupCfg(obs_terms),
    }

    actions = {}
    for robot in robots:
        actions.update(robot.action_terms)

    rewards = {
        "goal_reached": RewardTermCfg(
            func=reach_mdp.goal_reached_reward,
            params={"robots": robots, "play": play},
            weight=1.0,
        ),
        "action_rate_penalty": RewardTermCfg(
            func=mjlab_rewards.action_rate_l2,
            weight=-0.01,
        ),
        "out_of_bounds_penalty": RewardTermCfg(
            func=reach_mdp.out_of_bounds,
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
            func=reach_mdp.out_of_bounds,
            params={"robots": robots},
        ),
        "all_goals_reached": TerminationTermCfg(func=reach_mdp.all_goals_reached),
    }

    events = {}
    for robot in robots:
        events.update(robot.reset_terms)
    # IMPORTANT: reset_goals event must be triggered after robot resets to ensure goals are spawned in valid locations relative to robot positions.
    events["reset_goals"] = EventTermCfg(
        func=reach_mdp.reset_goal_state,
        mode="reset",
        params={
            "robots": robots,
            "radius": reach_constants.GOAL_WORKSPACE_RADIUS * 0.2,
            "dz": reach_constants.GOAL_WORKSPACE_HEIGHT / 2.0 * 0.2,
        },
    )

    metrics = {
        "goal_reached_fraction": MetricsTermCfg(
            func=reach_mdp.goal_reached_fraction,
        ),
    }

    curriculum = {
        "goal_spawn_curriculum": CurriculumTermCfg(
            func=common_curriculum.metric_event_curriculum,
            params={
                "event_name": "reset_goals",
                "metric_name": "goal_reached_fraction",
                "alpha": 0.01,
                "stages": [
                    {"metric_value": 0.0, "params": {"radius": reach_constants.GOAL_WORKSPACE_RADIUS * 0.2, "dz": reach_constants.GOAL_WORKSPACE_HEIGHT / 2.0 * 0.2}},
                    {"metric_value": 0.2, "params": {"radius": reach_constants.GOAL_WORKSPACE_RADIUS * 0.4, "dz": reach_constants.GOAL_WORKSPACE_HEIGHT / 2.0 * 0.4}},
                    {"metric_value": 0.4, "params": {"radius": reach_constants.GOAL_WORKSPACE_RADIUS * 0.6, "dz": reach_constants.GOAL_WORKSPACE_HEIGHT / 2.0 * 0.6}},
                    {"metric_value": 0.6, "params": {"radius": reach_constants.GOAL_WORKSPACE_RADIUS * 0.8, "dz": reach_constants.GOAL_WORKSPACE_HEIGHT / 2.0 * 0.8}},
                    {"metric_value": 0.8, "params": {"radius": reach_constants.GOAL_WORKSPACE_RADIUS * 1.0, "dz": reach_constants.GOAL_WORKSPACE_HEIGHT / 2.0 * 1.0}},
                ],
            },
        ),
    }

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.0, 0.0, 0.2),
        distance=2.0,
        elevation=-45.0,
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
            njmax=300,
        ),
        decimation=5,
        episode_length_s=5.0,
        scale_rewards_by_dt=False
    )
