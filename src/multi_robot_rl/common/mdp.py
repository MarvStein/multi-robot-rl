"""MDP helper functions for multi-robot environments.

Provides observation terms, reward terms, and scene-setup utilities used by
ManagerBasedRlEnv configurations: end-effector position accessors, inter-robot
contact sensor factories, collision penalties, and absolute joint state
observations.
"""

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensorCfg, ContactMatch
from multi_robot_rl.assets.robots.base import RobotConfig


def _get_ee_config(env: ManagerBasedRlEnv, robot_name: str, ee_site: str) -> SceneEntityCfg:
    """Lazily create and resolve a per-robot EE site config, cached on env.

    Args:
        env: The running environment whose ``scene`` is used to resolve the config.
        robot_name: Name of the robot asset as registered in the scene.
        ee_site: Name of the MuJoCo site on the robot that represents the end-effector.

    Returns:
        A resolved ``SceneEntityCfg`` for the named robot and EE site, with
        ``site_ids`` populated. Subsequent calls with the same ``robot_name``
        return the cached instance without re-resolving.

    Side Effects:
        - Attaches the resolved config to ``env`` under the attribute
          ``_ee_site_cfg_<robot_name>`` on first call.
    """
    key = f"_ee_site_cfg_{robot_name}"
    if not hasattr(env, key):
        cfg = SceneEntityCfg(robot_name, site_names=(ee_site,))
        cfg.resolve(env.scene)
        setattr(env, key, cfg)
    return getattr(env, key)


def get_ee_positions(env: ManagerBasedRlEnv, robots: list[RobotConfig]) -> torch.Tensor:
    """Return world-frame end-effector positions for every robot in the scene.

    Args:
        env: The running environment providing scene data.
        robots: Ordered list of robot configs; each must expose ``name`` and
            ``end_effector_site`` attributes.

    Returns:
        Float tensor of shape ``(num_envs, num_robots, 3)`` containing the
        world-frame XYZ position of each robot's end-effector site.
    """
    cfgs = [_get_ee_config(env, r.name, r.end_effector_site) for r in robots]
    return torch.stack(
        [env.scene[cfg.name].data.site_pos_w[:, cfg.site_ids, :].squeeze(1) for cfg in cfgs],
        dim=1,
    )



def ee_pos_obs(env: ManagerBasedRlEnv, robot_name: str, ee_site: str) -> torch.Tensor:
    """Observation term returning the world-frame end-effector position for a single robot.

    Args:
        env: The running environment providing scene data.
        robot_name: Name of the robot asset as registered in the scene.
        ee_site: Name of the MuJoCo site on the robot that represents the end-effector.

    Returns:
        Float tensor of shape ``(num_envs, 3)`` containing the world-frame XYZ
        position of the named robot's end-effector site.
    """
    cfg = _get_ee_config(env, robot_name, ee_site)
    return env.scene[cfg.name].data.site_pos_w[:, cfg.site_ids, :].squeeze(1)

def make_inter_robot_contact_sensors(robots: list[RobotConfig]) -> tuple[ContactSensorCfg, ...]:
    """Create one ContactSensor per robot pair to detect physical collisions via MuJoCo contact data.

    Each sensor monitors robot_i's full body subtree for contacts with robot_j's subtree.
    Add the returned tuple to SceneCfg.sensors and pass robots to robot_collision_penalty.
    """
    sensors = []
    for i, robot_i in enumerate(robots):
        for j, robot_j in enumerate(robots):
            if j <= i:
                continue
            sensors.append(ContactSensorCfg(
                name=f"contact_{robot_i.name}_vs_{robot_j.name}",
                primary=ContactMatch(mode="subtree", pattern=robot_i.root_body, entity=robot_i.name),
                secondary=ContactMatch(mode="subtree", pattern=robot_j.root_body, entity=robot_j.name),
                fields=("found",),
                reduce="none",
                num_slots=1,
            ))
    return tuple(sensors)


def robot_collision_penalty(env: ManagerBasedRlEnv, robots: list[RobotConfig]) -> torch.Tensor:
    """Return 1.0 for each env where any two robots are physically in contact, else 0.0.

    Requires ContactSensors from make_inter_robot_contact_sensors() to be present in the scene.
    """
    if len(robots) < 2:
        return torch.zeros(env.num_envs, device=env.device)

    collision = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i, robot_i in enumerate(robots):
        for j, robot_j in enumerate(robots):
            if j <= i:
                continue
            sensor = env.scene[f"contact_{robot_i.name}_vs_{robot_j.name}"]
            found = sensor.data.found  # [num_envs, P] where P=1 (root body)
            collision |= (found > 0).any(dim=-1)

    return collision.float()

def joint_pos_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the absolute joint positions of the asset.

    Args:
        env: The running environment providing scene data.
        asset_cfg: Scene entity config identifying the asset and optionally a
            subset of joint IDs to return; if ``joint_ids`` is ``None`` all
            joints are returned.

    Returns:
        Float tensor of shape ``(num_envs, num_joints)`` containing the current
        joint positions in radians (or metres for prismatic joints).
    """
    # Assuming joint_ids are either not provided (all) or slice
    if asset_cfg.joint_ids is None:
        return env.scene[asset_cfg.name].data.joint_pos
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

def joint_vel_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the absolute joint velocities of the asset.

    Args:
        env: The running environment providing scene data.
        asset_cfg: Scene entity config identifying the asset and optionally a
            subset of joint IDs to return; if ``joint_ids`` is ``None`` all
            joints are returned.

    Returns:
        Float tensor of shape ``(num_envs, num_joints)`` containing the current
        joint velocities in radians per second (or metres per second for
        prismatic joints).
    """
    if asset_cfg.joint_ids is None:
        return env.scene[asset_cfg.name].data.joint_vel
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]