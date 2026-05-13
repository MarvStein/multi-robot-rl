import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensorCfg, ContactMatch
from multi_robot_rl.assets.robots.base import RobotConfig


def _get_ee_config(env: ManagerBasedRlEnv, robot_name: str, ee_site: str) -> SceneEntityCfg:
    """Lazily create and resolve a per-robot EE site config, cached on env."""
    key = f"_ee_site_cfg_{robot_name}"
    if not hasattr(env, key):
        cfg = SceneEntityCfg(robot_name, site_names=(ee_site,))
        cfg.resolve(env.scene)
        setattr(env, key, cfg)
    return getattr(env, key)


def get_ee_positions(env: ManagerBasedRlEnv, robots: list[RobotConfig]) -> torch.Tensor:
    """Return world-frame EE positions for all robots: (num_envs, num_robots, 3)."""
    cfgs = [_get_ee_config(env, r.name, r.end_effector_site) for r in robots]
    return torch.stack(
        [env.scene[cfg.name].data.site_pos_w[:, cfg.site_ids, :].squeeze(1) for cfg in cfgs],
        dim=1,
    )


def ee_pos_obs(env: ManagerBasedRlEnv, robot_name: str, ee_site: str) -> torch.Tensor:
    """Obs-term: world-frame EE position for a single robot: (num_envs, 3)."""
    cfg = _get_ee_config(env, robot_name, ee_site)
    return env.scene[cfg.name].data.site_pos_w[:, cfg.site_ids, :].squeeze(1)

def action_magnitude_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Action magnitude generic penalty using L2 norm."""
    return torch.norm(env.action_manager.action, dim=-1)

def action_change_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions (L2 norm)."""
    # TODO: check mjlab.envs.mdp.rewards for built-in
    raise NotImplementedError()

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
    """Return the absolute joint positions of the asset."""
    # Assuming joint_ids are either not provided (all) or slice
    if asset_cfg.joint_ids is None:
        return env.scene[asset_cfg.name].data.joint_pos
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

def joint_vel_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the absolute joint velocities of the asset."""
    if asset_cfg.joint_ids is None:
        return env.scene[asset_cfg.name].data.joint_vel
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]