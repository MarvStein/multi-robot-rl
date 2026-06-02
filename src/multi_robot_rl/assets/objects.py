"""EntityCfg factory functions for all scene objects used across tasks (keyboard, markers, cuboid)."""
import mujoco
from pathlib import Path
from mjlab.entity import EntityCfg
import multi_robot_rl.assets.scripts.generate_xmls as generate_xmls

_ASSETS_DIR = Path(__file__).parent
_GENERATED_XML_DIR = _ASSETS_DIR / "generated"
generate_xmls.generate_all()

# =========================================================
# Marker / keyboard assets
# =========================================================

def get_keyboard_entity_cfg() -> EntityCfg:
    """Return an EntityCfg that loads the generated keyboard.xml asset.

    Returns:
        EntityCfg backed by the keyboard XML generated from type_constants.
    """
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / "keyboard.xml"))
    return EntityCfg(spec_fn=get_spec)

def get_active_key_marker_entity_cfg(i: int) -> EntityCfg:
    """Return an EntityCfg that loads the generated active_key_{i}.xml marker asset.

    Args:
        i: Zero-based slot index of the active key marker.

    Returns:
        EntityCfg backed by the active_key_{i}.xml marker XML.
    """
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / f"active_key_{i}.xml"))
    return EntityCfg(spec_fn=get_spec)

def get_reach_goal_marker_entity_cfg() -> EntityCfg:
    """Return an EntityCfg that loads the generated reach_goal.xml sphere marker asset.

    Returns:
        EntityCfg backed by the reach_goal.xml marker XML.
    """
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / "reach_goal.xml"))
    return EntityCfg(spec_fn=get_spec)

def get_push_target_marker_entity_cfg() -> EntityCfg:
    """Return an EntityCfg that loads the generated push_target.xml marker asset.

    Returns:
        EntityCfg backed by the push_target.xml marker XML.
    """
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / "push_target.xml"))
    return EntityCfg(spec_fn=get_spec)

# =========================================================
# Cuboid assets (for push task)
# =========================================================

def get_cuboid_entity_cfg() -> EntityCfg:
    """Return an EntityCfg for the passive 3-DOF cuboid used in the push task.

    The cuboid has three joints (cuboid_x, cuboid_y, cuboid_yaw) and is
    initialized with all joint positions and velocities at zero.

    Returns:
        EntityCfg backed by the cuboid.xml asset with zeroed initial state.
    """
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / "cuboid.xml"))
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={"cuboid_x": 0.0, "cuboid_y": 0.0, "cuboid_yaw": 0.0},
            joint_vel={".*": 0.0},
        ),
    )

