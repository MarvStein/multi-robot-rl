"""Asset configurations for masspoint environments."""
from pathlib import Path
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlVelocityActuatorCfg
import multi_robot_rl.masspoints.keyboard_constants as kc
import multi_robot_rl.masspoints.generate_xmls as generate_xmls

# Automatically generate XML files from Jinja templates before loading them
generate_xmls.generate_all()

_XML_DIR = Path(__file__).parent / "xmls"

def _get_masspoint_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "masspoint.xml"))

def _get_masspoint_3d_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "masspoint_3d.xml"))

def _get_goal_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "goal.xml"))

def _get_keyboard_board_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "keyboard_board.xml"))

def _get_active_key_marker_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "active_key.xml"))

def _get_next_key_marker_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "next_key.xml"))

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

def get_masspoint_3d_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_masspoint_3d_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlVelocityActuatorCfg(target_names_expr=("mp_x", "mp_y", "mp_z")),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={"mp_x": 0.0, "mp_y": 0.0, "mp_z": kc.MP_Z_RANGE[1]},
            joint_vel={".*": 0.0},
        ),
    )

def get_goal_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_goal_spec,
    )

def get_keyboard_board_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_keyboard_board_spec,
    )

def get_active_key_marker_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_active_key_marker_spec,
    )

def get_next_key_marker_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_next_key_marker_spec,
    )
