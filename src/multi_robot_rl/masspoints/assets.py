"""Asset configurations for masspoint environments."""
from pathlib import Path
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlVelocityActuatorCfg

_XML_DIR = Path(__file__).parent / "xmls"

def _get_masspoint_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "masspoint.xml"))

def _get_goal_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "goal.xml"))

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

def get_goal_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_goal_spec,
    )
