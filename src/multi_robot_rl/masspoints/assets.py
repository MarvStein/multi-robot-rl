"""Asset configurations for masspoint environments."""
import math
from pathlib import Path
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlVelocityActuatorCfg

_XML_DIR = Path(__file__).parent / "xmls"

def _get_masspoint_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "masspoint.xml"))

def _get_goal_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML_DIR / "goal.xml"))

def get_masspoint_cfg(init_x: float = 0.0, init_y: float = 0.0) -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_masspoint_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlVelocityActuatorCfg(target_names_expr=("mp_x", "mp_y")),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={"mp_x": init_x, "mp_y": init_y},
            joint_vel={".*": 0.0},
        ),
    )

def get_goal_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_goal_spec,
    )

def get_masspoints_cfg(n: int) -> dict[str, EntityCfg]:
    """Create N masspoint entity configs.

    When N > 1 the initial positions are spread evenly on a circle of radius
    0.2 m so that masspoints do not overlap at reset time.
    """
    if n == 1:
        return {"masspoint_0": get_masspoint_cfg()}
    configs: dict[str, EntityCfg] = {}
    for i in range(n):
        angle = 2 * math.pi * i / n
        configs[f"masspoint_{i}"] = get_masspoint_cfg(
            init_x=0.2 * math.cos(angle),
            init_y=0.2 * math.sin(angle),
        )
    return configs

def get_goals_cfg(m: int) -> dict[str, EntityCfg]:
    """Create M goal entity configs."""
    return {f"goal_{j}": get_goal_cfg() for j in range(m)}
