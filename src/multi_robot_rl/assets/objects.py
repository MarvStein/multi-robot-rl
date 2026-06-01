"""EntityCfg getters"""
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
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / "keyboard.xml"))
    return EntityCfg(spec_fn=get_spec)

def get_active_key_marker_entity_cfg(i: int) -> EntityCfg:
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / f"active_key_{i}.xml"))
    return EntityCfg(spec_fn=get_spec)

def get_reach_goal_marker_entity_cfg() -> EntityCfg:
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / "reach_goal.xml"))
    return EntityCfg(spec_fn=get_spec)

def get_push_target_marker_entity_cfg() -> EntityCfg:
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / "push_target.xml"))
    return EntityCfg(spec_fn=get_spec)

# =========================================================
# Cuboid assets (for push task)
# =========================================================

def get_cuboid_entity_cfg() -> EntityCfg:
    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(_GENERATED_XML_DIR / "cuboid.xml"))
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={"cuboid_x": 0.0, "cuboid_y": 0.0, "cuboid_yaw": 0.0},
            joint_vel={".*": 0.0},
        ),
    )

