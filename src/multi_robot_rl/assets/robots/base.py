from dataclasses import dataclass
from mjlab.entity import EntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg

@dataclass
class RobotConfig:
    """Generic Robot wrapper decoupling tasks from specific robotic kinematics."""
    name: str
    entity_cfg: EntityCfg
    joint_names: tuple[str, ...]
    end_effector_site: str
    action_terms: dict[str, ActionTermCfg]
    obs_terms: dict[str, ObservationTermCfg]
    reset_terms: dict[str, EventTermCfg]
