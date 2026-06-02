"""Base robot configuration dataclass shared across all tasks and robot types."""
from dataclasses import dataclass
from mjlab.entity import EntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg

@dataclass
class RobotConfig:
    """Generic robot wrapper that decouples task definitions from specific robotic kinematics.

    Attributes:
        name: Unique scene entity name used to look up the robot in the scene registry.
        entity_cfg: EntityCfg describing the robot's MuJoCo model, actuators, and initial state.
        joint_names: Ordered tuple of joint names belonging to this robot.
        end_effector_site: Name of the MuJoCo site used as the end-effector reference point.
        root_body: Name of the root body of the kinematic chain, used for inter-robot collision checking.
        action_terms: Mapping from action term name to ActionTermCfg, wired into the action manager.
        obs_terms: Mapping from observation term name to ObservationTermCfg, wired into the observation manager.
        reset_terms: Mapping from event term name to EventTermCfg that handles per-episode joint resets.
    """
    name: str
    entity_cfg: EntityCfg
    joint_names: tuple[str, ...]
    end_effector_site: str
    root_body: str # name of the root of the kinematic chain, used for inter-robot collision checking
    action_terms: dict[str, ActionTermCfg]
    obs_terms: dict[str, ObservationTermCfg]
    reset_terms: dict[str, EventTermCfg]
