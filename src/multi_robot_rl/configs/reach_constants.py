""" Constants for the reach task """
from math import pi, radians

# =========================================================
# ROBOT COUNTS
# =========================================================

NUM_MASSPOINTS = 1
NUM_UR10S = 0

# =========================================================
# GOALS
# =========================================================

NUM_GOALS = 1
MARKER_SIZE = (0.05, 0.05, 0.002)
REACH_GOAL_RGBA = (0.2, 0.8, 0.8, 0.6)

# =========================================================
# GOAL WORKSPACE
# =========================================================

# goals are spawned uniformly distributed in a cylinder,
# centered at the origin with the following dimensions:
GOAL_WORKSPACE_RADIUS = 0.3
GOAL_WORKSPACE_HEIGHT = 0.8

# =========================================================
# MASSPOINT
# =========================================================

MP_RADIUS = 0.03

# masspoint is spawned uniformly distributed in a cylinder,
# centered at the origin with the following dimensions:
MP_SPAWN_RADIUS = GOAL_WORKSPACE_RADIUS
MP_SPAWN_HEIGHT = GOAL_WORKSPACE_HEIGHT

MP_X_VEL_RANDOM_RANGE = (-0.01, 0.01)
MP_Y_VEL_RANDOM_RANGE = (-0.01, 0.01)
MP_Z_VEL_RANDOM_RANGE = (-0.01, 0.01)

# =========================================================
# UR10
# =========================================================

UR10_JOINT_RESET_RANGE = (-radians(10), radians(10))

UR10_JOINT_POS_LIMITS = {
    "shoulder_pan_joint":  (-pi/2 - radians(45), -pi/2 + radians(45)),
    "shoulder_lift_joint": (-pi/2 - radians(45), -pi/2 + radians(45)),
    "elbow_joint":         ( pi/2 - radians(45),  pi/2 + radians(45)),
    "wrist_1_joint":       (-pi/2 - radians(45), -pi/2 + radians(45)),
    "wrist_2_joint":       (-pi/2 - radians(45), -pi/2 + radians(45)),
    "wrist_3_joint":       (0,0),
}

# =========================================================
# THRESHOLDS
# =========================================================

GOAL_REACH_THRESHOLD = 0.03

# =========================================================
# AUTO-COMPUTED
# =========================================================

# joint ranges are rectangular but out of bounds check and spawning is circular and more strict.
MP_X_RANGE = (-GOAL_WORKSPACE_RADIUS, GOAL_WORKSPACE_RADIUS)
MP_Y_RANGE = (-GOAL_WORKSPACE_RADIUS, GOAL_WORKSPACE_RADIUS)
MP_Z_RANGE = (0.0, GOAL_WORKSPACE_HEIGHT)
MP_SPRING_STIFFNESS = 0.0  # no z-spring; masspoint floats freely via gravity compensation
MP_GRAVCOMP = True

OUT_OF_BOUNDS_RADIUS = GOAL_WORKSPACE_RADIUS + 0.2
OUT_OF_BOUNDS_HEIGHT = GOAL_WORKSPACE_HEIGHT + 0.2


