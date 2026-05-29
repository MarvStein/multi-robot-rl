""" Constants for the push task """
from math import pi, radians

# =========================================================
# ROBOT COUNTS
# =========================================================

NUM_MASSPOINTS = 2
NUM_UR10S = 1
NUM_CUBOIDS = 3  # also the number of target poses

# =========================================================
# CUBOIDS
# =========================================================

# HX, HY, HZ are half-extents
CUBOID_HX = 0.1
CUBOID_HY = 0.05
CUBOID_HZ = 0.05
CUBOID_MASS = 0.5
CUBOID_RGBA = (0.8, 0.4, 0.1, 1.0)
CUBOID_SPAWN_RADIUS = 0.3  # used for robot/masspoint spawn ranges and joint limits
CUBOID_MAX_PUSH_DISTANCE = 0.3  # max initial distance between a cuboid and its paired target (curriculum axis)
CUBOID_YAW_SPAWN_RANGE = (-radians(60), radians(60))

# =========================================================
# TARGETS
# =========================================================

TARGET_SPAWN_RADIUS = CUBOID_SPAWN_RADIUS
TARGET_YAW_RANGE = (-radians(60), radians(60))
PUSH_TARGET_RGBA = (0.2, 0.8, 0.8, 0.6)

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

POSITION_THRESHOLD = 0.10        # success threshold for xy distance from cuboid to target
YAW_THRESHOLD = radians(20)      # success threshold for difference in yaw between cuboid and target

# =========================================================
# AUTO-COMPUTED
# =========================================================

MP_RADIUS = 0.03

MP_SPAWN_RADIUS = CUBOID_SPAWN_RADIUS
MP_SPAWN_HEIGHT = MP_RADIUS # masspoint spawns exactly on that height
MP_SPRING_STIFFNESS = 0.0  # no z-spring; masspoint cannot move vertically
MP_GRAVCOMP = False

# joint ranges are rectangular but out of bounds check and spawning is circular and more strict.
MP_X_RANGE = (-CUBOID_SPAWN_RADIUS, CUBOID_SPAWN_RADIUS)
MP_Y_RANGE = (-CUBOID_SPAWN_RADIUS, CUBOID_SPAWN_RADIUS)
MP_Z_RANGE = (MP_SPAWN_HEIGHT, MP_SPAWN_HEIGHT + 1e-3) # workaround to reuse 3 dof masspoint jinja template

CUBOID_X_RANGE = (-CUBOID_SPAWN_RADIUS, CUBOID_SPAWN_RADIUS)
CUBOID_Y_RANGE = (-CUBOID_SPAWN_RADIUS, CUBOID_SPAWN_RADIUS)

MP_X_VEL_RANDOM_RANGE = (-0.01, 0.01)
MP_Y_VEL_RANDOM_RANGE = (-0.01, 0.01)
MP_Z_VEL_RANDOM_RANGE = (0.0, 0.0)

OUT_OF_BOUNDS_RADIUS = CUBOID_SPAWN_RADIUS + 0.2
OUT_OF_BOUNDS_HEIGHT = 0.8