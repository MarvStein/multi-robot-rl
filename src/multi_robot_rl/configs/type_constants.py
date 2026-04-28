""" Constants for the type task """
from math import pi, radians

# =========================================================
# ROBOT COUNTS
# =========================================================

NUM_MASSPOINTS = 1
NUM_UR10S = 1

# =========================================================
# KEYBOARD GEOMETRY
# =========================================================

CENTER_POS = (0.0, 0.0, 0.45)
NUM_COLS = 6
NUM_ROWS = 3
KEY_SIZE = (0.05, 0.05, 0.02)   # (half_x, half_y, half_z) extents
KEY_SPACING = (0.12, 0.12)      # (x, y) center-to-center distance between adjacent keys
KEY_JOINT_RANGE = (-0.024, 0.0) # physical sliding distance allowed (z_min, 0)
ACTIVE_KEY_RGBA = (0.2, 0.8, 0.2, 0.5)
NEXT_KEY_RGBA = (0.8, 0.5, 0.2, 0.5)

# =========================================================
# MASSPOINT
# =========================================================

MP_SPRING_STIFFNESS = 0.5 # spring in z-direction

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

KEY_PRESS_THRESHOLD_PCT = 0.5  # key needs to be pressed this fraction of the way down to count
FREEZE_STEPS = 5               # policy steps to freeze the masspoint that pressed the key

# =========================================================
# AUTO-COMPUTED
# =========================================================

TOTAL_KEYS = NUM_COLS * NUM_ROWS

KEYBOARD_SIZE = (
    ((NUM_COLS - 1) * KEY_SPACING[0]) / 2.0 + KEY_SIZE[0],
    ((NUM_ROWS - 1) * KEY_SPACING[1]) / 2.0 + KEY_SIZE[1],
    KEY_SIZE[2]
)

KEY_X_START = -((NUM_COLS - 1) * KEY_SPACING[0]) / 2.0
KEY_Y_START = -((NUM_ROWS - 1) * KEY_SPACING[1]) / 2.0

KEY_X_STEP = KEY_SPACING[0]
KEY_Y_STEP = KEY_SPACING[1]

KEY_SURFACE_Z = CENTER_POS[2] + KEY_SIZE[2]

KEY_PRESS_THRESHOLD = KEY_JOINT_RANGE[0] * KEY_PRESS_THRESHOLD_PCT

MARKER_SIZE = (KEY_SIZE[0], KEY_SIZE[1], 0.002)
MARKER_Z = KEY_SURFACE_Z + MARKER_SIZE[2]

MP_RADIUS = 0.6 * KEY_SIZE[0]
MP_X_RANGE = (-KEYBOARD_SIZE[0]+CENTER_POS[0], KEYBOARD_SIZE[0]+CENTER_POS[0])
MP_Y_RANGE = (-KEYBOARD_SIZE[1]+CENTER_POS[1], KEYBOARD_SIZE[1]+CENTER_POS[1])
MP_Z_RANGE = (CENTER_POS[2] + KEY_JOINT_RANGE[0], KEY_SURFACE_Z + MP_RADIUS)

MP_X_POS_RANDOM_RANGE = MP_X_RANGE
MP_Y_POS_RANDOM_RANGE = MP_Y_RANGE
MP_Z_POS_RANDOM_RANGE = (MP_Z_RANGE[1], MP_Z_RANGE[1])

MP_X_VEL_RANDOM_RANGE = (-0.01, 0.01)
MP_Y_VEL_RANDOM_RANGE = (-0.01, 0.01)
MP_Z_VEL_RANDOM_RANGE = (0.0, 0.0)

OUT_OF_BOUNDS_MARGIN = MP_RADIUS # margin in x and y
EE_Z_MIN = CENTER_POS[2] + KEY_JOINT_RANGE[0]
EE_Z_MAX = KEY_SURFACE_Z + 0.3


def get_key_pos_2d(col, row):
    """Returns the (X, Y) real-world location for a given col & row."""
    x = CENTER_POS[0] + KEY_X_START + col * KEY_X_STEP
    y = CENTER_POS[1] + KEY_Y_START + row * KEY_Y_STEP
    return (x, y)
