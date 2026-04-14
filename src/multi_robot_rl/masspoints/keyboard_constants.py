# Constants for the keyboard geometry and environment definitions
# These are used across the codebase, including in XML generation and documentation.

# =========================================================
# CORE USER-CONFIGURABLE PARAMETERS
# =========================================================

# Center position of the entire keyboard layout
CENTER_POS = (0.0, 0.0, 0.05)
# Number of keys
NUM_COLS = 6
NUM_ROWS = 3
# Key geometry
KEY_SIZE = (0.04, 0.04, 0.02)  # (half_x, half_y, half_z) extents
KEY_SPACING = (0.10, 0.10)     # (x, y) center-to-center distance between adjacent keys
# Key movement properties relative to the board base
KEY_Z_POS_REL = 0.06           # Base Z offset of the key from the center
KEY_JOINT_RANGE = (-0.024, 0.0)# The physical sliding distance allowed (z_min, 0)
# Interaction Thresholds
KEY_PRESS_THRESHOLD_PCT = 0.5 # (e.g. 0.9 means the key needs to be pressed 90% of the way down to count as a press)

# Masspoint features
NUM_MASSPOINTS = 2
MP_RADIUS = 0.02
MP_XY_RANGE = (-1.0, 1.0)
MP_Z_RANGE = (0.0, 0.15)
# Out of bounds margin
OUT_OF_BOUNDS_MARGIN = MP_RADIUS
FREEZE_STEPS = 5  # Number of policy steps to freeze the masspoint that pressed the key

# =========================================================
# AUTO-COMPUTED DERIVED CONSTANTS
# =========================================================

TOTAL_KEYS = NUM_COLS * NUM_ROWS

# Total physical size of the keyboard bounding area (half extents)
KEYBOARD_SIZE = (
    ((NUM_COLS - 1) * KEY_SPACING[0]) / 2.0 + KEY_SIZE[0],
    ((NUM_ROWS - 1) * KEY_SPACING[1]) / 2.0 + KEY_SIZE[1],
    KEY_SIZE[2]
)

# Start locations of the bottom-left key (local offsets from center)
KEY_X_START = -((NUM_COLS - 1) * KEY_SPACING[0]) / 2.0
KEY_Y_START = -((NUM_ROWS - 1) * KEY_SPACING[1]) / 2.0

# Steps (for compatibility with loop generators)
KEY_X_STEP = KEY_SPACING[0]
KEY_Y_STEP = KEY_SPACING[1]

# Top surface of an unpressed key in world coordinates
KEY_SURFACE_Z = CENTER_POS[2] + KEY_Z_POS_REL + KEY_SIZE[2]

# The threshold at which a key is considered pressed (absolute relative offset)
KEY_PRESS_THRESHOLD = KEY_JOINT_RANGE[0] * KEY_PRESS_THRESHOLD_PCT

# Marker geometry
MARKER_SIZE = (KEY_SIZE[0], KEY_SIZE[1], 0.002) # a thin box atop the key
MARKER_Z = KEY_SURFACE_Z + MARKER_SIZE[2] # slightly above the surface to prevent z-fighting

# Helper to compute 2d locations of keys generically without knowing loop iterators
def get_key_pos_2d(col, row):
    """Returns the (X, Y) real-world location for a given col & row."""
    x = CENTER_POS[0] + KEY_X_START + col * KEY_X_STEP
    y = CENTER_POS[1] + KEY_Y_START + row * KEY_Y_STEP
    return (x, y)

