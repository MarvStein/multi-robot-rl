"""Generate MuJoCo XML assets for all tasks from Jinja2 templates and task constants, writing files to assets/generated/."""
import os
import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
# Add src to python path to import configs
SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(SRC_DIR))
import multi_robot_rl.configs.type_constants as type_constants
import multi_robot_rl.configs.reach_constants as reach_constants
import multi_robot_rl.configs.push_constants as push_constants

ASSETS_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = SRC_DIR.parent / "docs"
GENERATED_XML_DIR = ASSETS_DIR / "generated" # directory to store generated XML files (gitignored)
GENERATED_XML_DIR.mkdir(parents=True, exist_ok=True)

def _generate_keyboard_xml():
    """Generate the keyboard XML from the keyboard Jinja2 template using type_constants.

    Side Effects:
        - Writes the rendered XML to assets/generated/keyboard.xml.
    """
    env = Environment(loader=FileSystemLoader(str(ASSETS_DIR)))
    template = env.get_template("keyboard.xml.jinja")
    
    xs = [type_constants.KEY_X_START + i * type_constants.KEY_X_STEP for i in range(type_constants.NUM_COLS)]
    ys = [type_constants.KEY_Y_START + i * type_constants.KEY_Y_STEP for i in range(type_constants.NUM_ROWS)]
    
    keys = []
    idx = 0
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            keys.append({
                "name": f"key_{idx}",
                "x": round(x, 4),
                "y": round(y, 4),
                "col": col,
                "row": row
            })
            idx += 1
            
    context = {
        "CENTER_POS": type_constants.CENTER_POS,
        "KEY_JOINT_RANGE": type_constants.KEY_JOINT_RANGE,
        "KEY_SIZE": type_constants.KEY_SIZE,
        "keys": keys
    }
    
    xml_str = template.render(**context)
    
    out_path = GENERATED_XML_DIR / "keyboard.xml"
    with open(out_path, "w") as f:
        f.write(xml_str)

def _generate_marker_xml(name: str, rgba: tuple, marker_size: float):
    """Generate a marker XML file from the marker Jinja2 template.

    Args:
        name: Name of the marker (e.g., "active_key_0"); also used as the output filename stem.
        rgba: RGBA color tuple for the marker (e.g., (0.2, 0.8, 0.2, 0.5)).
        marker_size: Size of the marker sphere.

    Side Effects:
        - Writes the rendered XML to assets/generated/{name}.xml.
    """
    env = Environment(loader=FileSystemLoader(str(ASSETS_DIR)))
    template = env.get_template("marker.xml.jinja")
    
    context = {
        "name": name,
        "rgba": " ".join(map(str, rgba)),
        "MARKER_SIZE": marker_size
    }
    
    xml_str = template.render(**context)
    
    out_path = GENERATED_XML_DIR / f"{name}.xml"
    with open(out_path, "w") as f:
        f.write(xml_str)

def _generate_masspoint_xml(suffix: str,
                            radius: float,
                            x_range: tuple[float, float],
                            y_range: tuple[float, float],
                            z_range: tuple[float, float],
                            mass: float,
                            diaginertia: float,
                            kv: float,
                            spring_stiffness: float = 0.5,
                            gravcomp: bool = False):
    """
    Generates the masspoint XML file based on the provided ranges.
    The generated XML is saved to assets/generated/masspoint_{suffix}.xml.

    - spring_stiffness > 0 adds a z-spring pulling to z_range[1] (used in type task)
    - gravcomp=True adds gravcomp="1" to the body (used in reach task for free 3D flight)

    Args:
        suffix: A string suffix to distinguish different masspoint XMLs (e.g. "type_task")
        radius: Radius of the masspoint
        x_range: Tuple specifying the (min, max) range for the x position
        y_range: Tuple specifying the (min, max) range for the y position
        z_range: Tuple specifying the (min, max) range for the z position
        mass: Mass of the masspoint body in kg
        diaginertia: Diagonal inertia value (same for all three axes) in kg*m^2
        kv: Velocity servo gain; set high enough that kv > mu*mass*g to overcome ground friction
        spring_stiffness: Stiffness of the z-spring (0.0 = no spring)
        gravcomp: Whether to enable gravity compensation on the body
    """
    env = Environment(loader=FileSystemLoader(str(ASSETS_DIR)))
    template = env.get_template("masspoint.xml.jinja")

    context = {
        "MP_RADIUS": radius,
        "MP_X_RANGE": x_range,
        "MP_Y_RANGE": y_range,
        "MP_Z_RANGE": z_range,
        "MP_MASS": mass,
        "MP_DIAGINERTIA": diaginertia,
        "MP_KV": kv,
        "SPRING_STIFFNESS": spring_stiffness if spring_stiffness > 0 else None,
        "GRAVCOMP": gravcomp,
    }

    xml_str = template.render(**context)

    out_path = GENERATED_XML_DIR / f"masspoint_{suffix}.xml"
    with open(out_path, "w") as f:
        f.write(xml_str)

def _generate_everything_for_type_task():
    """Generate all XML assets required by the type task from type_constants.py.

    Note that there may be multiple masspoints which all share the same generated XML.

    Side Effects:
        - Writes assets/generated/keyboard.xml.
        - Writes assets/generated/active_key_{i}.xml for each active key slot.
        - Writes assets/generated/masspoint_type_task.xml.
    """
    _generate_keyboard_xml()
    for i in range(type_constants.NUM_ACTIVE_KEYS):
        _generate_marker_xml(f"active_key_{i}", type_constants.ACTIVE_KEY_RGBA, type_constants.MARKER_SIZE)
    _generate_masspoint_xml(
        "type_task",
        type_constants.MP_RADIUS,
        type_constants.MP_X_RANGE,
        type_constants.MP_Y_RANGE,
        type_constants.MP_Z_RANGE,
        mass=type_constants.MP_MASS,
        diaginertia=type_constants.MP_DIAGINERTIA,
        kv=type_constants.MP_KV,
        spring_stiffness=type_constants.MP_SPRING_STIFFNESS,
        gravcomp=False,
    )

def _generate_reach_goal_marker_xml():
    """Generate the reach goal sphere marker XML from the reach_goal Jinja2 template.

    Side Effects:
        - Writes the rendered XML to assets/generated/reach_goal.xml.
    """
    env = Environment(loader=FileSystemLoader(str(ASSETS_DIR)))
    template = env.get_template("reach_goal.xml.jinja")
    context = {
        "name": "reach_goal",
        "rgba": " ".join(map(str, reach_constants.REACH_GOAL_RGBA)),
        "ARM_LEN": reach_constants.MARKER_SIZE[0],
    }
    xml_str = template.render(**context)
    with open(GENERATED_XML_DIR / "reach_goal.xml", "w") as f:
        f.write(xml_str)

def _generate_everything_for_reach_task():
    """Generate all XML assets required by the reach task from reach_constants.py.

    Side Effects:
        - Writes assets/generated/masspoint_reach_task.xml.
        - Writes assets/generated/reach_goal.xml.
    """
    _generate_masspoint_xml(
        "reach_task",
        radius=reach_constants.MP_RADIUS,
        x_range=reach_constants.MP_X_RANGE,
        y_range=reach_constants.MP_Y_RANGE,
        z_range=reach_constants.MP_Z_RANGE,
        mass=reach_constants.MP_MASS,
        diaginertia=reach_constants.MP_DIAGINERTIA,
        kv=reach_constants.MP_KV,
        spring_stiffness=reach_constants.MP_SPRING_STIFFNESS,
        gravcomp=reach_constants.MP_GRAVCOMP,
    )
    _generate_reach_goal_marker_xml()

def _generate_cuboid_xml():
    """Generate the passive 3-DOF cuboid XML (x, y, yaw joints) for the push task.

    Side Effects:
        - Writes the rendered XML to assets/generated/cuboid.xml.
    """
    env = Environment(loader=FileSystemLoader(str(ASSETS_DIR)))
    template = env.get_template("cuboid.xml.jinja")

    x_lo = push_constants.CUBOID_X_RANGE[0]
    x_hi = push_constants.CUBOID_X_RANGE[1]
    y_lo = push_constants.CUBOID_Y_RANGE[0]
    y_hi = push_constants.CUBOID_Y_RANGE[1]

    context = {
        "HX": push_constants.CUBOID_HX,
        "HY": push_constants.CUBOID_HY,
        "HZ": push_constants.CUBOID_HZ,
        "MASS": push_constants.CUBOID_MASS,
        "X_RANGE": (round(x_lo, 3), round(x_hi, 3)),
        "Y_RANGE": (round(y_lo, 3), round(y_hi, 3)),
        "rgba": " ".join(map(str, push_constants.CUBOID_RGBA)),
    }

    xml_str = template.render(**context)
    out_path = GENERATED_XML_DIR / "cuboid.xml"
    with open(out_path, "w") as f:
        f.write(xml_str)


def _generate_push_target_marker_xml():
    """Generate the push target marker XML from the push_target Jinja2 template.

    Side Effects:
        - Writes the rendered XML to assets/generated/push_target.xml.
    """
    env = Environment(loader=FileSystemLoader(str(ASSETS_DIR)))
    template = env.get_template("push_target.xml.jinja")
    context = {
        "name": "push_target",
        "rgba": " ".join(map(str, push_constants.PUSH_TARGET_RGBA)),
        "HX": push_constants.CUBOID_HX,
        "HY": push_constants.CUBOID_HY,
        "HZ": push_constants.CUBOID_HZ,
    }
    xml_str = template.render(**context)
    with open(GENERATED_XML_DIR / "push_target.xml", "w") as f:
        f.write(xml_str)

def _generate_everything_for_push_task():
    """Generate all XML assets required by the push task from push_constants.py.

    Side Effects:
        - Writes assets/generated/cuboid.xml.
        - Writes assets/generated/push_target.xml.
        - Writes assets/generated/masspoint_push_task.xml.
    """
    _generate_cuboid_xml()
    _generate_push_target_marker_xml()
    _generate_masspoint_xml(
        "push_task",
        radius=push_constants.MP_RADIUS,
        x_range=push_constants.MP_X_RANGE,
        y_range=push_constants.MP_Y_RANGE,
        z_range=push_constants.MP_Z_RANGE,
        mass=push_constants.MP_MASS,
        diaginertia=push_constants.MP_DIAGINERTIA,
        kv=push_constants.MP_KV,
        spring_stiffness=push_constants.MP_SPRING_STIFFNESS,
        gravcomp=push_constants.MP_GRAVCOMP,
    )


def generate_all():
    """Generate all XML assets for every task (type, reach, and push).

    Side Effects:
        - Writes all XML files under assets/generated/ by delegating to
          _generate_everything_for_type_task, _generate_everything_for_reach_task,
          and _generate_everything_for_push_task.
    """
    _generate_everything_for_type_task()
    _generate_everything_for_reach_task()
    _generate_everything_for_push_task()

if __name__ == "__main__":
    generate_all()
