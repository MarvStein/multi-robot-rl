import os
import sys
from jinja2 import Environment, FileSystemLoader

# Import the centralized constants
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import keyboard_constants as kc

def get_template_env():
    # Use FileSystemLoader to load templates from the xmls directory
    # Note: FileSystemLoader instead of FileSystemLoader
    from jinja2 import Environment, FileSystemLoader
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xmls")
    return Environment(loader=FileSystemLoader(templates_dir))

def generate_keyboard_xml():
    env = get_template_env()
    template = env.get_template("keyboard_board.xml.jinja")
    
    xs = [kc.KEY_X_START + i * kc.KEY_X_STEP for i in range(kc.NUM_COLS)]
    ys = [kc.KEY_Y_START + i * kc.KEY_Y_STEP for i in range(kc.NUM_ROWS)]
    
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
            
    # variables to pass to the template
    context = {
        "CENTER_POS": kc.CENTER_POS,
        "KEY_Z_POS_REL": kc.KEY_Z_POS_REL,
        "KEY_JOINT_RANGE": kc.KEY_JOINT_RANGE,
        "KEY_SIZE": kc.KEY_SIZE,
        "keys": keys
    }
    
    xml_str = template.render(**context)
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xmls", "keyboard_board.xml")
    with open(out_path, "w") as f:
        f.write(xml_str)

def generate_marker_xml(name, rgba):
    env = get_template_env()
    template = env.get_template("marker.xml.jinja")
    
    context = {
        "name": name,
        "rgba": rgba,
        "MARKER_SIZE": kc.MARKER_SIZE
    }
    
    xml_str = template.render(**context)
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xmls", f"{name}.xml")
    with open(out_path, "w") as f:
        f.write(xml_str)

def generate_masspoint_xml():
    env = get_template_env()
    template = env.get_template("masspoint_3d.xml.jinja")
    
    context = {
        "MP_RADIUS": kc.MP_RADIUS,
        "MP_XY_RANGE": kc.MP_XY_RANGE,
        "MP_Z_RANGE": kc.MP_Z_RANGE
    }
    
    xml_str = template.render(**context)
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xmls", "masspoint_3d.xml")
    with open(out_path, "w") as f:
        f.write(xml_str)

def generate_markdown_docs():
    # Load from the docs folder
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs")
    env = Environment(loader=FileSystemLoader(docs_dir))
    template = env.get_template("keyboard_dimensions.md.jinja")
    
    xs = [kc.KEY_X_START + i * kc.KEY_X_STEP for i in range(kc.NUM_COLS)]
    ys = [kc.KEY_Y_START + i * kc.KEY_Y_STEP for i in range(kc.NUM_ROWS)]
    
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
        "CENTER_POS": kc.CENTER_POS,
        "KEYBOARD_SIZE": kc.KEYBOARD_SIZE,
        "NUM_COLS": kc.NUM_COLS,
        "NUM_ROWS": kc.NUM_ROWS,
        "KEY_X_START": kc.KEY_X_START,
        "KEY_X_STEP": kc.KEY_X_STEP,
        "KEY_Y_START": kc.KEY_Y_START,
        "KEY_Y_STEP": kc.KEY_Y_STEP,
        "KEY_Z_POS_REL": kc.KEY_Z_POS_REL,
        "KEY_SIZE": kc.KEY_SIZE,
        "KEY_SPACING": kc.KEY_SPACING,
        "KEY_SURFACE_Z": kc.KEY_SURFACE_Z,
        "KEY_PRESS_THRESHOLD": kc.KEY_PRESS_THRESHOLD,
        "KEY_PRESS_THRESHOLD_PCT": kc.KEY_PRESS_THRESHOLD_PCT,
        "TOTAL_KEYS": kc.TOTAL_KEYS,
        "OUT_OF_BOUNDS_MARGIN": kc.OUT_OF_BOUNDS_MARGIN,
        "keys": keys
    }
    
    md_str = template.render(**context)
    
    out_path = os.path.join(docs_dir, "keyboard_dimensions.md")
    with open(out_path, "w") as f:
        f.write(md_str)

def generate_all():
    generate_keyboard_xml()
    generate_markdown_docs()
    # Marker for active key
    generate_marker_xml("active_key", "0.2 0.8 0.2 0.5")
    # Marker for next key
    generate_marker_xml("next_key", "0.8 0.5 0.2 0.5")
    # 3D masspoint agent
    generate_masspoint_xml()

if __name__ == "__main__":
    generate_all()
