"""Capture the initial rendered frame of a task environment and save to PNG.

Usage (from repo root):
    NUM_MASSPOINTS=0 NUM_UR10S=2 NUM_GOALS=5  uv run scripts/capture_frame.py reach
    NUM_MASSPOINTS=4 NUM_UR10S=1 NUM_CUBOIDS=2 uv run scripts/capture_frame.py push
    NUM_MASSPOINTS=2 NUM_UR10S=1 NUM_ACTIVE_KEYS=3 uv run scripts/capture_frame.py type

Output is saved to figures/frame_<task>_<variant>.png.
"""

import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("MUJOCO_GL", "egl")

import multi_robot_rl.tasks  # noqa: F401 — populates the mjlab task registry

from mjlab.envs import ManagerBasedRlEnv
from multi_robot_rl.tasks.reach.env_cfg import make_reach_env
from multi_robot_rl.tasks.push.env_cfg import make_push_env
from multi_robot_rl.tasks.type.env_cfg import make_type_env

FIGURES_DIR = Path(__file__).parent.parent / "figures"

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("reach", "push", "type"):
        sys.exit("Usage: uv run scripts/capture_frame.py <reach|push|type>")

    task = sys.argv[1]

    factories = {"reach": make_reach_env, "push": make_push_env, "type": make_type_env}
    env_cfg = factories[task](play=True, no_curriculum=True)
    env_cfg.viewer.width = 1920
    env_cfg.viewer.height = 1080
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
    env.reset()
    env.step(torch.zeros(1, env.single_action_space.shape[0], device=env.device))
    frame = env.render()
    env.close()

    if frame is None:
        sys.exit("render() returned None — check that MUJOCO_GL=egl and a GPU is available.")

    from PIL import Image
    FIGURES_DIR.mkdir(exist_ok=True)
    out = FIGURES_DIR / f"{task}_env.png"
    Image.fromarray(frame).save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
