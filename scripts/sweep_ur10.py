"""Sweep all UR10e joints through [-1, +1] one at a time to see the full range of motion.

Run with:
    uv run python scripts/sweep_ur10.py

Each joint traces a triangle wave (home-1 → home+1 → home-1) while all other
joints hold at home. The raw action is scaled by 3.0 inside the env, so
action=±1 maps to ±3 rad relative to the home position.

End-effector Z positions are printed to the console every PRINT_EVERY steps so
you can observe the reachable Z range and tune EE_Z_MIN / EE_Z_MAX in
type_constants.py accordingly.
"""

import os

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import multi_robot_rl.tasks.type  # noqa: F401 — registers tasks
from multi_robot_rl.configs.type_constants import EE_Z_MIN, EE_Z_MAX, NUM_UR10S
from multi_robot_rl.tasks.type.env_cfg import make_type_env

PRINT_EVERY = 10  # print EE position every N policy steps

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)

# env step_dt = timestep * decimation = 0.01 * 5 = 0.05 s
# 60 steps/direction × 0.05 s = 3 s per sweep direction → 6 s per joint
STEPS_PER_HALF = 60


class JointSweepPolicy:
    """Sweeps each joint through [-1, +1] one at a time (triangle wave).

    All other joints hold at 0 (home pose offset).
    """

    def __init__(self, device: str = "cpu", steps_per_half: int = STEPS_PER_HALF):
        self._device = device
        self._sph = steps_per_half
        self._steps_per_joint = 2 * steps_per_half  # full -1 → +1 → -1 cycle
        self._step = 0
        self._cur_joint = -1

    def __call__(self, obs: object) -> torch.Tensor:
        actions = torch.zeros(1, len(JOINT_NAMES), device=self._device)

        cycle_len = len(JOINT_NAMES) * self._steps_per_joint
        t = self._step % cycle_len
        joint_idx = t // self._steps_per_joint
        local = t % self._steps_per_joint

        if joint_idx != self._cur_joint:
            self._cur_joint = joint_idx
            print(f"[sweep] joint {joint_idx + 1}/6 — {JOINT_NAMES[joint_idx]}")

        # Triangle wave: -1 → +1 (first half), +1 → -1 (second half)
        if local < self._sph:
            val = -1.0 + 2.0 * local / self._sph
        else:
            val = 1.0 - 2.0 * (local - self._sph) / self._sph

        actions[0, joint_idx] = val
        self._step += 1
        return actions


class EEPrintPolicy:
    """Wraps a policy and prints end-effector XYZ (env 0) every PRINT_EVERY steps.

    Reads site_pos_w from the previous step's state (one-step lag is fine for
    range exploration). Tracks and displays the running Z min/max alongside the
    current EE_Z_MIN / EE_Z_MAX thresholds so you can tune them interactively.
    """

    def __init__(self, inner, base_env: ManagerBasedRlEnv, num_robots: int, print_every: int = PRINT_EVERY):
        self._inner = inner
        self._env = base_env
        self._num_robots = num_robots
        self._print_every = print_every
        self._step = 0
        self._z_min = float("inf")
        self._z_max = float("-inf")
        self._cfgs: list[SceneEntityCfg] | None = None

    def _get_cfgs(self) -> list[SceneEntityCfg]:
        if self._cfgs is None:
            cfgs = [SceneEntityCfg(f"ur10_{i}", site_names=("attachment_site",)) for i in range(self._num_robots)]
            for cfg in cfgs:
                cfg.resolve(self._env.scene)
            self._cfgs = cfgs
        return self._cfgs

    def __call__(self, obs: object) -> torch.Tensor:
        if self._step % self._print_every == 0:
            parts = []
            for cfg in self._get_cfgs():
                pos = self._env.scene[cfg.name].data.site_pos_w[0, cfg.site_ids, :].squeeze(0)
                z = pos[2].item()
                self._z_min = min(self._z_min, z)
                self._z_max = max(self._z_max, z)
                parts.append(f"{cfg.name} xyz=({pos[0]:.3f}, {pos[1]:.3f}, {z:.3f})")
            print(
                f"[EE step {self._step:5d}]  "
                + "  ".join(parts)
                + f"  |  z_seen=[{self._z_min:.3f}, {self._z_max:.3f}]"
                + f"  limits=[{EE_Z_MIN}, {EE_Z_MAX}]"
            )
        self._step += 1
        return self._inner(obs)

    def reset(self, *args: object, **kwargs: object) -> object:
        if hasattr(self._inner, "reset"):
            return self._inner.reset(*args, **kwargs)  # type: ignore[return-value]
        return None


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = make_type_env(play=True)
    # Disable terminations so the sweep never gets interrupted by episode limits
    # or the out-of-bounds check (arm will leave the keyboard area during the sweep).
    env_cfg.terminations = {}

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    sweep = JointSweepPolicy(device=str(device), steps_per_half=STEPS_PER_HALF)
    policy = EEPrintPolicy(sweep, base_env=env.unwrapped, num_robots=NUM_UR10S)

    step_dt = 0.01 * 5  # timestep * decimation
    print(f"UR10e joint sweep  |  EE_Z_MIN={EE_Z_MIN}  EE_Z_MAX={EE_Z_MAX}")
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer_cls = NativeMujocoViewer if has_display else ViserPlayViewer
    viewer_cls(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
