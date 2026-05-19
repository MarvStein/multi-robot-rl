"""Play script that prints camera state on C keypress.

Usage (same args as `uv run play`):
    uv run python src/multi_robot_rl/scripts/play_cam_debug.py <task-name> --agent random

Press C in the viewer to print current azimuth, elevation, distance, and lookat.
"""

from mjlab.viewer.native.viewer import NativeMujocoViewer

_KEY_C = 67
_orig_init = NativeMujocoViewer.__init__


def _patched_init(self, env, policy, key_callback=None, **kwargs):
    def _cam_callback(key):
        if key == _KEY_C and self.viewer is not None:
            cam = self.viewer.cam
            print(
                f"[CAM] azimuth={cam.azimuth:.1f}  elevation={cam.elevation:.1f}"
                f"  distance={cam.distance:.3f}  lookat={[round(v, 3) for v in cam.lookat]}"
            )
        if key_callback is not None:
            key_callback(key)

    _orig_init(self, env, policy, key_callback=_cam_callback, **kwargs)


NativeMujocoViewer.__init__ = _patched_init

if __name__ == "__main__":
    from mjlab.scripts.play import main
    main()
