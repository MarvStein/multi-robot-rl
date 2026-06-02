"""Play script that prints camera state on C keypress.

Usage (same args as `uv run play`):
    uv run python scripts/play_cam_debug.py <task-name> --agent random

Press C in the viewer to print current azimuth, elevation, distance, and lookat.
"""

_KEY_C = 67


if __name__ == "__main__":
    from mjlab.viewer.native.viewer import NativeMujocoViewer
    from mjlab.scripts.play import main

    _orig_init = NativeMujocoViewer.__init__

    def _patched_init(self, env, policy, key_callback=None, **kwargs):
        """Wrap NativeMujocoViewer.__init__ to inject a camera-debug key callback.

        Args:
            self: The viewer instance being initialised.
            env: The environment passed to the viewer.
            policy: The policy passed to the viewer.
            key_callback: Optional existing key callback to chain after the camera callback.
            **kwargs: Additional keyword arguments forwarded to the original __init__.

        Side Effects:
            - Replaces ``key_callback`` with a wrapper that prints camera state on C keypress
              before delegating to the original callback.
        """
        def _cam_callback(key):
            """Print camera pose to stdout when C is pressed, then forward to the original callback.

            Args:
                key: GLFW key code of the pressed key.

            Side Effects:
                - Prints azimuth, elevation, distance, and lookat to stdout when key == _KEY_C.
            """
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
    main()
