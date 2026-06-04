"""
Script to record a video of a trained RL agent (headless, no viewer).

Usage:
uv run record <reach/push/type> --checkpoint-file logs/rsl_rl/<...>.pt --video-length <N>

where N is the number of steps to record. I.e. for 50ms steps, 600 steps corresponds to a 30 second video.

Example:
uv run record reach --checkpoint-file logs/rsl_rl/reach_task/2026-05-19_10-43-18/model_21200.pt --video-length 600
"""
# Note that a recording feature exists in mjlab via play --video True but there's a bug which launches the endless viewer after recording, so i implemented this separate script.


import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


class _NamedVideoRecorder(VideoRecorder):
    """VideoRecorder that saves as `<name_prefix>.mp4` without a step/episode suffix."""

    def _start_recording(self) -> None:
        """Start a new recording session, using only the name prefix as the filename.

        Side Effects:
            - Sets self.is_recording to True.
            - Resets self.current_video_frames to an empty list.
            - Sets self.current_video_path to {video_folder}/{name_prefix}.mp4 (no step suffix).
        """
        self.is_recording = True
        self.current_video_frames = []
        self.current_video_path = self.video_folder / f"{self.name_prefix}.mp4"
        if not self.disable_logger:
            print(f"[INFO] Recording video to {self.current_video_path}")


@dataclass(frozen=True)
class RecordConfig:
    checkpoint_file: str
    """Path to the checkpoint .pt file."""
    video_length: int = 1200
    num_envs: int | None = None
    video_height: int | None = 720
    video_width: int | None = 1280
    device: str | None = None


def run_record(task_id: str, cfg: RecordConfig) -> None:
    """Load a trained agent checkpoint and run headless inference, saving the result to an MP4 file.

    Args:
        task_id: Registered task identifier used to look up env and RL configs.
        cfg: RecordConfig specifying checkpoint path, video length, and optional overrides.

    Side Effects:
        - Creates a videos/ subdirectory inside the checkpoint's log directory.
        - Writes a {checkpoint_stem}.mp4 file to that videos/ directory.
    """
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = _NamedVideoRecorder(
        env,
        video_folder=log_dir / "videos",
        step_trigger=lambda step: step == 0,
        video_length=cfg.video_length,
        name_prefix=resume_path.stem,
        disable_logger=False,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    with torch.no_grad():
        for _ in range(cfg.video_length):
            obs = env.get_observations()
            actions = policy(obs)
            env.step(actions)

    env.close()


def main():
    """CLI entry point: parse task id and RecordConfig from argv, then run recording.

    Side Effects:
        - Calls run_record, which creates a videos/ subdirectory and writes an MP4 file to disk.
    """
    import mjlab.tasks  # noqa: F401
    import multi_robot_rl.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        RecordConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    run_record(chosen_task, args)


if __name__ == "__main__":
    main()
