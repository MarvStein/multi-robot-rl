# Getting Started

This guide covers local setup and the first environment run.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

Clone the repository and install dependencies:

```bash
git clone git@github.com:MarvStein/multi-robot-rl.git
cd multi-robot-rl
uv sync
```

This creates `.venv`, installs runtime dependencies (including `mjlab` and transitive dependencies), and installs this project in editable mode.

## Smoke Test

```bash
uv run play reach --agent random
```

## Available Tasks

- `reach`: End effectors should reach unassigned goal poses.
- `push`: Cuboids should me pushed along the surface to goal poses (x/y position + yaw).
- `type`: Keyboard typing.

## Training

Because this project uses the `mjlab.tasks` entry point in `pyproject.toml`, the training CLI can discover registered tasks automatically.
To train a task, run the following:

```bash
uv run train <reach/push/type>
```

Robot counts and task complexity can be overridden via environment variables without modifying any files:

```bash
NUM_MASSPOINTS=2 NUM_GOALS=5 uv run train reach
```

### Outputs

By default, logs and model checkpoints are written under `logs/rsl_rl`

## Playing And Evaluation

!!! note
    In all of the commands below (`uv run play <...>`) you can add `--agent random` or `--agent zero` and/or `--no-terminations True`to debug the environments. You can also double click an entity in the viewer and inject forces with `RMB` while holding `CTRL`.

To evaluate a trained model run the following:

```bash
uv run play <reach/push/type> --checkpoint-file logs/rsl_rl/<...>.pt
```

Don't forget to set the env vars to what they were during training, e.g.

```bash
NUM_MASSPOINTS=2 NUM_GOALS=5 uv run play <reach/push/type> --checkpoint-file logs/rsl_rl/<...>.pt
```

## Recording Videos

```bash
uv run record <reach/push/type> --checkpoint-file logs/rsl_rl/<...>.pt --video-length <N>
```
where `N` is the number of steps to record. I.e. for 50ms steps, 600 steps corresponds to a 30 second video.

!!! note
    A recording feature exists in mjlab ([see docs](https://mujocolab.github.io/mjlab/main/source/viewers.html)) but there's a bug which causes the GUI to be launched after recording, so i implemented the separate script `record.py`
