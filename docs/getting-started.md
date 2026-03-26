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
uv run play Mjlab-Masspoint-Reach --agent zero
```

You should see a (stationary) masspoint and goal.

## Available Environments

- `Mjlab-Masspoint-Reach`: A single 2-DoF masspoint navigating to randomly spawned 2D goals.
- `Mjlab-Masspoint-MultiReach`: Multiple masspoints controlled by one policy with multiple unordered goals and delayed goal respawn.

## Training

Because this project uses the `mjlab.tasks` entry point in `pyproject.toml`, the training CLI can discover registered tasks automatically.

### MjLab-Masspoint-Reach

```bash
uv run train Mjlab-Masspoint-Reach
```

### Mjlab-Masspoint-MultiReach

```bash
uv run train Mjlab-Masspoint-MultiReach --num-masspoints <N> --num-goals <M>
```
`--num-masspoints` and `--num-goals` are required and $M \stackrel{!}{\geq} N$.

!!! note
    You can also override other arguments provided by `mjlab` (e.g. `--agent.max-iterations`).
    Use `uv run train Mjlab-Masspoint-MultiReach --help` to see all available options.
    `--num-masspoints` and `--num-goals` do **not** appear in the help message because they are custom flags implemented in `cli_shims.py`.

### Outputs

By default, logs and model checkpoints are written under:

```text
logs/rsl_rl/masspoint_reach/
```

## Playing And Evaluation

### MjLab-Masspoint-Reach

```bash
uv run play Mjlab-Masspoint-Reach --checkpoint-file logs/rsl_rl/masspoint_reach/<run_name>/model_<iteration_number>.pt
```

### Mjlab-Masspoint-MultiReach

```bash
uv run play Mjlab-Masspoint-MultiReach --num-masspoints <N> --num-goals <M> --checkpoint-file logs/rsl_rl/masspoint_multi_reach/<run_name>/model_<iteration_number>.pt
```

Replace `<run_name>` and `<iteration_number>` with values from your training output and `N`, `M` with the same values used during training.

