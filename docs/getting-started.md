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
- `Mjlab-Masspoint-Keyboard`: Keyboard typing with masspoints. See also [Keyboard Dimensions](keyboard_dimensions.md)


!!! note
    Training or playing the `Mjlab-Masspoint-Keyboard` task automatically executes `src/multi_robot_rl/masspoints/generate_xmls.py` which updates the XMLs and documentation with the parameters in `src/multi_robot_rl/masspoints/keyboards_constants.py`. Note that the online documentation only reflects the current version on the main branch however. To view the most recent version of the documentation locally, run `uv run mkdocs serve`.

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

### MjLab-Masspoint-Keyboard

```bash
uv run train Mjlab-Masspoint-Keyboard
```

### Outputs

By default, logs and model checkpoints are written under:

```text
logs/rsl_rl/masspoint_reach/
logs/rsl_rl/masspoint_multi_reach/
logs/rsl_rl/masspoint_keyboard/
```

## Playing And Evaluation

!!! note
    In all of the commands below (`uv run play <...>`) you can add `--agent random` or `--agent zero` and/or `--no-terminations True`to debug the environments. You can also double click an entity in the viewer and inject forces with `RMB` while holding `CTRL`.

### MjLab-Masspoint-Reach

```bash
uv run play Mjlab-Masspoint-Reach --checkpoint-file logs/rsl_rl/masspoint_reach/<run_name>/model_<iteration_number>.pt
```

### Mjlab-Masspoint-MultiReach

```bash
uv run play Mjlab-Masspoint-MultiReach --num-masspoints <N> --num-goals <M> --checkpoint-file logs/rsl_rl/masspoint_multi_reach/<run_name>/model_<iteration_number>.pt
```

Replace `<run_name>` and `<iteration_number>` with values from your training output and `N`, `M` with the same values used during training.

### MjLab-Masspoint-Keyboard

```bash
uv run play Mjlab-Masspoint-Keyboard --checkpoint-file logs/rsl_rl/masspoint_keyboard/<run_name>/model_<iteration_number>.pt
```