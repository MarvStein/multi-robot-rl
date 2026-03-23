# Multi-Robot RL

A repository for multi-robot reinforcement learning environments, built on top of [mjlab](https://mujocolab.github.io/mjlab/main/index.html).

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast Python environment and dependency management.

1. **Install uv** (if you do not have it installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone git@github.com:MarvStein/multi-robot-rl.git
   cd multi-robot-rl
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```
   *This command automatically creates a virtual environment (`.venv`), installs `mjlab` (which brings in MuJoCo, PyTorch, etc.), and installs this project in editable mode.*

## Environments

Currently available tasks:
*   `Mjlab-Masspoint-Reach`: A single 3-DOF point mass navigating to randomly spawned 3D goals using PPO.

## Training

Because the project leverages the `mjlab.tasks` entrypoint in `pyproject.toml`, the training CLI can automatically discover your registered tasks.

To train the `masspoint-reach` environment with default settings, simply run:
```bash
uv run train Mjlab-Masspoint-Reach
```

You can optionally override configuration values via the CLI. For example, to run with more parallel environments:
```bash
uv run train Mjlab-Masspoint-Reach --num-envs 4096
```

By default, tensorboard logs and model checkpoints will be saved in the `logs/rsl_rl/masspoint_reach/` directory.

## Visualizing/Playing

To play back a trained checkpoint from a local file:

```bash
uv run play Mjlab-Masspoint-Reach --checkpoint-file logs/rsl_rl/masspoint_reach/<run_name>/model_<iteration_number>.pt
```
*(Make sure to replace `<run_name>` and `<iteration_number>` with the actual path generated during your training run).*

Alternatively, if you use Weights & Biases (W&B) for tracking experiments, you can stream the trained model directly using the run path:

```bash
uv run play Mjlab-Masspoint-Reach --wandb-run-path <user/project/run_id>
```

You can also do a quick sanity check to spawn the environment and watch it under zero actions (without needing a trained policy):

```bash
uv run play Mjlab-Masspoint-Reach --agent zero
```
