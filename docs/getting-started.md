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

## First Run

Run the environment with a zero-action agent:

```bash
uv run play Mjlab-Masspoint-Reach --agent zero
```

## Available Environment

- `Mjlab-Masspoint-Reach`: A single 3-DOF point mass navigating to randomly spawned 3D goals.

## Training

Because this project uses the `mjlab.tasks` entry point in `pyproject.toml`, the training CLI can discover registered tasks automatically.

### Default Training Run

```bash
uv run train Mjlab-Masspoint-Reach
```

### Override Configuration From CLI

Example with more parallel environments:

```bash
uv run train Mjlab-Masspoint-Reach --num-envs 4096
```

### Outputs

By default, logs and model checkpoints are written under:

```text
logs/rsl_rl/masspoint_reach/
```

## Playing And Evaluation

### Play From A Local Checkpoint

```bash
uv run play Mjlab-Masspoint-Reach --checkpoint-file logs/rsl_rl/masspoint_reach/<run_name>/model_<iteration_number>.pt
```

Replace `<run_name>` and `<iteration_number>` with values from your training output.

### Play From Weights & Biases

```bash
uv run play Mjlab-Masspoint-Reach --wandb-run-path <user/project/run_id>
```

### Sanity Check Without A Trained Policy

```bash
uv run play Mjlab-Masspoint-Reach --agent zero
```
