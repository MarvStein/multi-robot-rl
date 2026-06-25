# Multi-Robot RL

[![Docs Deploy](https://github.com/MarvStein/multi-robot-rl/actions/workflows/docs.yml/badge.svg)](https://github.com/MarvStein/multi-robot-rl/actions/workflows/docs.yml)

A research codebase for multi-robot reinforcement learning in MuJoCo, built on [mjlab](https://mujocolab.github.io/mjlab/main/index.html). It defines three tasks where a variable number of robots (masspoints or UR10e arms) must jointly solve a shared objective, trained with PPO.

## Tasks

| Task | Description |
|------|-------------|
| `reach` | Robots must move their end-effectors to a set of unassigned 3D goal positions. |
| `push` | Robots must push cuboids along a surface to unassigned target poses (XY position + yaw). |
| `type` | Robots must press a sequence of highlighted keys on a physical keyboard. |

## Installation

This project requires [uv](https://docs.astral.sh/uv/), which can be installed with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The following commands clone the repo, install the dependencies and visualize a task with a random policy. Robot counts and other parameters can be changed in the [configs directory](./src/multi_robot_rl/configs/).

```bash
git clone git@github.com:MarvStein/multi-robot-rl.git
cd multi-robot-rl
uv sync
uv run play <reach/push/type> --agent random
```

For more details on training, rollouts, recordings and more checkout the documentation below.

## Documentation

- [Getting Started](https://marvinsteinkellner.ch/multi-robot-rl/getting-started/)
- [API Reference](https://marvinsteinkellner.ch/multi-robot-rl/reference/)
