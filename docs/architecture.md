# Architecture

The `multi-robot-rl` repository utilizes a highly modular structure to decouple the functional elements of MDPs (Markov Decision Processes) from standard configuration wrappers.

## Files & Architecture

- `src/multi_robot_rl/`
    - `masspoints/` - The base folder for the tasks.
        - `env_cfgs.py` - Core configuration manager that links observations, rewards, and events into `ManagerBasedRlEnvCfg`.
        - `rl_cfg.py` - Reinforcement Learning (PPO) configuration defining network architecture and hyperparameters.
        - `assets.py` - Definitions, articulation information, and XML loading.
        - `keyboard_constants.py` - Contains constants that define the keyboard layout.
        - `generate_xmls.py` - Jinja2 generator that uses `keyboard_constants.py` to generate XMLs and updates [Keyboard Dimensions](keyboard_dimensions.md).
        - `xmls/` - Contains raw Jinja template files like `keyboard_board.xml.jinja` and (gitignored) populated XMLs
        - `mdp/` - Module containing the environments logic.
            - `observations.py`
            - `rewards.py`
            - `events.py`
            - `terminations.py`
