"""Environment configuration for the reach-dense task.

Delegates to make_reach_env and applies two targeted changes:
  - Adds the PBRS dense proximity reward alongside the sparse goal_reached_reward.
  - Replaces the reset function to also clear the cached PBRS potential.
"""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from multi_robot_rl.tasks.reach.env_cfg import make_reach_env
from multi_robot_rl.tasks.reach_dense import mdp as reach_dense_mdp


def make_reach_dense_env(play: bool = False, no_curriculum: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the reach-dense environment config by extending the sparse reach config.

    Args:
        play: Passed through to make_reach_env for single-env interactive mode.
        no_curriculum: When True, disables the goal spawn curriculum and spawns
            goals at full workspace radius from the start.

    Returns:
        ManagerBasedRlEnvCfg with PBRS dense proximity reward added to the sparse
        goal_reached_reward, with or without curriculum.
    """
    cfg = make_reach_env(play=play, no_curriculum=no_curriculum)

    robots = cfg.rewards["goal_reached"].params["robots"]

    cfg.rewards["goal_proximity"] = RewardTermCfg(
        func=reach_dense_mdp.pbrs_dense_reward,
        params={"robots": robots},
        weight=1.0,
    )

    # Replace the reset function so it also clears the cached PBRS potential.
    cfg.events["reset_goals"].func = reach_dense_mdp.reset_goal_state_dense

    return cfg
