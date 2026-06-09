"""Environment configuration for the reach-dense task.

Delegates to make_reach_env and applies two targeted changes:
  - Swaps the sparse goal_reached_reward for the dense pbrs_dense_reward.
  - Uses no_curriculum=True so goals always spawn at full workspace from episode 1.
"""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from multi_robot_rl.tasks.reach.env_cfg import make_reach_env
from multi_robot_rl.tasks.reach_dense import mdp as reach_dense_mdp


def make_reach_dense_env(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build the reach-dense environment config by extending the sparse reach config.

    Args:
        play: Passed through to make_reach_env for single-env interactive mode.

    Returns:
        ManagerBasedRlEnvCfg with dense exponential proximity reward and no curriculum.
    """
    cfg = make_reach_env(play=play, no_curriculum=True)

    robots = cfg.rewards["goal_reached"].params["robots"]

    cfg.rewards["goal_proximity"] = RewardTermCfg(
        func=reach_dense_mdp.pbrs_dense_reward,
        params={"robots": robots},
        weight=1.0,
    )

    # Replace the reset function so it also clears the cached PBRS potential.
    cfg.events["reset_goals"].func = reach_dense_mdp.reset_goal_state_dense

    return cfg
