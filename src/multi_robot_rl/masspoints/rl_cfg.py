"""RL configurations for the masspoint task."""
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

def masspoint_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Runner configuration for PPO."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=200,
        save_interval=50,
        experiment_name="masspoint_reach",
        actor=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


def masspoint_multi_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Runner configuration for PPO in the configurable multi-reach task."""
    cfg = masspoint_ppo_runner_cfg()
    cfg.experiment_name = "masspoint_multi_reach"
    return cfg
