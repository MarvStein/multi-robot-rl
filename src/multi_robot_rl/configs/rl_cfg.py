"""RL runner configurations."""
from mjlab.rl import (
  RslRlFastSacAlgorithmCfg,
  RslRlFastSacRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

def ppo_runner_cfg_default() -> RslRlOnPolicyRunnerCfg:
    """Default runner configuration for PPO"""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=200,
        save_interval=50,
        experiment_name="<CHANGE_ME>",
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
            hidden_dims=(256, 128),
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

def ppo_runner_cfg_reach_task() -> RslRlOnPolicyRunnerCfg:
    """Runner configuration for PPO in the reach task."""
    cfg = ppo_runner_cfg_default()
    cfg.experiment_name = "reach_task"
    cfg.max_iterations = 1 + 100_000
    return cfg

def ppo_runner_cfg_type_task() -> RslRlOnPolicyRunnerCfg:
    """Runner configuration for PPO in the type task."""
    cfg = ppo_runner_cfg_default()
    cfg.experiment_name = "type_task"
    cfg.max_iterations = 3901
    return cfg

def ppo_runner_cfg_push_task() -> RslRlOnPolicyRunnerCfg:
    """Runner configuration for PPO in the push task."""
    cfg = ppo_runner_cfg_default()
    cfg.experiment_name = "push_task"
    cfg.max_iterations = 3901
    return cfg

def fast_sac_runner_cfg_reach_task() -> RslRlFastSacRunnerCfg:
    """Runner configuration for FastSAC in the reach task."""
    return RslRlFastSacRunnerCfg(
        experiment_name="reach_task_fast_sac",
        max_iterations=500_000,
        save_interval=1200,
        algorithm=RslRlFastSacAlgorithmCfg(
            gamma=0.99,
            batch_size=4096,
            buffer_size=2048,
            num_updates=4,
            compile=False,
            amp=False,
        ),
    )
