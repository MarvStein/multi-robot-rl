"""Factory functions that return mjlab RL runner configurations for PPO and FastSAC."""
from mjlab.rl import (
  RslRlFastSacAlgorithmCfg,
  RslRlFastSacRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

def ppo_runner_cfg_default() -> RslRlOnPolicyRunnerCfg:
    """Return the default PPO runner configuration shared by all tasks."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=200,
        save_interval=100,
        experiment_name="<CHANGE_ME>",
        actor=RslRlModelCfg(
            hidden_dims=(256, 128),
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
    """Return the PPO runner configuration for the reach task."""
    cfg = ppo_runner_cfg_default()
    cfg.experiment_name = "reach_task"
    cfg.max_iterations = 1 + 25_000 # total env steps = max_iterations * num_steps_per_env * num_envs = 25_000 * 24 * 2048
    return cfg

def ppo_runner_cfg_type_task() -> RslRlOnPolicyRunnerCfg:
    """Return the PPO runner configuration for the type task."""
    cfg = ppo_runner_cfg_default()
    cfg.experiment_name = "type_task"
    cfg.max_iterations = 1 + 25_000 # total env steps = max_iterations * num_steps_per_env * num_envs = 25_000 * 24 * 2048
    return cfg

def ppo_runner_cfg_push_task() -> RslRlOnPolicyRunnerCfg:
    """Return the PPO runner configuration for the push task."""
    cfg = ppo_runner_cfg_default()
    cfg.experiment_name = "push_task"
    cfg.max_iterations = 1 + 25_000 # total env steps = max_iterations * num_steps_per_env * num_envs = 25_000 * 24 * 2048
    return cfg

def fast_sac_runner_cfg_reach_task() -> RslRlFastSacRunnerCfg:
    """Return the FastSAC runner configuration for the reach task."""
    return RslRlFastSacRunnerCfg(
        experiment_name="reach_task",
        max_iterations=1+600_000, # 24 times larger than PPO to get equal total env steps
        save_interval=2400,
        algorithm=RslRlFastSacAlgorithmCfg(
            gamma=0.99,
            batch_size=4096,
            buffer_size=2048,
            num_updates=4,
            compile=False,
            amp=False,
        ),
    )
