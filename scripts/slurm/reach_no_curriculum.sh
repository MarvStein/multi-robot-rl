#!/bin/bash
#SBATCH --job-name=reach-no-curriculum
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --array=0-4
#SBATCH --output=scripts/slurm/logs/reach_no_curriculum_%A_%a.out
#SBATCH --error=scripts/slurm/logs/reach_no_curriculum_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p scripts/slurm/logs

export PATH="$HOME/.local/bin:$PATH"
module load eth_proxy

cd /cluster/project/coros/msteinkel/multi-robot-rl

NUM_MASSPOINTS=2
NUM_UR10S=0
NUM_GOALS=5
LABEL="2mp_5goals"
SEEDS=(0 1 2 3 4)

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="ppo_${LABEL}_seed${SEED}"

export NUM_MASSPOINTS NUM_UR10S NUM_GOALS

uv run train reach-no-curriculum \
    --agent.seed "$SEED" \
    --agent.run-name "$RUN_NAME" \
    --agent.logger wandb \
    --agent.wandb-project multi-robot-rl-euler-reach-no-curriculum \
    --agent.wandb-tags "('reach-sweep','$RUN_NAME')"
