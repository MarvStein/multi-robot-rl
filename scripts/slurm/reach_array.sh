#!/bin/bash
#SBATCH --job-name=reach
#SBATCH --partition=gpupr.4h
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:59:00
#SBATCH --array=0-19
#SBATCH --output=scripts/slurm/logs/reach_%A_%a.out
#SBATCH --error=scripts/slurm/logs/reach_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p scripts/slurm/logs

export PATH="$HOME/.local/bin:$PATH"
module load eth_proxy

cd /cluster/project/coros/msteinkel/multi-robot-rl

# Each entry: NUM_MASSPOINTS NUM_UR10S NUM_GOALS label
VARIANTS=(
    "0 2 5 2ur10_5goals"
    "0 1 5 1ur10_5goals"
    "2 0 5 2mp_5goals"
    "1 0 5 1mp_5goals"
)
SEEDS=(0 1 2 3 4)
NUM_SEEDS=5

VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

read NUM_MASSPOINTS NUM_UR10S NUM_GOALS LABEL <<< "${VARIANTS[$VARIANT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="ppo_${LABEL}_seed${SEED}"

export NUM_MASSPOINTS NUM_UR10S NUM_GOALS

uv run train reach \
    --agent.seed "$SEED" \
    --agent.run-name "$RUN_NAME" \
    --agent.logger wandb \
    --agent.wandb-project multi-robot-rl-euler-reach-2 \
    --agent.wandb-tags "('reach-sweep','$RUN_NAME')"
