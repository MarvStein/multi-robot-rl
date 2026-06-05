#!/bin/bash
#SBATCH --job-name=type
#SBATCH --partition=gpupr.4h
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-9
#SBATCH --output=scripts/slurm/logs/type_%A_%a.out
#SBATCH --error=scripts/slurm/logs/type_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p scripts/slurm/logs

export PATH="$HOME/.local/bin:$PATH"
module load eth_proxy

cd $HOME/multi-robot-rl

# Each entry: NUM_MASSPOINTS NUM_UR10S NUM_ACTIVE_KEYS label
VARIANTS=(
    "2 0 3 2mp_3ak"
    "1 0 3 1mp_3ak"
)
SEEDS=(0 1 2 3 4)
NUM_SEEDS=5

VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

read NUM_MASSPOINTS NUM_UR10S NUM_ACTIVE_KEYS LABEL <<< "${VARIANTS[$VARIANT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="ppo_${LABEL}_seed${SEED}"

export NUM_MASSPOINTS NUM_UR10S NUM_ACTIVE_KEYS

uv run train type \
    --agent.seed "$SEED" \
    --agent.run-name "$RUN_NAME" \
    --agent.logger wandb \
    --agent.wandb-project multi-robot-rl-euler-type \
    --agent.wandb-tags "('type-sweep','$RUN_NAME')"
