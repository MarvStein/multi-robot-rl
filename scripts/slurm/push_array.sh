#!/bin/bash
#SBATCH --job-name=push
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:59:00
#SBATCH --array=0-9
#SBATCH --output=scripts/slurm/logs/push_%A_%a.out
#SBATCH --error=scripts/slurm/logs/push_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p scripts/slurm/logs

export PATH="$HOME/.local/bin:$PATH"
module load eth_proxy

cd $HOME/multi-robot-rl

# Each entry: NUM_MASSPOINTS NUM_UR10S NUM_CUBOIDS label
VARIANTS=(
    "2 0 1 2mp_1cube"
    "1 0 1 1mp_1cube"
)
SEEDS=(0 1 2 3 4)
NUM_SEEDS=5

VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

read NUM_MASSPOINTS NUM_UR10S NUM_CUBOIDS LABEL <<< "${VARIANTS[$VARIANT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="ppo_${LABEL}_seed${SEED}"

export NUM_MASSPOINTS NUM_UR10S NUM_CUBOIDS

uv run train push \
    --agent.seed "$SEED" \
    --agent.run-name "$RUN_NAME" \
    --agent.logger wandb \
    --agent.wandb-project multi-robot-rl-euler-push \
    --agent.wandb-tags "('push-sweep','$RUN_NAME')"
