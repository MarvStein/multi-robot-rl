#!/bin/bash
#SBATCH --job-name=reach
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=11:59:00
#SBATCH --array=0-3
#SBATCH --output=scripts/slurm/logs/reach_%A_%a.out
#SBATCH --error=scripts/slurm/logs/reach_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p scripts/slurm/logs

export PATH="$HOME/.local/bin:$PATH"
module load eth_proxy

cd /cluster/project/coros/msteinkel/multi-robot-rl

# Each entry: NUM_MASSPOINTS NUM_UR10S NUM_GOALS label
VARIANTS=(
    "0 2 5 2ur10_5goals_0_005"
    "0 1 5 1ur10_5goals_0_005"
    "2 0 5 2mp_5goals_0_005"
    "1 0 5 1mp_5goals_0_005"
)
SEEDS=(0)
NUM_SEEDS=1

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
    --agent.wandb-project multi-robot-rl-reach-test-alive-penalty \
    --agent.wandb-tags "('reach-sweep','$RUN_NAME')"
