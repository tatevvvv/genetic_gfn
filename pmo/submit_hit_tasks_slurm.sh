#!/bin/bash
#SBATCH --job-name=hit_tasks
#SBATCH --output=logs/hit_task_%A_%a.out
#SBATCH --error=logs/hit_task_%A_%a.err
#SBATCH --array=1-5
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# All 5 targets for hit task
TARGETS=(
    'parp1'
    'fa7'
    '5ht1b'
    'braf'
    'jak2'
)

# Get target name for this array task
TARGET_NAME=${TARGETS[$((SLURM_ARRAY_TASK_ID-1))]}
ORACLE_NAME="${TARGET_NAME}_hit_task"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Target: $TARGET_NAME"
echo "Oracle: $ORACLE_NAME"
echo "Method: genetic_gfn"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gflownet-pmo

# Change to PMO directory
cd /auto/home/tatevvvv/genetic_gfn/pmo

# Run the experiment - each target runs 5 runs with oracle budget 3000
python run.py genetic_gfn \
    --task production \
    --n_runs 5 \
    --oracles $ORACLE_NAME \
    --max_oracle_calls 3000 \
    --wandb offline

echo "=========================================="
echo "End time: $(date)"
echo "Target $TARGET_NAME completed"
echo "=========================================="

