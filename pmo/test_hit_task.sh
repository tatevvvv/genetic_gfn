#!/bin/bash
# Test script for hit task - runs a single target with 3000 oracle budget
# Usage: ./test_hit_task.sh [target_name] [seed]
# Example: ./test_hit_task.sh parp1 0

TARGET=${1:-parp1}  # Default to parp1 if not specified
SEED=${2:-0}        # Default to seed 0 if not specified
ORACLE_NAME="${TARGET}_hit_task"

echo "=========================================="
echo "Testing Hit Task"
echo "Target: $TARGET"
echo "Oracle: $ORACLE_NAME"
echo "Seed: $SEED"
echo "Oracle Budget: 3000"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gflownet-pmo

# Change to PMO directory
cd /auto/home/tatevvvv/genetic_gfn/pmo

# Run the test - single run with 3000 oracle budget
python run.py genetic_gfn \
    --task simple \
    --oracles $ORACLE_NAME \
    --max_oracle_calls 3000 \
    --wandb offline \
    --seed $SEED

echo "=========================================="
echo "End time: $(date)"
echo "Test completed for $TARGET (seed $SEED)"
echo "=========================================="

