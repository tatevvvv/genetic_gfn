#!/bin/bash
#SBATCH --job-name=gflownet_pmo
#SBATCH --output=logs/gflownet_%A_%a.out
#SBATCH --error=logs/gflownet_%A_%a.err
#SBATCH --array=1-23
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# All 23 PMO tasks
ORACLES=(
    'albuterol_similarity'
    'amlodipine_mpo'
    'celecoxib_rediscovery'
    'deco_hop'
    'drd2'
    'fexofenadine_mpo'
    'gsk3b'
    'isomers_c7h8n2o2'
    'isomers_c9h10n2o2pf2cl'
    'jnk3'
    'median1'
    'median2'
    'mestranol_similarity'
    'osimertinib_mpo'
    'perindopril_mpo'
    'qed'
    'ranolazine_mpo'
    'scaffold_hop'
    'sitagliptin_mpo'
    'thiothixene_rediscovery'
    'troglitazone_rediscovery'
    'valsartan_smarts'
    'zaleplon_mpo'
)

# Get oracle name for this array task
ORACLE_NAME=${ORACLES[$((SLURM_ARRAY_TASK_ID-1))]}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Oracle: $ORACLE_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gflownet

# Change to PMO directory
cd /auto/home/tatevvvv/genetic_gfn/pmo

# Run the experiment
python run.py gflownet \
    --task production \
    --n_runs 5 \
    --oracles $ORACLE_NAME \
    --wandb offline

echo "=========================================="
echo "End time: $(date)"
echo "Oracle $ORACLE_NAME completed"
echo "=========================================="

