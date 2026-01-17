#!/bin/bash

# Script to submit individual SLURM jobs for each oracle
# This creates 23 separate jobs (one per oracle)
# Each job runs 5 production runs

METHOD="gflownet"
N_RUNS=5
WANDB_MODE="offline"
MAX_ORACLE_CALLS=10000

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

# Create logs directory
mkdir -p logs
mkdir -p slurm_scripts

echo "Creating SLURM scripts for ${#ORACLES[@]} oracles..."

# Create a SLURM script for each oracle
for i in "${!ORACLES[@]}"; do
    ORACLE_NAME=${ORACLES[$i]}
    SCRIPT_NAME="slurm_scripts/gflownet_${ORACLE_NAME}.sh"
    
    cat > $SCRIPT_NAME << EOF
#!/bin/bash
#SBATCH --job-name=gfn_${ORACLE_NAME}
#SBATCH --output=logs/gflownet_${ORACLE_NAME}_%j.out
#SBATCH --error=logs/gflownet_${ORACLE_NAME}_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Oracle: ${ORACLE_NAME}"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gflownet

# Change to PMO directory
cd /auto/home/tatevvvv/genetic_gfn/pmo

# Run the experiment
python run.py ${METHOD} \\
    --task production \\
    --n_runs ${N_RUNS} \\
    --oracles ${ORACLE_NAME} \\
    --wandb ${WANDB_MODE} \\
    --max_oracle_calls ${MAX_ORACLE_CALLS}

echo "=========================================="
echo "End time: \$(date)"
echo "Oracle ${ORACLE_NAME} completed"
echo "=========================================="
EOF

    chmod +x $SCRIPT_NAME
    echo "Created: $SCRIPT_NAME"
done

echo ""
echo "=========================================="
echo "All SLURM scripts created!"
echo "To submit all jobs, run:"
echo "  for script in slurm_scripts/gflownet_*.sh; do sbatch \$script; done"
echo ""
echo "Or submit individually:"
echo "  sbatch slurm_scripts/gflownet_<oracle_name>.sh"
echo "=========================================="

