#!/bin/bash
#SBATCH --job-name=genetic_gfn_pmo
#SBATCH --output=logs/genetic_gfn_%A_%a.out
#SBATCH --error=logs/genetic_gfn_%A_%a.err
#SBATCH --array=1-23%8
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# All 23 PMO tasks
ORACLES=(
    albuterol_similarity
    amlodipine_mpo
    celecoxib_rediscovery
    deco_hop
    drd2
    fexofenadine_mpo
    gsk3b
    isomers_c7h8n2o2
    isomers_c9h10n2o2pf2cl
    jnk3
    median1
    median2
    mestranol_similarity
    osimertinib_mpo
    perindopril_mpo
    qed
    ranolazine_mpo
    scaffold_hop
    sitagliptin_mpo
    thiothixene_rediscovery
    troglitazone_rediscovery
    valsartan_smarts
    zaleplon_mpo
)

# Get oracle name for this array task
ORACLE_NAME="${ORACLES[$((SLURM_ARRAY_TASK_ID-1))]}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Oracle: $ORACLE_NAME"
echo "Method: genetic_gfn"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gflownet-pmo

# Change to PMO directory
cd /auto/home/tatevvvv/genetic_gfn/pmo

# ---- MPS for better GPU sharing (A100) ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

# Avoid CPU oversubscription when running 3 python processes
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Where to put outputs (optional; adjust if your run.py already manages it)
BASE_OUT="results/${ORACLE_NAME}"
mkdir -p "$BASE_OUT"

run_one () {
  local seed="$1"
  local outdir="${BASE_OUT}/seed_${seed}"
  mkdir -p "$outdir"

  # Use srun --exclusive so the 3 processes split CPUs cleanly.
  # IMPORTANT: --n_runs 1 to prevent internal sequential loop.
  srun --exclusive -n1 bash -lc "
    python run.py genetic_gfn \
      --task production \
      --n_runs 1 \
      --seed ${seed} \
      --oracles ${ORACLE_NAME} \
      --output_dir ${outdir} \
      --run_name ${ORACLE_NAME}_seed_${seed} \
      --wandb offline
  "
}

# ---- Pack seeds: 3 + 2 waves (fits ~30% GPU each) ----
echo "Wave 1: seeds 0 1 2"
run_one 0 &
run_one 1 &
run_one 2 &
wait

echo "Wave 2: seeds 3 4"
run_one 3 &
run_one 4 &
wait

# ---- Stop MPS ----
echo quit | nvidia-cuda-mps-control
rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "=========================================="
echo "End time: $(date)"
echo "Oracle $ORACLE_NAME completed"
echo "=========================================="
