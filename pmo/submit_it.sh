#!/bin/bash
# Script to submit SLURM jobs for all oracles

echo "Submitting SLURM jobs for all oracles..."
echo "This will submit 23 array jobs (one per oracle)"
echo ""

# Submit the SLURM job
sbatch submit_oracles_slurm.sh

echo ""
echo "Job submitted! Check status with: squeue -u $USER"
echo "Monitor logs in: logs/"

