#!/usr/bin/env bash
# ==============================================================================
# CS769 DiLoCo Project — SLURM Cluster Batch Submission Script
# ==============================================================================
# Usage:
#   sbatch scripts/slurm_template.sbatch E0
#   sbatch scripts/slurm_template.sbatch E1
#   sbatch scripts/slurm_template.sbatch E2
# ==============================================================================

#SBATCH --job-name=cs769_diloco
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -eo pipefail

# Experiment selection from first positional argument (defaults to E0)
EXPERIMENT="${1:-E0}"

echo "=================================================================="
echo "  SLURM Job ID:      $SLURM_JOB_ID"
echo "  Running on Host:   $(hostname)"
echo "  Allocated GPUs:    $CUDA_VISIBLE_DEVICES"
echo "  Experiment Target: $EXPERIMENT"
echo "  Start Time:        $(date)"
echo "=================================================================="

# Optional: activate cluster conda environment or modules
# module load cuda/12.1
# source activate diloco

mkdir -p logs

# Run the selected experiment across seeds
python3 scripts/run_experiments.py \
    --experiment "$EXPERIMENT" \
    --seeds 0 1 2

echo "=================================================================="
echo "  Job finished at: $(date)"
echo "=================================================================="
