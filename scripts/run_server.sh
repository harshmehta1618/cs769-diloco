#!/usr/bin/env bash
# ==============================================================================
# CS769 DiLoCo Project — Server Experiment Runner
# ==============================================================================
# Usage:
#   bash scripts/run_server.sh --experiment E0
#   bash scripts/run_server.sh --experiment E2 --seeds 0 1 2 --gpus 0,1,2,3
#   bash scripts/run_server.sh --experiment E2 --dry_run
# ==============================================================================

set -eo pipefail

EXPERIMENT="E0"
GPUS=""
DRY_RUN=""
SEEDS="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --seeds)
            shift
            SEEDS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SEEDS="$SEEDS $1"
                shift
            done
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Setup logging directory
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/run_${EXPERIMENT}_${TIMESTAMP}.log"

# Device selection
if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    echo "Assigned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo "=================================================================="
echo "  CS769 DiLoCo Server Execution"
echo "=================================================================="
echo "  Experiment: $EXPERIMENT"
echo "  Seeds:      $SEEDS"
echo "  Dry-run:    ${DRY_RUN:-no}"
echo "  Git commit: $GIT_SHA"
echo "  Log output: $LOGFILE"
echo "=================================================================="

# Execute experiment runner with tee to logfile
python3 scripts/run_experiments.py \
    --experiment "$EXPERIMENT" \
    --seeds $SEEDS \
    $DRY_RUN \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOGFILE"

echo "=================================================================="
echo "  Experiment $EXPERIMENT completed! Log saved -> $LOGFILE"
echo "=================================================================="
