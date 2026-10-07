#!/usr/bin/env bash
# ==============================================================================
# CS769 DiLoCo Project — Server Environment Setup & Verification
# ==============================================================================
# Usage:
#   bash scripts/setup_server.sh
# ==============================================================================

set -eo pipefail

echo "=================================================================="
echo "  CS769 DiLoCo: Initializing Server Environment"
echo "=================================================================="

# 1. Check Python version
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 could not be found. Please install Python 3.10+."
    exit 1
fi
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Found Python $PYTHON_VER"

# 2. Check GPU / CUDA
echo "[2/5] Checking NVIDIA GPU and CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "  WARNING: nvidia-smi not detected. Server might not have NVIDIA drivers attached."
fi

# 3. Virtual environment (optional: activate if exists)
echo "[3/5] Checking environment..."
if [ -n "$VIRTUAL_ENV" ]; then
    echo "  Active virtualenv detected: $VIRTUAL_ENV"
elif [ -n "$CONDA_PREFIX" ]; then
    echo "  Active conda environment detected: $CONDA_PREFIX"
else
    echo "  Notice: No active venv/conda detected. Installing into current user environment."
fi

# 4. Install dependencies
echo "[4/5] Installing core dependencies from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Verify PyTorch CUDA detection
python3 -c "
import torch
print('  PyTorch version:', torch.__version__)
print('  CUDA available:  ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  Device count:    ', torch.cuda.device_count())
    print('  Device name:     ', torch.cuda.get_device_name(0))
"

# 5. Run smoke tests
echo "[5/5] Running test suite to verify algorithmic correctness..."
python3 smoke_test.py

echo "=================================================================="
echo "  [SUCCESS] Server environment is verified and ready for training!"
echo "=================================================================="
echo "Next step: Run an experiment using scripts/run_server.sh or sbatch:"
echo "  bash scripts/run_server.sh --experiment E0"
echo "=================================================================="
