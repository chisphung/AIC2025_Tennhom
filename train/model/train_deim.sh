#!/bin/bash
# =============================================================================
# DEIM Training Script for AI City Challenge 2025 - Fisheye8K Dataset
# Team Tennhom
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Model variant: n (nano), s (small), m (medium), l (large), x (xlarge)
MODEL_SIZE="${MODEL_SIZE:-l}"

# Number of GPUs to use
NUM_GPUS="${NUM_GPUS:-1}"

# Training settings
SEED="${SEED:-0}"
USE_AMP="${USE_AMP:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/deim_fisheye8k_${MODEL_SIZE}}"

# Resume training from checkpoint (leave empty for fresh start)
RESUME="${RESUME:-}"

# Fine-tune from pretrained checkpoint (leave empty for fresh start)
TUNING="${TUNING:-}"

# Master port for distributed training
MASTER_PORT="${MASTER_PORT:-7777}"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEIM_DIR="${SCRIPT_DIR}/DEIM"
CONFIG_FILE="${DEIM_DIR}/configs/deim_dfine/deim_hgnetv2_${MODEL_SIZE}_coco.yml"
DATASET_CONFIG="${DEIM_DIR}/configs/dataset/fisheye8k_detection.yml"

# -----------------------------------------------------------------------------
# Validate configuration
# -----------------------------------------------------------------------------

echo "=============================================="
echo "DEIM Training Script - AI City Challenge 2025"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model Size:     ${MODEL_SIZE}"
echo "  Number of GPUs: ${NUM_GPUS}"
echo "  Seed:           ${SEED}"
echo "  Use AMP:        ${USE_AMP}"
echo "  Output Dir:     ${OUTPUT_DIR}"
echo "  DEIM Dir:       ${DEIM_DIR}"
echo ""

# Check if DEIM directory exists
if [ ! -d "${DEIM_DIR}" ]; then
    echo "Error: DEIM directory not found at ${DEIM_DIR}"
    echo "Please clone DEIM first: git clone https://github.com/Intellindust-AI-Lab/DEIM/"
    exit 1
fi

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    echo "Available model sizes: n, s, m, l, x"
    exit 1
fi

# Check if dataset config exists
if [ ! -f "${DATASET_CONFIG}" ]; then
    echo "Warning: Dataset config not found: ${DATASET_CONFIG}"
    echo "Please create the Fisheye8K dataset configuration file."
fi

# -----------------------------------------------------------------------------
# Setup environment
# -----------------------------------------------------------------------------

cd "${DEIM_DIR}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build training command
TRAIN_CMD="python train.py"
TRAIN_CMD="${TRAIN_CMD} -c ${CONFIG_FILE}"
TRAIN_CMD="${TRAIN_CMD} --output-dir ${OUTPUT_DIR}"
TRAIN_CMD="${TRAIN_CMD} --seed ${SEED}"

if [ "${USE_AMP}" = "true" ]; then
    TRAIN_CMD="${TRAIN_CMD} --use-amp"
fi

if [ -n "${RESUME}" ]; then
    TRAIN_CMD="${TRAIN_CMD} -r ${RESUME}"
fi

if [ -n "${TUNING}" ]; then
    TRAIN_CMD="${TRAIN_CMD} -t ${TUNING}"
fi

# -----------------------------------------------------------------------------
# Run training
# -----------------------------------------------------------------------------

echo "Starting training..."
echo ""

if [ "${NUM_GPUS}" -gt 1 ]; then
    # Multi-GPU training with torchrun
    echo "Running multi-GPU training with ${NUM_GPUS} GPUs..."
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) torchrun \
        --master_port=${MASTER_PORT} \
        --nproc_per_node=${NUM_GPUS} \
        ${TRAIN_CMD}
else
    # Single GPU training
    echo "Running single GPU training..."
    CUDA_VISIBLE_DEVICES=0 ${TRAIN_CMD}
fi

echo ""
echo "=============================================="
echo "Training completed!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=============================================="
