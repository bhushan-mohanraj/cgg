#!/usr/bin/env bash
# scripts/train.sh
# Training script for code graph generation models.
# 
# Usage:
#   ./scripts/train.sh                    # Train all models
#   ./scripts/train.sh cgvae hier         # Train only cgvae and hier
#   ./scripts/train.sh vae_gen diff_gen   # Train only new generator models
#
# Available models:
#   Original (backwards compatible): cgvae, hier, diff
#   New (shared encoder): vae_gen, diff_gen, autoreg

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters (shared across models unless overridden)
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH="data"
EPOCHS=32
BATCH_SIZE=16
MAX_NODES=64
MAX_SAMPLES=400      # Max samples to use (before train-test split)
TEST_SPLIT=0.5       # Fraction held out for testing (50-50 split)
LR=1e-3
DEVICE="cpu"  # change to "cuda" if you have a GPU

# Output directory for checkpoints
CHECKPOINT_DIR="results/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Parse command line arguments
# ─────────────────────────────────────────────────────────────────────────────
# If no arguments provided, train all models
if [ $# -eq 0 ]; then
    MODELS_TO_TRAIN=("cgvae" "hier" "diff" "vae_gen" "diff_gen" "autoreg")
else
    MODELS_TO_TRAIN=("$@")
fi

echo "Models to train: ${MODELS_TO_TRAIN[*]}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Helper function to train a model
# ─────────────────────────────────────────────────────────────────────────────
train_model() {
    local model_name=$1
    local output_file="${CHECKPOINT_DIR}/${model_name}.pt"
    
    echo "=== Training ${model_name} ==="
    python -m cgg.train \
        --model "${model_name}" \
        --data-path "${DATA_PATH}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --max-nodes "${MAX_NODES}" \
        --max-samples "${MAX_SAMPLES}" \
        --test-split "${TEST_SPLIT}" \
        --lr "${LR}" \
        --device "${DEVICE}" \
        --out "${output_file}"
    echo "Saved ${model_name} checkpoint to ${output_file}"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Training runs (conditionally based on arguments)
# ─────────────────────────────────────────────────────────────────────────────

# Function to check if a model should be trained
should_train() {
    local model=$1
    for m in "${MODELS_TO_TRAIN[@]}"; do
        if [ "$m" == "$model" ]; then
            return 0
        fi
    done
    return 1
}

# Original models
if should_train "cgvae"; then
    train_model "cgvae"
fi

if should_train "hier"; then
    train_model "hier"
fi

if should_train "diff"; then
    train_model "diff"
fi

# New encoder+generator models
if should_train "vae_gen"; then
    train_model "vae_gen"
fi

if should_train "diff_gen"; then
    train_model "diff_gen"
fi

if should_train "autoreg"; then
    train_model "autoreg"
fi

echo "=== All training runs complete ==="
echo "Checkpoints saved in ${CHECKPOINT_DIR}/"
