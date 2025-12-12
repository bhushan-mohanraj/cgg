#!/usr/bin/env bash
# scripts/test.sh
# Test script to generate samples from trained models.
# 
# Usage:
#   ./scripts/test.sh                    # Test all new generator models
#   ./scripts/test.sh vae_gen            # Test only vae_gen
#   ./scripts/test.sh autoreg diff_gen   # Test multiple models
#
# Available models (new shared encoder architecture):
#   vae_gen, diff_gen, autoreg
#
# Output format: results/outputs/{sample_index}-{model_name_or_original}.png

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Sample indices to test (space-separated)
SAMPLES=(0 1 2)

# Max nodes for generation (should be <= training max_nodes)
MAX_NODES=32

# Sampling temperature (higher = more random)
TEMPERATURE=1.0

# Device
DEVICE="cpu"  # change to "cuda" if you have a GPU

# Checkpoint directory (must match train.sh)
CHECKPOINT_DIR="results/checkpoints"

# Output directory for generated graphs
OUTPUT_DIR="results/outputs"
mkdir -p "${OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Parse command line arguments
# ─────────────────────────────────────────────────────────────────────────────
# If no arguments provided, test all generator models
if [ $# -eq 0 ]; then
    MODELS_TO_TEST=("vae_gen" "diff_gen" "autoreg")
else
    MODELS_TO_TEST=("$@")
fi

echo "Models to test: ${MODELS_TO_TEST[*]}"
echo "Samples to test: ${SAMPLES[*]}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Function to check if a model should be tested
should_test() {
    local model=$1
    for m in "${MODELS_TO_TEST[@]}"; do
        if [ "$m" == "$model" ]; then
            return 0
        fi
    done
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Test generator models
# ─────────────────────────────────────────────────────────────────────────────

test_generator() {
    local model_name=$1
    local ckpt_file="${CHECKPOINT_DIR}/${model_name}.pt"
    
    if [ -f "${ckpt_file}" ]; then
        echo "=== Testing ${model_name} ==="
        
        for sample in "${SAMPLES[@]}"; do
            echo "  Sample ${sample}..."
            python -m cgg.generate_sample \
                --checkpoint "${ckpt_file}" \
                --sample "${sample}" \
                --use-test-set \
                --max-nodes "${MAX_NODES}" \
                --temperature "${TEMPERATURE}" \
                --device "${DEVICE}" \
                --out-dir "${OUTPUT_DIR}"
        done
        
        echo ""
    else
        echo "Skipping ${model_name}: checkpoint not found at ${ckpt_file}"
        echo ""
    fi
}

# Test each selected model
if should_test "vae_gen"; then
    test_generator "vae_gen"
fi

if should_test "diff_gen"; then
    test_generator "diff_gen"
fi

if should_test "autoreg"; then
    test_generator "autoreg"
fi

echo "=== All testing complete ==="
echo "Output files in ${OUTPUT_DIR}/"
echo ""
echo "Expected output files for each sample:"
echo "  {sample_index}-original.png    (ground truth graph)"
echo "  {sample_index}-{model}.png     (generated graph)"
