#!/bin/bash
# Quick inference on a feature sequence
# Usage: bash scripts/quick_inference.sh <checkpoint> <features_file> [output_dir]

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/quick_inference.sh <checkpoint> <features_file> [output_dir]"
    echo "Example: bash scripts/quick_inference.sh runs/paraformer_full/checkpoints/best.pt test_features.npy results/"
    exit 1
fi

CHECKPOINT=$1
FEATURES=$2
OUTPUT_DIR=${3:-inference_results}

echo "Running ASR inference"
echo "====================="
echo "Checkpoint: $CHECKPOINT"
echo "Features: $FEATURES"
echo "Output: $OUTPUT_DIR"
echo ""

python scripts/inference.py \
    --checkpoint "$CHECKPOINT" \
    --features "$FEATURES" \
    --output "$OUTPUT_DIR"

echo ""
echo "Inference complete! Results saved to $OUTPUT_DIR/"
