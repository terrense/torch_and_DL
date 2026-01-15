#!/bin/bash
# Quick inference on a single image
# Usage: bash scripts/quick_inference.sh <checkpoint> <image_path> [output_dir]

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/quick_inference.sh <checkpoint> <image_path> [output_dir]"
    echo "Example: bash scripts/quick_inference.sh runs/unet_baseline/checkpoints/best.pt test_image.png results/"
    exit 1
fi

CHECKPOINT=$1
IMAGE=$2
OUTPUT_DIR=${3:-inference_results}

echo "Running inference"
echo "================="
echo "Checkpoint: $CHECKPOINT"
echo "Image: $IMAGE"
echo "Output: $OUTPUT_DIR"
echo ""

python scripts/inference.py \
    --checkpoint "$CHECKPOINT" \
    --image "$IMAGE" \
    --output "$OUTPUT_DIR"

echo ""
echo "Inference complete! Results saved to $OUTPUT_DIR/"
