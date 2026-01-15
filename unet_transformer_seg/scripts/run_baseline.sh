#!/bin/bash
# Train U-Net baseline model
# Usage: bash scripts/run_baseline.sh [run_name]

RUN_NAME=${1:-unet_baseline}

echo "Training U-Net Baseline Model"
echo "Run name: $RUN_NAME"
echo "================================"

python scripts/train.py \
    --config configs/unet_baseline.yaml \
    --run-name "$RUN_NAME"

echo ""
echo "Training complete! Results saved to runs/$RUN_NAME/"
echo "To evaluate: python scripts/evaluate.py --checkpoint runs/$RUN_NAME/checkpoints/best.pt"
