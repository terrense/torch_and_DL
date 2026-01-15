#!/bin/bash
# Train U-Net + Transformer hybrid model
# Usage: bash scripts/run_hybrid.sh [run_name]

RUN_NAME=${1:-unet_transformer}

echo "Training U-Net + Transformer Hybrid Model"
echo "Run name: $RUN_NAME"
echo "=========================================="

python scripts/train.py \
    --config configs/unet_transformer.yaml \
    --run-name "$RUN_NAME"

echo ""
echo "Training complete! Results saved to runs/$RUN_NAME/"
echo "To evaluate: python scripts/evaluate.py --checkpoint runs/$RUN_NAME/checkpoints/best.pt"
