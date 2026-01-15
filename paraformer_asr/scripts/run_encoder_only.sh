#!/bin/bash
# Train encoder-only baseline model
# Usage: bash scripts/run_encoder_only.sh [run_name]

RUN_NAME=${1:-encoder_only}

echo "Training Encoder-Only Baseline Model"
echo "Run name: $RUN_NAME"
echo "====================================="

python scripts/train.py \
    --config configs/encoder_only.yaml \
    --run-name "$RUN_NAME"

echo ""
echo "Training complete! Results saved to runs/$RUN_NAME/"
echo "To evaluate: python scripts/evaluate.py --checkpoint runs/$RUN_NAME/checkpoints/best.pt"
