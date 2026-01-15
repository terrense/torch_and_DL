#!/bin/bash
# Train full Paraformer model (encoder + predictor + decoder)
# Usage: bash scripts/run_full.sh [run_name]

RUN_NAME=${1:-paraformer_full}

echo "Training Full Paraformer Model"
echo "Run name: $RUN_NAME"
echo "=============================="

python scripts/train.py \
    --config configs/paraformer_base.yaml \
    --run-name "$RUN_NAME"

echo ""
echo "Training complete! Results saved to runs/$RUN_NAME/"
echo "To evaluate: python scripts/evaluate.py --checkpoint runs/$RUN_NAME/checkpoints/best.pt"
