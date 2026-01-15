#!/bin/bash
# Run comparison between encoder-only and full Paraformer models
# Usage: bash scripts/run_comparison.sh

echo "Running Model Comparison: Encoder-Only vs Full Paraformer"
echo "=========================================================="
echo ""

# Train encoder-only baseline
echo "[1/2] Training Encoder-Only Baseline..."
python scripts/train.py \
    --config configs/encoder_only.yaml \
    --run-name comparison_encoder_only

echo ""
echo "[2/2] Training Full Paraformer..."
python scripts/train.py \
    --config configs/paraformer_base.yaml \
    --run-name comparison_full

echo ""
echo "=========================================================="
echo "Training complete!"
echo ""
echo "Results:"
echo "  Encoder-Only: runs/comparison_encoder_only/"
echo "  Full:         runs/comparison_full/"
echo ""
echo "To generate comparison report:"
echo "  python scripts/summarize.py --runs-dir runs/ --filter comparison_"
