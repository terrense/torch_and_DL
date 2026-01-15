#!/bin/bash
# Run comparison between baseline and hybrid models
# Usage: bash scripts/run_comparison.sh

echo "Running Model Comparison: Baseline vs Hybrid"
echo "============================================="
echo ""

# Train baseline
echo "[1/2] Training U-Net Baseline..."
python scripts/train.py \
    --config configs/unet_baseline.yaml \
    --run-name comparison_baseline

echo ""
echo "[2/2] Training U-Net + Transformer Hybrid..."
python scripts/train.py \
    --config configs/unet_transformer.yaml \
    --run-name comparison_hybrid

echo ""
echo "============================================="
echo "Training complete!"
echo ""
echo "Results:"
echo "  Baseline: runs/comparison_baseline/"
echo "  Hybrid:   runs/comparison_hybrid/"
echo ""
echo "To generate comparison report:"
echo "  python scripts/summarize.py --runs-dir runs/ --filter comparison_"
