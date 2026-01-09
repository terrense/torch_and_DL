# U-Net Transformer Segmentation

This project implements a hybrid U-Net + Transformer architecture for image segmentation from scratch using only PyTorch fundamentals.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline U-Net on toy shapes dataset
python scripts/train.py --config configs/unet_baseline.yaml --run-name unet_baseline_toy

# Train U-Net + Transformer hybrid
python scripts/train.py --config configs/unet_transformer.yaml --run-name unet_transformer_toy

# Evaluate trained model
python scripts/eval.py --checkpoint runs/unet_baseline_toy/checkpoints/best.pt --output results/eval_baseline.json

# Run inference on single image
python scripts/infer.py --checkpoint runs/unet_transformer_toy/checkpoints/best.pt --image data/test_image.jpg --output results/prediction.png

# Generate toy dataset samples
python scripts/generate_toy_data.py --output data/toy_samples/ --num-samples 100

# Run smoke test (quick training verification)
python scripts/smoke_test.py --config configs/unet_baseline.yaml --steps 50
```

## Project Structure

- `configs/` - YAML configuration files for different model variants
- `src/` - Source code for models, data, training, and utilities
- `tests/` - Unit tests and smoke tests
- `scripts/` - Training, evaluation, and inference scripts
- `docs/` - Detailed documentation and specifications

## Features

- Pure U-Net baseline implementation
- U-Net + Transformer hybrid with bottleneck integration
- Synthetic toy dataset generator for shapes segmentation
- From-scratch transforms and data loading
- Dice and IoU metrics implementation
- Comprehensive experiment tracking and logging

See the main repository README for detailed setup and learning guidance.