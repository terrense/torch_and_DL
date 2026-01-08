# U-Net Transformer Segmentation

This project implements a hybrid U-Net + Transformer architecture for image segmentation from scratch using only PyTorch fundamentals.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline U-Net
python scripts/train.py --config configs/unet_baseline.yaml

# Train U-Net + Transformer hybrid
python scripts/train.py --config configs/unet_transformer.yaml

# Evaluate model
python scripts/eval.py --checkpoint runs/latest/checkpoints/best.pt

# Run inference
python scripts/infer.py --checkpoint runs/latest/checkpoints/best.pt --image path/to/image.jpg
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