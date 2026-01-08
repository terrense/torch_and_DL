# Paraformer ASR

This project implements a Paraformer-style automatic speech recognition model from scratch using only PyTorch fundamentals.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train.py --config configs/paraformer_base.yaml

# Evaluate model
python scripts/eval.py --checkpoint runs/latest/checkpoints/best.pt

# Run inference
python scripts/infer.py --checkpoint runs/latest/checkpoints/best.pt --features path/to/features.npy

# Start inference service (optional)
python scripts/serve.py --checkpoint runs/latest/checkpoints/best.pt --port 8000
```

## Project Structure

- `configs/` - YAML configuration files for model variants
- `src/` - Source code for models, data, training, and utilities
- `tests/` - Unit tests and smoke tests
- `scripts/` - Training, evaluation, inference, and service scripts
- `docs/` - Detailed documentation and specifications

## Features

- Transformer/Conformer-style encoder implementation
- Predictor module for token alignment estimation
- Decoder/refiner for token generation
- Greedy decoding from logits
- Synthetic sequence-to-sequence dataset generator
- Character-level tokenizer from scratch
- Comprehensive experiment tracking and logging

See the main repository README for detailed setup and learning guidance.