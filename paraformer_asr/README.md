# Paraformer ASR

This project implements a Paraformer-style automatic speech recognition model from scratch using only PyTorch fundamentals.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model on toy dataset
python scripts/train.py --config configs/paraformer_base.yaml --run-name my_first_run

# Evaluate trained model
python scripts/eval.py --checkpoint runs/my_first_run/checkpoints/best.pt

# Run inference on single sequence
python scripts/infer.py --checkpoint runs/my_first_run/checkpoints/best.pt --text "hello world"

# Start inference service (optional)
python scripts/serve.py --checkpoint runs/my_first_run/checkpoints/best.pt --port 8000

# Test service
curl -X POST "http://localhost:8000/transcribe" \
     -H "Content-Type: application/json" \
     -d '{"features": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}'
```

## Project Structure

- `configs/` - YAML configuration files for model variants
- `src/` - Source code for models, data, training, and utilities
- `tests/` - Unit tests and smoke tests
- `scripts/` - Training, evaluation, inference, and service scripts
- `docs/` - Detailed documentation and specifications

## Detailed Usage

### Training
```bash
# Basic training
python scripts/train.py --config configs/paraformer_base.yaml

# Custom run name and overrides
python scripts/train.py --config configs/paraformer_base.yaml \
    --run-name experiment_1 \
    --override training.batch_size=32 training.learning_rate=2e-4

# Resume from checkpoint
python scripts/train.py --config configs/paraformer_base.yaml \
    --resume runs/experiment_1/checkpoints/latest.pt
```

### Evaluation
```bash
# Evaluate on validation set
python scripts/eval.py --checkpoint runs/experiment_1/checkpoints/best.pt

# Evaluate with different decoding methods
python scripts/eval.py --checkpoint runs/experiment_1/checkpoints/best.pt \
    --decode-method beam --beam-size 4

# Detailed evaluation with metrics breakdown
python scripts/eval.py --checkpoint runs/experiment_1/checkpoints/best.pt \
    --detailed --output results/detailed_eval.json
```

### Inference
```bash
# Single text to features and back
python scripts/infer.py --checkpoint runs/experiment_1/checkpoints/best.pt \
    --text "hello world" --output predictions.txt

# Batch inference from file
python scripts/infer.py --checkpoint runs/experiment_1/checkpoints/best.pt \
    --input-file texts.txt --output-file predictions.txt

# Interactive inference
python scripts/infer.py --checkpoint runs/experiment_1/checkpoints/best.pt --interactive
```

## Configuration

Models are configured via YAML files. Key parameters:

```yaml
model:
  encoder:
    num_layers: 6        # Transformer encoder layers
    hidden_dim: 512      # Hidden dimension
    num_heads: 8         # Attention heads
  predictor:
    num_layers: 2        # Predictor layers
    hidden_dim: 256      # Predictor hidden dim
  decoder:
    num_layers: 6        # Decoder layers

training:
  batch_size: 16         # Training batch size
  learning_rate: 1e-4    # Learning rate
  num_epochs: 100        # Training epochs
  max_grad_norm: 1.0     # Gradient clipping

data:
  max_feat_len: 300      # Maximum feature sequence length
  max_token_len: 60      # Maximum token sequence length
  vocab_size: 100        # Vocabulary size
```

## Results on Toy Dataset

| Model | Token Accuracy | Sequence Accuracy | Training Time |
|-------|---------------|-------------------|---------------|
| Encoder-only | 0.90 | 0.75 | 10 min |
| Full Paraformer | 0.95 | 0.85 | 15 min |

*Results on synthetic sequence-to-sequence dataset with 100 epochs training*
## 
Features

- **Encoder**: Transformer/Conformer-style encoder with self-attention
- **Predictor**: Duration/alignment prediction for better sequence modeling
- **Decoder**: Token generation with cross-attention over encoder features
- **Tokenizer**: Character-level tokenization from scratch
- **Dataset**: Synthetic sequence-to-sequence data generator
- **Training**: Complete training loop with logging and checkpointing
- **Inference**: Greedy and beam search decoding
- **Service**: Optional FastAPI service for model serving

## Documentation

- `docs/SPEC.md` - Detailed architecture specification
- `docs/CONTRACTS.md` - Tensor shapes and data contracts
- `docs/ABLATIONS.md` - Ablation studies and model comparisons

## Learning Path

1. Start with `src/config.py` to understand configuration system
2. Explore `src/data/tokenizer.py` for text processing
3. Study `src/models/transformer.py` for attention mechanisms
4. Follow the encoder → predictor → decoder architecture
5. Examine training loop in `src/training/trainer.py`
6. Try inference with `scripts/infer.py`

See the main repository README for detailed setup and learning guidance.