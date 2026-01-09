# Paraformer ASR Specification

## Overview

This project implements a Paraformer-style automatic speech recognition model from scratch using only PyTorch fundamentals. The implementation demonstrates sequence-to-sequence modeling for speech recognition, featuring an encoder-predictor-decoder architecture that handles variable-length audio features and text sequences.

## Architecture Components

### 1. Encoder (Transformer/Conformer-style)
- Multi-layer transformer encoder with self-attention
- Bidirectional processing of input feature sequences
- Layer normalization and residual connections
- Configurable depth and hidden dimensions

### 2. Predictor Module
- Duration/alignment prediction component
- Estimates token boundaries in feature sequences
- Provides conditioning signals for decoder processing
- Helps with attention alignment during training

### 3. Decoder/Refiner
- Token sequence generation from encoder features
- Attention over encoder outputs with proper masking
- Integration of predictor signals for improved alignment
- Greedy decoding for inference

## Model Components

### Core Building Blocks
- **MultiHeadAttention**: From-scratch implementation without torch.nn.MultiheadAttention
- **FeedForward**: Linear layers with activation and residual connections
- **TransformerBlock**: MHA + FFN with layer normalization
- **PositionalEncoding**: Learnable or sinusoidal position embeddings

### Data Pipeline
- **ToySeq2SeqDataset**: Synthetic speech-like feature generator
- **Tokenizer**: Character-level tokenization with vocabulary management
- **Collation**: Proper padding and masking for variable-length sequences

### Training Infrastructure
- **Loss Functions**: Masked cross-entropy for variable-length sequences
- **Metrics**: Token accuracy, sequence-level accuracy
- **Training Loop**: AdamW optimizer, scheduler, gradient clipping
- **Checkpointing**: Model state, optimizer state, metrics tracking

## Key Design Decisions

### Sequence Processing
The model handles variable-length sequences throughout:
- Input features: [B, T_feat, F] where T_feat varies per sample
- Output tokens: [B, T_token] where T_token varies per sample
- Proper masking ensures padded positions don't affect training

### Predictor Integration
The predictor module provides alignment information:
- Estimates how many tokens correspond to each feature frame
- Conditions the decoder attention mechanism
- Helps with monotonic alignment during training

### Attention Masking
Multiple types of attention masks are used:
- Padding masks: Ignore padded positions in sequences
- Causal masks: Prevent future information leakage (if needed)
- Cross-attention masks: Align encoder and decoder sequences

## Configuration System

Models are configured via YAML files with dataclass parsing:

```yaml
model:
  name: "paraformer"
  encoder:
    num_layers: 6
    hidden_dim: 512
    num_heads: 8
    dropout: 0.1
  predictor:
    hidden_dim: 256
    num_layers: 2
  decoder:
    num_layers: 6
    hidden_dim: 512
    num_heads: 8
    dropout: 0.1

training:
  batch_size: 16
  learning_rate: 1e-4
  num_epochs: 100
  max_grad_norm: 1.0
  warmup_steps: 1000
```

## Experiment Tracking

Each training run creates a timestamped directory:
```
runs/20240115_143022_paraformer/
├── config.yaml          # Configuration used
├── train.log            # Training logs
├── metrics.json         # Per-epoch metrics
├── results.csv          # Tabular results
└── checkpoints/
    ├── best.pt          # Best validation model
    └── latest.pt        # Latest checkpoint
```

## Performance Expectations

### Toy Dataset Results
- **Token Accuracy**: ~0.95 on synthetic sequences
- **Sequence Accuracy**: ~0.85 exact sequence matches
- **Training Time**: ~15 minutes for 100 epochs on toy data (GPU)

### Memory Requirements
- **Training**: ~4GB GPU memory (batch_size=16, seq_len=512)
- **Inference**: ~1GB GPU memory for single sequences

## Extension Points

### Adding New Architectures
1. Implement new encoder/decoder classes in `src/models/`
2. Add configuration schema in `src/config.py`
3. Register model in `src/models/__init__.py`
4. Create corresponding config YAML file

### Custom Datasets
1. Implement dataset class following `ToySeq2SeqDataset` interface
2. Add dataset configuration options
3. Update data loading pipeline in training scripts

### Advanced Decoding
1. Implement beam search in `src/decode/`
2. Add length normalization and coverage penalties
3. Update inference scripts to use new decoding methods

## Troubleshooting

### Common Issues
- **Sequence Length Mismatches**: Check tensor contracts in CONTRACTS.md
- **Attention Mask Errors**: Verify mask shapes and boolean values
- **NaN Losses**: Check learning rate, gradient clipping, and sequence lengths
- **Memory Errors**: Reduce batch size or maximum sequence length

### Debug Tools
- Tensor shape assertions throughout the pipeline
- Attention weight visualization utilities
- Sequence alignment debugging tools
- Training curve and loss component analysis

## Service Deployment

### FastAPI Inference Service
Optional REST API for model serving:
- JSON input: speech features as nested arrays
- JSON output: decoded text with confidence scores
- Configurable batch processing and timeout handling

### Usage Example
```bash
# Start service
python scripts/serve.py --checkpoint runs/latest/checkpoints/best.pt --port 8000

# Send request
curl -X POST "http://localhost:8000/transcribe" \
     -H "Content-Type: application/json" \
     -d '{"features": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]}'
```

## Learning Path

### Recommended Code Reading Order
1. `src/config.py` - Configuration system
2. `src/data/tokenizer.py` - Text processing
3. `src/data/toy_seq2seq.py` - Data generation
4. `src/models/transformer.py` - Core attention mechanisms
5. `src/models/encoder.py` - Encoder architecture
6. `src/models/predictor.py` - Alignment prediction
7. `src/models/decoder.py` - Token generation
8. `src/training/trainer.py` - Training loop
9. `src/decode/greedy.py` - Inference decoding

### Key Concepts to Understand
- Variable-length sequence handling
- Attention masking strategies
- Encoder-decoder architectures
- Sequence-to-sequence training
- Greedy vs beam search decoding