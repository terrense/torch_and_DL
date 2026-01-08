# Design Document

## Overview

This repository implements two independent machine learning projects from scratch to provide a comprehensive learning experience. The design emphasizes educational value through explicit implementation of all components, strict tensor contracts, and comprehensive experiment management. Each subproject demonstrates different aspects of modern ML: computer vision with hybrid CNN-transformer architectures, and sequence modeling for speech recognition.

## Architecture

### Repository Structure

```
repo/
├── README.md                    # Root documentation and quickstart
├── unet_transformer_seg/        # Project A: Image Segmentation
│   ├── README.md
│   ├── requirements.txt
│   ├── configs/
│   ├── docs/
│   ├── src/
│   ├── tests/
│   └── scripts/
└── paraformer_asr/             # Project B: Speech Recognition
    ├── README.md
    ├── requirements.txt
    ├── configs/
    ├── docs/
    ├── src/
    ├── tests/
    └── scripts/
```

### Design Principles

1. **Educational First**: Every component is implemented explicitly to demonstrate underlying mathematics
2. **Zero Code Sharing**: Complete independence between subprojects for focused learning
3. **Production Patterns**: Industry-standard structure with proper separation of concerns
4. **Fail-Fast Validation**: Comprehensive tensor contracts and runtime assertions
5. **Reproducible Experiments**: Deterministic training with comprehensive logging

## Components and Interfaces

### Project A: U-Net Transformer Segmentation

#### Core Architecture Components

**U-Net Baseline (unet.py)**
- Encoder: Downsampling path with skip connections
- Decoder: Upsampling path with skip connection fusion
- Interface: `forward(x: Tensor[B,C,H,W]) -> Tensor[B,num_classes,H,W]`

**Transformer Module (transformer.py)**
- Multi-Head Attention: Explicit QKV projection and scaled dot-product
- Feed-Forward Network: Linear layers with activation
- Layer Normalization and residual connections
- Positional encoding for spatial tokens

**Hybrid Integration (unet_transformer.py)**
- Bottleneck conversion: [B,C,H,W] ↔ [B,H*W,C] for transformer processing
- Spatial position encoding for 2D feature maps
- Configurable number of transformer layers at bottleneck

#### Data Pipeline Components

**Toy Dataset Generator (toy_shapes.py)**
```python
class ToyShapesDataset:
    def generate_sample(self) -> Tuple[Tensor, Tensor]:
        # Returns (image[C,H,W], mask[H,W])
        pass
```

**Transform Pipeline (transforms.py)**
- Resize with mask alignment
- Normalization and augmentation
- Tensor conversion utilities

#### Training Infrastructure

**Loss Functions (losses/)**
- Dice Loss: Soft dice coefficient for segmentation
- BCE + Dice: Combined binary cross-entropy and dice loss
- Interface: `loss_fn(pred: Tensor, target: Tensor) -> Tensor`

**Metrics (metrics/seg_metrics.py)**
- Intersection over Union (IoU)
- Pixel accuracy
- Per-class dice scores

### Project B: Paraformer ASR

#### Core Architecture Components

**Encoder (encoder.py)**
- Multi-layer transformer/conformer-style encoder
- Self-attention with causal or bidirectional masking
- Feed-forward networks with residual connections
- Interface: `forward(features: Tensor[B,T,F], mask: Tensor[B,T]) -> Tensor[B,T,D]`

**Predictor (predictor.py)**
- Duration/alignment prediction module
- Estimates token boundaries in feature sequence
- Interface: `forward(encoder_out: Tensor[B,T,D]) -> Tensor[B,T,1]`

**Decoder (decoder.py)**
- Token sequence generation from encoder features
- Attention over encoder outputs
- Interface: `forward(encoder_out: Tensor, predictor_out: Tensor) -> Tensor[B,S,vocab_size]`

#### Data Pipeline Components

**Toy Sequence Generator (toy_seq2seq.py)**
```python
class ToySeq2SeqDataset:
    def generate_sample(self) -> Tuple[Tensor, Tensor, int, int]:
        # Returns (features[T,F], tokens[S], feat_len, token_len)
        pass
```

**Tokenizer (tokenizer.py)**
- Character-level tokenization
- Vocabulary management
- Encode/decode utilities

#### Training Infrastructure

**Loss Functions (losses/seq_loss.py)**
- Masked cross-entropy for variable-length sequences
- Optional auxiliary losses for predictor training

**Decoding (decode/greedy.py)**
- Greedy decoding from logits
- Beam search (optional extension)

## Data Models

### Configuration System

Both projects use YAML configuration with dataclass parsing:

```python
@dataclass
class ModelConfig:
    name: str
    hidden_dim: int
    num_layers: int
    # ... other parameters

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    # ... other parameters
```

### Experiment Tracking

**Run Management**
```
runs/
├── 20240115_143022_unet_baseline/
│   ├── config.yaml
│   ├── train.log
│   ├── metrics.json
│   ├── results.csv
│   └── checkpoints/
└── 20240115_143155_unet_transformer/
    └── ...
```

**Results Schema**
- CSV format with columns: run_id, model_type, epoch, loss, accuracy, dice_score, etc.
- JSON format for detailed per-epoch metrics

### Tensor Contracts

Each module documents explicit tensor contracts:

| Module | Input | Shape | Dtype | Range | Output | Shape | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|
| UNet.forward | image | [B,3,H,W] | float32 | [0,1] | logits | [B,C,H,W] | C=num_classes |
| MHA.forward | x | [B,T,D] | float32 | any | out | [B,T,D] | residual added |

## Error Handling

### Runtime Validation

**Shape Assertions**
```python
def assert_shape(tensor: Tensor, expected: str, name: str = "tensor"):
    """Assert tensor matches expected shape pattern like 'B,C,H,W'"""
    pass
```

**NaN/Inf Detection**
```python
def check_nan_inf(tensor: Tensor, name: str):
    """Detect and report NaN/Inf values with context"""
    pass
```

### Graceful Degradation

- Checkpoint recovery on training interruption
- Validation of config parameters before training
- Clear error messages for common shape mismatches

## Testing Strategy

### Unit Testing

**Model Shape Tests**
- Verify output shapes for all model components
- Test with various input dimensions
- Validate gradient flow

**Data Pipeline Tests**
- Toy dataset generation consistency
- Transform pipeline correctness
- Batch collation and masking

### Integration Testing

**Smoke Tests**
- End-to-end training for 30-100 steps
- Loss decrease validation
- Checkpoint save/load functionality

**CLI Testing**
- Train/eval/infer command validation
- Configuration loading and validation
- Output file generation

### Performance Testing

**Memory Profiling**
- Monitor GPU memory usage during training
- Validate batch size scaling

**Speed Benchmarking**
- Training throughput measurement
- Inference latency testing

## Implementation Phases

### Phase 1: Core Infrastructure
1. Project structure setup
2. Configuration system implementation
3. Logging and checkpointing utilities
4. Basic tensor validation utilities

### Phase 2: Data Pipelines
1. Toy dataset generators
2. Data loading and batching
3. Transform pipelines
4. Collation functions with masking

### Phase 3: Model Implementation
1. Basic building blocks (linear, conv, norm layers)
2. Attention mechanisms from scratch
3. Core model architectures
4. Loss functions and metrics

### Phase 4: Training Systems
1. Training loop implementation
2. Evaluation pipelines
3. Inference scripts
4. CLI interfaces

### Phase 5: Testing and Documentation
1. Comprehensive test suite
2. Documentation generation
3. Example scripts and tutorials
4. Performance optimization

## Deployment Considerations

### Development Environment
- Python 3.10+ virtual environment
- CUDA-capable GPU (optional but recommended)
- Sufficient disk space for experiment logs

### Optional Service Deployment
- FastAPI service for Paraformer ASR inference
- Docker containerization (optional extension)
- REST API for model serving

## Security Considerations

- Input validation for all user-provided data
- Safe file I/O operations
- Memory management for large tensors
- Secure configuration loading

## Performance Optimization

### Memory Efficiency
- Gradient checkpointing for large models
- Mixed precision training support
- Efficient tensor operations

### Computational Efficiency
- Vectorized operations where possible
- Minimal Python loops in hot paths
- Efficient attention implementations

## Extensibility

### Model Variants
- Easy configuration-based model switching
- Modular component design for experimentation
- Clear interfaces for adding new architectures

### Dataset Integration
- Abstract dataset interfaces for real data integration
- Transform pipeline extensibility
- Custom metric implementation support