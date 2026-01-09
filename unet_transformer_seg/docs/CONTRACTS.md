# Tensor Contracts Documentation

This document defines explicit tensor shapes, dtypes, and value ranges for all major components in the U-Net Transformer Segmentation project.

## Notation

- `B` = Batch size
- `C` = Number of channels
- `H` = Height dimension
- `W` = Width dimension
- `T` = Sequence length (H*W for flattened spatial)
- `D` = Hidden dimension
- `N` = Number of classes
- `F` = Number of features

## Data Pipeline Contracts

### ToyShapesDataset

| Method | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `__getitem__` | index | scalar | int | [0, len) | image | [3,H,W] | float32 | [0,1] | RGB image |
| | | | | | mask | [H,W] | long | [0,N-1] | Class labels |
| `generate_shape` | shape_type | str | - | - | image | [3,H,W] | float32 | [0,1] | Generated shape |
| | | | | | mask | [H,W] | long | [0,N-1] | Corresponding mask |

### Transforms

| Transform | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|-----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `Resize` | image | [3,H,W] | float32 | [0,1] | image | [3,H',W'] | float32 | [0,1] | Bilinear interpolation |
| | mask | [H,W] | long | [0,N-1] | mask | [H',W'] | long | [0,N-1] | Nearest neighbor |
| `Normalize` | image | [3,H,W] | float32 | [0,1] | image | [3,H,W] | float32 | [-2,2] | ImageNet stats |
| `RandomFlip` | image | [3,H,W] | float32 | any | image | [3,H,W] | float32 | any | 50% probability |
| | mask | [H,W] | long | [0,N-1] | mask | [H,W] | long | [0,N-1] | Same flip as image |

### DataLoader Collation

| Function | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `collate_fn` | images | List[[3,H,W]] | float32 | any | batch | [B,3,H,W] | float32 | any | Padded to max size |
| | masks | List[[H,W]] | long | [0,N-1] | batch | [B,H,W] | long | [0,N-1] | Same padding |

## Model Architecture Contracts

### U-Net Components

| Component | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|-----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `ConvBlock` | x | [B,C_in,H,W] | float32 | any | x | [B,C_out,H,W] | float32 | any | Conv→BN→ReLU |
| `DownBlock` | x | [B,C_in,H,W] | float32 | any | x | [B,C_out,H/2,W/2] | float32 | any | ConvBlock→MaxPool |
| `UpBlock` | x | [B,C_in,H,W] | float32 | any | x | [B,C_out,H*2,W*2] | float32 | any | ConvTranspose→ConvBlock |
| | skip | [B,C_skip,H*2,W*2] | float32 | any | | | | | Concatenated input |

### U-Net Model

| Method | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `forward` | x | [B,3,H,W] | float32 | any | logits | [B,N,H,W] | float32 | any | Raw predictions |
| `encode` | x | [B,3,H,W] | float32 | any | features | [B,C,H/16,W/16] | float32 | any | Bottleneck features |
| | | | | | skips | List[[B,C_i,H_i,W_i]] | float32 | any | Skip connections |
| `decode` | features | [B,C,H/16,W/16] | float32 | any | logits | [B,N,H,W] | float32 | any | Final predictions |
| | skips | List[[B,C_i,H_i,W_i]] | float32 | any | | | | | From encoder |

### Transformer Components

| Component | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|-----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `MultiHeadAttention` | x | [B,T,D] | float32 | any | x | [B,T,D] | float32 | any | Self-attention |
| | mask | [B,T,T] | bool | {0,1} | | | | | Optional attention mask |
| `FeedForward` | x | [B,T,D] | float32 | any | x | [B,T,D] | float32 | any | Linear→ReLU→Linear |
| `TransformerBlock` | x | [B,T,D] | float32 | any | x | [B,T,D] | float32 | any | MHA + FFN + residual |
| `PositionalEncoding` | x | [B,T,D] | float32 | any | x | [B,T,D] | float32 | any | Adds position info |

### U-Net + Transformer Hybrid

| Method | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `forward` | x | [B,3,H,W] | float32 | any | logits | [B,N,H,W] | float32 | any | End-to-end prediction |
| `reshape_for_transformer` | x | [B,C,H,W] | float32 | any | x | [B,H*W,C] | float32 | any | CNN → Transformer |
| `reshape_from_transformer` | x | [B,H*W,C] | float32 | any | x | [B,C,H,W] | float32 | any | Transformer → CNN |
| `apply_transformer` | x | [B,T,D] | float32 | any | x | [B,T,D] | float32 | any | Multi-layer transformer |

## Loss Functions Contracts

| Loss Function | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|---------------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `DiceLoss` | pred | [B,N,H,W] | float32 | any | loss | scalar | float32 | [0,1] | Soft dice coefficient |
| | target | [B,H,W] | long | [0,N-1] | | | | | Ground truth labels |
| `BCEDiceLoss` | pred | [B,N,H,W] | float32 | any | loss | scalar | float32 | [0,∞) | Weighted combination |
| | target | [B,H,W] | long | [0,N-1] | | | | | Ground truth labels |

## Metrics Contracts

| Metric | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `IoU` | pred | [B,N,H,W] | float32 | any | iou | [N] | float32 | [0,1] | Per-class IoU |
| | target | [B,H,W] | long | [0,N-1] | | | | | Ground truth |
| `PixelAccuracy` | pred | [B,N,H,W] | float32 | any | acc | scalar | float32 | [0,1] | Overall accuracy |
| | target | [B,H,W] | long | [0,N-1] | | | | | Ground truth |
| `DiceScore` | pred | [B,N,H,W] | float32 | any | dice | [N] | float32 | [0,1] | Per-class dice |
| | target | [B,H,W] | long | [0,N-1] | | | | | Ground truth |

## Training Loop Contracts

| Component | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|-----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `train_step` | batch | ([B,3,H,W], [B,H,W]) | (float32, long) | ([0,1], [0,N-1]) | loss | scalar | float32 | [0,∞) | Single training step |
| `val_step` | batch | ([B,3,H,W], [B,H,W]) | (float32, long) | ([0,1], [0,N-1]) | metrics | dict | float32 | [0,1] | Validation metrics |
| `predict` | image | [3,H,W] | float32 | [0,1] | pred | [H,W] | long | [0,N-1] | Single image prediction |

## Utility Functions Contracts

| Function | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `assert_shape` | tensor | any | any | any | None | - | - | - | Raises on mismatch |
| | expected | str | - | - | | | | | Pattern like "B,C,H,W" |
| `check_nan_inf` | tensor | any | float32 | any | bool | scalar | bool | {0,1} | True if NaN/Inf found |
| `set_seed` | seed | scalar | int | [0,∞) | None | - | - | - | Sets all random seeds |

## Common Shape Patterns

### Standard Image Sizes
- **Training**: 256×256 (configurable)
- **Inference**: Variable size (handled by transforms)

### Batch Sizes
- **Training**: 8-16 (depending on GPU memory)
- **Validation**: 16-32 (larger batches for efficiency)

### Channel Configurations
- **Input**: 3 (RGB images)
- **Encoder**: [64, 128, 256, 512] (configurable)
- **Decoder**: [256, 128, 64, 32] (configurable)
- **Output**: N (number of classes, typically 2-10)

### Transformer Dimensions
- **Hidden Dim**: 512 (matches bottleneck channels)
- **Num Heads**: 8 (hidden_dim must be divisible)
- **Sequence Length**: H*W/256 (after max pooling)

## Validation Guidelines

### Runtime Assertions
All major functions include shape assertions:
```python
assert_shape(x, "B,3,H,W", "input image")
assert_shape(mask, "B,H,W", "target mask")
```

### NaN/Inf Checks
Critical points include NaN/Inf detection:
```python
check_nan_inf(loss, "training loss")
check_nan_inf(gradients, "model gradients")
```

### Memory Monitoring
Track GPU memory usage during training:
- Peak memory should not exceed available GPU memory
- Memory leaks indicated by steadily increasing usage
- Batch size adjustments based on memory constraints