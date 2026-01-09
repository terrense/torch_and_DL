# U-Net Transformer Segmentation Specification

## Overview

This project implements a hybrid U-Net + Transformer architecture for image segmentation, demonstrating the integration of convolutional and attention-based approaches. The implementation is built from scratch using only PyTorch fundamentals to provide educational insight into both architectures.

## Architecture Variants

### 1. U-Net Baseline
- Pure convolutional encoder-decoder architecture
- Skip connections between encoder and decoder layers
- Configurable depth and channel dimensions
- Standard downsampling/upsampling operations

### 2. U-Net + Transformer Hybrid
- U-Net encoder and decoder with transformer bottleneck
- Tensor reshaping between [B,C,H,W] and [B,T,D] formats
- Multi-head attention at the bottleneck level
- Positional encoding for spatial tokens

## Model Components

### Core Building Blocks
- **ConvBlock**: Conv2d → BatchNorm → ReLU
- **DownBlock**: ConvBlock → MaxPool2d
- **UpBlock**: ConvTranspose2d → ConvBlock
- **MultiHeadAttention**: From-scratch implementation without torch.nn.MultiheadAttention
- **TransformerBlock**: MHA + FFN with residual connections

### Data Pipeline
- **ToyShapesDataset**: Synthetic segmentation data generator
- **Transforms**: Resize, normalize, augmentation with mask alignment
- **DataLoader**: Batching with proper collation for variable sizes

### Training Infrastructure
- **Loss Functions**: Dice loss, BCE + Dice combination
- **Metrics**: IoU, pixel accuracy, per-class dice scores
- **Training Loop**: AdamW optimizer, scheduler, gradient clipping
- **Checkpointing**: Model state, optimizer state, metrics tracking

## Key Design Decisions

### Tensor Format Conversion
The hybrid model requires careful conversion between CNN format [B,C,H,W] and transformer format [B,T,D]:
- Flatten spatial dimensions: [B,C,H,W] → [B,C,H*W] → [B,H*W,C]
- Apply transformer layers in [B,T,D] format where T=H*W, D=C
- Reshape back: [B,H*W,C] → [B,C,H*W] → [B,C,H,W]

### Positional Encoding
2D spatial positional encoding for transformer tokens:
- Separate encodings for height and width dimensions
- Learnable or sinusoidal position embeddings
- Added to flattened spatial tokens before attention

### Skip Connection Integration
In the hybrid model, skip connections from encoder are preserved:
- Encoder features bypass the transformer bottleneck
- Decoder receives both transformed bottleneck features and skip connections
- Proper channel dimension alignment required

## Configuration System

Models are configured via YAML files with dataclass parsing:

```yaml
model:
  name: "unet_transformer"
  encoder_channels: [64, 128, 256, 512]
  decoder_channels: [256, 128, 64, 32]
  transformer:
    num_layers: 4
    num_heads: 8
    hidden_dim: 512
    dropout: 0.1

training:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 100
  loss_weights:
    dice: 0.5
    bce: 0.5
```

## Experiment Tracking

Each training run creates a timestamped directory:
```
runs/20240115_143022_unet_transformer/
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
- **U-Net Baseline**: ~0.85 IoU on synthetic shapes
- **U-Net + Transformer**: ~0.87 IoU with improved boundary detection
- **Training Time**: ~10 minutes for 100 epochs on toy data (GPU)

### Memory Requirements
- **U-Net Baseline**: ~2GB GPU memory (batch_size=8, 256x256 images)
- **U-Net + Transformer**: ~3GB GPU memory (additional transformer overhead)

## Extension Points

### Adding New Architectures
1. Implement new model class in `src/models/`
2. Add configuration schema in `src/config.py`
3. Register model in `src/models/__init__.py`
4. Create corresponding config YAML file

### Custom Datasets
1. Implement dataset class following `ToyShapesDataset` interface
2. Add dataset configuration options
3. Update data loading pipeline in training scripts

### New Loss Functions
1. Implement loss class in `src/losses/`
2. Add loss configuration options
3. Update training loop to use new loss

## Troubleshooting

### Common Issues
- **Shape Mismatches**: Check tensor contracts in CONTRACTS.md
- **Memory Errors**: Reduce batch size or image resolution
- **NaN Losses**: Check learning rate and gradient clipping
- **Slow Training**: Verify GPU utilization and data loading efficiency

### Debug Tools
- Tensor shape assertions throughout the pipeline
- NaN/Inf detection in losses and gradients
- Memory profiling utilities
- Training curve visualization