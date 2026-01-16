## U-Net Transformer Segmentation Architecture

This document provides a detailed explanation of the hybrid U-Net + Transformer architecture for image segmentation.

## Overview

The architecture combines the strengths of two powerful approaches:
- **U-Net**: Excellent at capturing spatial hierarchies through encoder-decoder structure with skip connections
- **Transformer**: Powerful at modeling long-range dependencies through self-attention mechanisms

## Architecture Components

### 1. U-Net Baseline

The baseline U-Net follows the classic encoder-decoder architecture:

```
Input Image [B, 3, H, W]
    ↓
Encoder Path (Downsampling)
    ├─ Conv Block 1 [B, 64, H, W]
    ├─ Down Block 1 [B, 128, H/2, W/2]
    ├─ Down Block 2 [B, 256, H/4, W/4]
    ├─ Down Block 3 [B, 512, H/8, W/8]
    └─ Bottleneck [B, 1024, H/16, W/16]
    ↓
Decoder Path (Upsampling)
    ├─ Up Block 1 + Skip [B, 512, H/8, W/8]
    ├─ Up Block 2 + Skip [B, 256, H/4, W/4]
    ├─ Up Block 3 + Skip [B, 128, H/2, W/2]
    └─ Up Block 4 + Skip [B, 64, H, W]
    ↓
Output Logits [B, num_classes, H, W]
```

#### Encoder Path
Each encoder block consists of:
1. **Convolution**: 3×3 conv with padding=1
2. **Batch Normalization**: Normalizes activations
3. **ReLU Activation**: Non-linearity
4. **Max Pooling** (except first block): 2×2 pooling for downsampling

The encoder progressively:
- Reduces spatial dimensions (H, W → H/16, W/16)
- Increases channel dimensions (3 → 1024)
- Captures hierarchical features from low-level to high-level

#### Decoder Path
Each decoder block consists of:
1. **Transposed Convolution**: 2×2 conv for upsampling
2. **Skip Connection Concatenation**: Fuses encoder features
3. **Convolution Blocks**: Refines combined features

The decoder progressively:
- Increases spatial dimensions (H/16, W/16 → H, W)
- Decreases channel dimensions (1024 → 64)
- Recovers spatial details using skip connections

#### Skip Connections
Skip connections are crucial for:
- **Preserving spatial information** lost during downsampling
- **Enabling gradient flow** during backpropagation
- **Combining multi-scale features** from different levels

### 2. Transformer Module

The transformer module processes features at the bottleneck:

```
Bottleneck Features [B, C, H/16, W/16]
    ↓
Reshape to Sequence [B, (H/16)×(W/16), C]
    ↓
Add Positional Encoding [B, T, C]
    ↓
Transformer Block 1
    ├─ Multi-Head Self-Attention
    ├─ Add & Norm (Residual)
    ├─ Feed-Forward Network
    └─ Add & Norm (Residual)
    ↓
Transformer Block 2...N
    ↓
Reshape to Spatial [B, C, H/16, W/16]
```

#### Multi-Head Self-Attention

The attention mechanism allows each position to attend to all other positions:

```python
# For each head h:
Q = Linear(x)  # Query: [B, T, D/H]
K = Linear(x)  # Key: [B, T, D/H]
V = Linear(x)  # Value: [B, T, D/H]

# Scaled dot-product attention
scores = (Q @ K.T) / sqrt(D/H)  # [B, T, T]
attention = softmax(scores)      # [B, T, T]
output = attention @ V           # [B, T, D/H]

# Concatenate all heads
multi_head_output = concat([head_1, ..., head_H])  # [B, T, D]
```

**Why it works:**
- **Global receptive field**: Each position can attend to all positions
- **Adaptive weights**: Attention weights learned based on content
- **Multiple heads**: Different heads can focus on different patterns

#### Positional Encoding

Since transformers don't have inherent position information, we add positional encodings:

```python
# 2D positional encoding for spatial features
pos_h = torch.arange(H).unsqueeze(1).repeat(1, W)  # [H, W]
pos_w = torch.arange(W).unsqueeze(0).repeat(H, 1)  # [H, W]

# Sinusoidal encoding
pos_encoding_h = sin(pos_h / 10000^(2i/D))
pos_encoding_w = sin(pos_w / 10000^(2i/D))

# Combine and add to features
pos_encoding = concat([pos_encoding_h, pos_encoding_w])
features = features + pos_encoding
```

### 3. Hybrid U-Net + Transformer

The hybrid model integrates transformers at the bottleneck:

```
Encoder Path
    ↓
Bottleneck [B, C, H/16, W/16]
    ↓
Reshape for Transformer [B, T, C]  where T = (H/16) × (W/16)
    ↓
Transformer Layers (N layers)
    ├─ Self-Attention: Models global dependencies
    ├─ Feed-Forward: Processes attended features
    └─ Residual Connections: Preserves information
    ↓
Reshape back to Spatial [B, C, H/16, W/16]
    ↓
Decoder Path
```

**Key Design Decisions:**

1. **Where to apply transformers?**
   - At the bottleneck (lowest spatial resolution)
   - Reason: Computational cost of attention is O(T²), so we apply it where T is smallest

2. **How many transformer layers?**
   - Default: 4 layers
   - Trade-off: More layers = better modeling but slower training

3. **How to reshape?**
   - Flatten spatial dimensions: [B, C, H, W] → [B, H×W, C]
   - Treat each spatial position as a token in the sequence
   - Reshape back after transformer: [B, H×W, C] → [B, C, H, W]

## Loss Functions

### Dice Loss

Dice loss measures overlap between prediction and ground truth:

```python
def dice_loss(pred, target):
    # Soft dice coefficient
    pred = softmax(pred, dim=1)  # [B, C, H, W]
    target_one_hot = one_hot(target, num_classes=C)  # [B, C, H, W]
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  # [B, C]
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # [B, C]
    
    dice = (2 * intersection + epsilon) / (union + epsilon)  # [B, C]
    return 1 - dice.mean()
```

**Why Dice Loss?**
- Handles class imbalance better than cross-entropy
- Directly optimizes the IoU-like metric
- Smooth and differentiable

### Combined BCE + Dice Loss

```python
def combined_loss(pred, target, bce_weight=0.5, dice_weight=0.5):
    bce = F.cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice
```

**Why combine?**
- BCE provides stable gradients
- Dice handles class imbalance
- Together they complement each other

## Training Strategy

### 1. Data Augmentation

```python
transforms = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.2, contrast=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 2. Optimization

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)
```

### 3. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Inference

### Single Image Prediction

```python
model.eval()
with torch.no_grad():
    # Preprocess
    image = transform(image).unsqueeze(0)  # [1, 3, H, W]
    
    # Forward pass
    logits = model(image)  # [1, C, H, W]
    
    # Get predictions
    pred_mask = logits.argmax(dim=1)  # [1, H, W]
```

### Batch Prediction

```python
model.eval()
with torch.no_grad():
    for batch in dataloader:
        images, _ = batch
        logits = model(images)
        pred_masks = logits.argmax(dim=1)
```

## Performance Considerations

### Memory Usage

- **Baseline U-Net**: ~2GB GPU memory (batch_size=8, 256×256 images)
- **Hybrid Model**: ~3GB GPU memory (4 transformer layers)

Memory scales with:
- Batch size (linear)
- Image size (quadratic for transformers)
- Number of transformer layers (linear)

### Training Speed

- **Baseline U-Net**: ~8 min for 100 epochs
- **Hybrid Model**: ~12 min for 100 epochs

Speed depends on:
- GPU capability
- Batch size
- Image resolution
- Number of transformer layers

### Optimization Tips

1. **Use mixed precision training**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       logits = model(images)
       loss = criterion(logits, masks)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **Gradient checkpointing** for large models:
   ```python
   from torch.utils.checkpoint import checkpoint
   
   def forward_with_checkpoint(self, x):
       return checkpoint(self.transformer_block, x)
   ```

3. **Efficient data loading**:
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=16,
       num_workers=4,  # Parallel data loading
       pin_memory=True,  # Faster GPU transfer
       prefetch_factor=2  # Prefetch batches
   )
   ```

## Extending the Architecture

### Adding More Transformer Layers

```python
# In config
model:
  transformer:
    num_layers: 8  # Increase from 4
```

### Using Different Attention Mechanisms

```python
# Replace standard attention with efficient variants
from src.models.efficient_attention import LinearAttention

class EfficientTransformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = LinearAttention(hidden_dim)  # O(T) instead of O(T²)
```

### Multi-Scale Transformer

```python
# Apply transformers at multiple resolutions
class MultiScaleTransformer(nn.Module):
    def forward(self, encoder_features):
        # Apply at H/8, H/16, H/32
        for feat in encoder_features:
            feat = self.transformer(feat)
        return encoder_features
```

## References

- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Transformer: Vaswani et al., "Attention Is All You Need" (2017)
- Vision Transformer: Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020)
