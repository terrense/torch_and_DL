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


## Troubleshooting Guide

### Common Shape Mismatches

#### Problem: "Expected shape [B,3,H,W] but got [B,H,W,3]"
**Cause**: Image tensor has channels in wrong position (HWC instead of CHW format)

**Solution**:
```python
# Convert from HWC to CHW
if image.shape[-1] == 3:
    image = image.permute(0, 3, 1, 2)  # [B,H,W,3] -> [B,3,H,W]
```

**Prevention**: Always use `torchvision.transforms.ToTensor()` which handles this conversion automatically.

#### Problem: "RuntimeError: Sizes of tensors must match except in dimension 1"
**Cause**: Skip connection dimensions don't match during upsampling in U-Net

**Solution**:
```python
# In UpBlock, ensure proper spatial alignment
def forward(self, x, skip):
    x = self.upsample(x)
    # Handle size mismatch due to odd dimensions
    if x.shape[2:] != skip.shape[2:]:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
    x = torch.cat([x, skip], dim=1)
    return self.conv(x)
```

**Prevention**: Use even-sized input dimensions (e.g., 256×256 instead of 255×255).

#### Problem: "Expected 4D tensor but got 3D tensor"
**Cause**: Missing batch dimension in single-image inference

**Solution**:
```python
# Add batch dimension for single image
if image.dim() == 3:
    image = image.unsqueeze(0)  # [3,H,W] -> [1,3,H,W]
```

**Prevention**: Always maintain batch dimension, even for single samples.

### Common Mask Issues

#### Problem: "Target mask contains values outside [0, num_classes-1]"
**Cause**: Mask values not properly normalized or contain invalid labels

**Solution**:
```python
# Validate and clip mask values
mask = torch.clamp(mask, 0, num_classes - 1)

# Check for invalid values before training
assert mask.min() >= 0 and mask.max() < num_classes, \
    f"Mask contains invalid values: min={mask.min()}, max={mask.max()}"
```

**Prevention**: Validate dataset generation and ensure mask values are integers in valid range.

#### Problem: "Dice loss returns NaN"
**Cause**: Division by zero when both prediction and target are empty (all zeros)

**Solution**:
```python
# Add epsilon to prevent division by zero
def dice_loss(pred, target, epsilon=1e-7):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return 1 - dice
```

**Prevention**: Always include epsilon in dice coefficient calculations.

#### Problem: "Mask and image dimensions don't match after augmentation"
**Cause**: Transforms applied to image but not mask, or vice versa

**Solution**:
```python
# Apply same transform to both image and mask
class PairedTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image, mask):
        # Use same random state for both
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        image = self.transform(image)
        torch.manual_seed(seed)
        mask = self.transform(mask)
        return image, mask
```

**Prevention**: Use paired transforms that maintain spatial correspondence.

### Transformer-Specific Issues

#### Problem: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
**Cause**: Hidden dimension not divisible by number of attention heads

**Solution**:
```python
# Ensure hidden_dim is divisible by num_heads
assert hidden_dim % num_heads == 0, \
    f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
```

**Prevention**: Choose compatible hidden_dim and num_heads (e.g., 512 and 8).

#### Problem: "Transformer output shape doesn't match CNN input"
**Cause**: Incorrect reshaping between [B,T,D] and [B,C,H,W] formats

**Solution**:
```python
# Proper reshaping with dimension tracking
def reshape_for_transformer(x):
    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)  # [B,C,H,W] -> [B,H*W,C]
    return x, (H, W)

def reshape_from_transformer(x, spatial_dims):
    B, T, C = x.shape
    H, W = spatial_dims
    assert T == H * W, f"Sequence length {T} doesn't match spatial dims {H}×{W}"
    x = x.transpose(1, 2).reshape(B, C, H, W)
    return x
```

**Prevention**: Always track and validate spatial dimensions during reshaping.

#### Problem: "Attention weights sum to NaN"
**Cause**: Attention scores overflow or underflow before softmax

**Solution**:
```python
# Use scaled dot-product attention with proper scaling
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weights = F.softmax(scores, dim=-1)
    
    # Check for NaN
    if torch.isnan(attn_weights).any():
        print(f"NaN in attention! Scores range: [{scores.min()}, {scores.max()}]")
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    
    return torch.matmul(attn_weights, v)
```

**Prevention**: Always scale attention scores by sqrt(d_k) and check for extreme values.

### Training Issues

#### Problem: "Loss is NaN after a few iterations"
**Cause**: Exploding gradients or learning rate too high

**Solution**:
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Check for NaN in loss
if torch.isnan(loss):
    print("NaN loss detected! Skipping batch.")
    continue
```

**Prevention**: Use gradient clipping and start with conservative learning rates.

#### Problem: "Model predicts all background (class 0)"
**Cause**: Class imbalance in dataset or improper loss weighting

**Solution**:
```python
# Use weighted loss for class imbalance
class_weights = torch.tensor([0.1, 1.0, 1.0, ...])  # Lower weight for background
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Or use focal loss for hard examples
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()
```

**Prevention**: Analyze class distribution and use appropriate loss weighting.

#### Problem: "Validation metrics don't improve but training loss decreases"
**Cause**: Overfitting or train/val data distribution mismatch

**Solution**:
```python
# Add regularization
model = UNet(dropout=0.2)  # Add dropout
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Use data augmentation
transforms = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.2, contrast=0.2)
])

# Early stopping
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping triggered")
        break
```

**Prevention**: Use proper regularization and monitor train/val gap.

### Memory Issues

#### Problem: "CUDA out of memory"
**Cause**: Batch size too large or model too big for available GPU memory

**Solution**:
```python
# Reduce batch size
batch_size = 4  # Instead of 16

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = train_step(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Enable gradient checkpointing
model.use_checkpoint = True
```

**Prevention**: Profile memory usage and adjust batch size accordingly.

#### Problem: "Memory usage increases over time"
**Cause**: Memory leak from retaining computation graphs

**Solution**:
```python
# Detach tensors when not needed for backprop
with torch.no_grad():
    val_loss = validate(model, val_loader)

# Clear cache periodically
if epoch % 10 == 0:
    torch.cuda.empty_cache()

# Don't accumulate losses in lists
# Bad: losses.append(loss)
# Good: losses.append(loss.item())
```

**Prevention**: Use `torch.no_grad()` for inference and detach tensors appropriately.

### Debugging Tips

#### Enable Detailed Error Messages
```python
# Set environment variable for better error messages
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
```

#### Add Comprehensive Logging
```python
# Log tensor statistics
def log_tensor_stats(tensor, name):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"min={tensor.min():.4f}, max={tensor.max():.4f}, "
          f"mean={tensor.mean():.4f}, std={tensor.std():.4f}, "
          f"nan={torch.isnan(tensor).sum()}, inf={torch.isinf(tensor).sum()}")

# Use in training loop
log_tensor_stats(images, "input_images")
log_tensor_stats(logits, "model_output")
log_tensor_stats(loss, "loss")
```

#### Visualize Intermediate Outputs
```python
# Save intermediate feature maps
def visualize_features(features, save_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < features.shape[1]:
            ax.imshow(features[0, i].detach().cpu().numpy())
            ax.axis('off')
    plt.savefig(save_path)
    plt.close()

# Use during debugging
features = model.encode(images)
visualize_features(features, 'debug_features.png')
```
