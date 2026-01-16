# U-Net Transformer Segmentation Tutorial

This tutorial walks you through using and extending the U-Net Transformer Segmentation project.

## Getting Started

### Installation

```bash
cd unet_transformer_seg
pip install -r requirements.txt
```

### Quick Test

```bash
# Run a quick smoke test
python -m pytest tests/test_smoke.py -v
```

## Tutorial 1: Training Your First Model

### Step 1: Understand the Configuration

Open `configs/unet_baseline.yaml`:

```yaml
model:
  name: unet
  num_classes: 4
  base_channels: 64
  
data:
  image_size: 256
  num_samples: 1000
  
training:
  batch_size: 8
  learning_rate: 0.0001
  num_epochs: 100
```

### Step 2: Train the Baseline

```bash
python scripts/train.py --config configs/unet_baseline.yaml --run-name my_first_model
```

Watch the training progress:
- Loss should decrease steadily
- IoU should increase
- Check `runs/my_first_model/` for logs and checkpoints

### Step 3: Evaluate the Model

```bash
python scripts/evaluate.py --checkpoint runs/my_first_model/checkpoints/best.pt
```

### Step 4: Run Inference

```bash
python scripts/inference.py \
    --checkpoint runs/my_first_model/checkpoints/best.pt \
    --image path/to/image.jpg \
    --output prediction.png
```

## Tutorial 2: Understanding the Code

### Data Pipeline

Let's trace how data flows through the system:

```python
# 1. Dataset generates samples
from src.data.toy_shapes import ToyShapesDataset

dataset = ToyShapesDataset(num_samples=100, image_size=256, num_classes=4)
image, mask = dataset[0]
print(f"Image shape: {image.shape}")  # [3, 256, 256]
print(f"Mask shape: {mask.shape}")    # [256, 256]

# 2. DataLoader batches samples
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
for images, masks in dataloader:
    print(f"Batch images: {images.shape}")  # [8, 3, 256, 256]
    print(f"Batch masks: {masks.shape}")    # [8, 256, 256]
    break
```

### Model Forward Pass

```python
# 3. Model processes batch
from src.models.unet import UNet

model = UNet(in_channels=3, num_classes=4, base_channels=64)
logits = model(images)
print(f"Logits shape: {logits.shape}")  # [8, 4, 256, 256]

# 4. Get predictions
predictions = logits.argmax(dim=1)
print(f"Predictions shape: {predictions.shape}")  # [8, 256, 256]
```

### Loss Computation

```python
# 5. Compute loss
from src.losses.dice_loss import DiceLoss

criterion = DiceLoss()
loss = criterion(logits, masks)
print(f"Loss: {loss.item()}")

# 6. Backpropagation
loss.backward()
```

## Tutorial 3: Experimenting with Hyperparameters

### Experiment 1: Different Learning Rates

```bash
# Low learning rate
python scripts/train.py --config configs/unet_baseline.yaml \
    --override training.learning_rate=0.00001 \
    --run-name lr_1e5

# High learning rate
python scripts/train.py --config configs/unet_baseline.yaml \
    --override training.learning_rate=0.001 \
    --run-name lr_1e3
```

Compare results:
```bash
python scripts/summarize.py --runs-dir runs/ --output results/
```

### Experiment 2: Model Capacity

```bash
# Smaller model
python scripts/train.py --config configs/unet_baseline.yaml \
    --override model.base_channels=32 \
    --run-name small_model

# Larger model
python scripts/train.py --config configs/unet_baseline.yaml \
    --override model.base_channels=128 \
    --run-name large_model
```

### Experiment 3: Data Augmentation

```bash
# No augmentation
python scripts/train.py --config configs/unet_baseline.yaml \
    --override data.augmentation.enabled=false \
    --run-name no_aug

# Heavy augmentation
python scripts/train.py --config configs/unet_baseline.yaml \
    --override data.augmentation.rotation_degrees=30 \
    --override data.augmentation.flip_prob=0.7 \
    --run-name heavy_aug
```

## Tutorial 4: Adding Custom Components

### Custom Loss Function

Create `src/losses/focal_loss.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # pred: [B, C, H, W]
        # target: [B, H, W]
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

Use it in training:

```python
from src.losses.focal_loss import FocalLoss

criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### Custom Data Augmentation

Create `src/data/custom_transforms.py`:

```python
import torch
import random

class RandomNoise:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
    
    def __call__(self, image, mask):
        if random.random() < 0.5:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            image = torch.clamp(image, 0, 1)
        return image, mask
```

### Custom Metric

Create `src/metrics/custom_metrics.py`:

```python
import torch

def boundary_iou(pred, target):
    """Compute IoU only at object boundaries."""
    # Extract boundaries using morphological operations
    pred_boundary = extract_boundary(pred)
    target_boundary = extract_boundary(target)
    
    intersection = (pred_boundary & target_boundary).sum()
    union = (pred_boundary | target_boundary).sum()
    
    return intersection / (union + 1e-7)

def extract_boundary(mask):
    # Simple boundary extraction
    kernel = torch.ones(1, 1, 3, 3)
    dilated = F.conv2d(mask.float().unsqueeze(1), kernel, padding=1) > 0
    eroded = F.conv2d(mask.float().unsqueeze(1), kernel, padding=1) == 9
    boundary = dilated & ~eroded
    return boundary.squeeze(1)
```

## Tutorial 5: Using Real Datasets

### Preparing Your Data

Organize your data:
```
data/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── masks/
    ├── img001.png
    ├── img002.png
    └── ...
```

### Create Custom Dataset

Create `src/data/custom_dataset.py`:

```python
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        mask = Image.open(mask_path).convert('L')
        
        # Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask
```

### Update Training Script

```python
from src.data.custom_dataset import CustomSegmentationDataset

# Replace toy dataset
train_dataset = CustomSegmentationDataset(
    image_dir='data/train/images',
    mask_dir='data/train/masks',
    transform=train_transforms
)

val_dataset = CustomSegmentationDataset(
    image_dir='data/val/images',
    mask_dir='data/val/masks',
    transform=val_transforms
)
```

## Tutorial 6: Debugging Common Issues

### Issue 1: Loss is NaN

**Symptoms**: Loss becomes NaN after a few iterations

**Solutions**:
```python
# 1. Check for NaN in data
assert not torch.isnan(images).any(), "NaN in input images"
assert not torch.isnan(masks).any(), "NaN in masks"

# 2. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Reduce learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 4. Check loss computation
if torch.isnan(loss):
    print("NaN loss detected!")
    print(f"Logits range: [{logits.min()}, {logits.max()}]")
    print(f"Masks range: [{masks.min()}, {masks.max()}]")
```

### Issue 2: Model Predicts All Background

**Symptoms**: Model always predicts class 0

**Solutions**:
```python
# 1. Check class distribution
class_counts = torch.bincount(masks.flatten())
print(f"Class distribution: {class_counts}")

# 2. Use weighted loss
class_weights = 1.0 / class_counts.float()
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 3. Use focal loss
from src.losses.focal_loss import FocalLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### Issue 3: Out of Memory

**Symptoms**: CUDA out of memory error

**Solutions**:
```python
# 1. Reduce batch size
batch_size = 4  # Instead of 16

# 2. Use gradient accumulation
accumulation_steps = 4
for i, (images, masks) in enumerate(dataloader):
    loss = train_step(images, masks) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    logits = model(images)
    loss = criterion(logits, masks)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Next Steps

1. **Read the Architecture Guide**: `docs/ARCHITECTURE.md`
2. **Study Tensor Contracts**: `docs/CONTRACTS.md`
3. **Run Ablation Studies**: `docs/ABLATIONS.md`
4. **Explore the Code**: Start with `src/models/unet.py`
5. **Join the Community**: Share your experiments and results!
