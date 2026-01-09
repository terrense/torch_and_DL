# Ablation Studies and Model Comparisons

This document provides exact commands and configurations for comparing different model variants and conducting ablation studies.

## Model Variants

### 1. U-Net Baseline
Pure convolutional encoder-decoder architecture without transformer components.

**Configuration**: `configs/unet_baseline.yaml`

**Training Command**:
```bash
python scripts/train.py --config configs/unet_baseline.yaml --run-name unet_baseline
```

**Expected Performance**:
- IoU: ~0.85 on toy shapes dataset
- Training time: ~8 minutes (100 epochs, GPU)
- Memory usage: ~2GB GPU memory

### 2. U-Net + Transformer Hybrid
U-Net with transformer layers at the bottleneck for enhanced feature processing.

**Configuration**: `configs/unet_transformer.yaml`

**Training Command**:
```bash
python scripts/train.py --config configs/unet_transformer.yaml --run-name unet_transformer
```

**Expected Performance**:
- IoU: ~0.87 on toy shapes dataset
- Training time: ~12 minutes (100 epochs, GPU)
- Memory usage: ~3GB GPU memory

## Systematic Ablation Studies

### A. Transformer Layer Count

Test the effect of different numbers of transformer layers at the bottleneck.

**Commands**:
```bash
# 1 transformer layer
python scripts/train.py --config configs/unet_transformer.yaml --override model.transformer.num_layers=1 --run-name transformer_1layer

# 2 transformer layers
python scripts/train.py --config configs/unet_transformer.yaml --override model.transformer.num_layers=2 --run-name transformer_2layers

# 4 transformer layers (default)
python scripts/train.py --config configs/unet_transformer.yaml --run-name transformer_4layers

# 8 transformer layers
python scripts/train.py --config configs/unet_transformer.yaml --override model.transformer.num_layers=8 --run-name transformer_8layers
```

**Expected Results**:
- 1 layer: Minimal improvement over baseline
- 2-4 layers: Optimal performance/efficiency trade-off
- 8+ layers: Diminishing returns, increased training time

### B. Attention Head Count

Evaluate the impact of different numbers of attention heads.

**Commands**:
```bash
# 4 attention heads
python scripts/train.py --config configs/unet_transformer.yaml --override model.transformer.num_heads=4 --run-name heads_4

# 8 attention heads (default)
python scripts/train.py --config configs/unet_transformer.yaml --run-name heads_8

# 16 attention heads
python scripts/train.py --config configs/unet_transformer.yaml --override model.transformer.num_heads=16 --run-name heads_16
```

**Expected Results**:
- 4 heads: Slightly lower performance
- 8 heads: Good balance of performance and efficiency
- 16 heads: Marginal improvement, higher memory usage

### C. Loss Function Comparison

Compare different loss function combinations for segmentation.

**Commands**:
```bash
# Pure Dice Loss
python scripts/train.py --config configs/unet_baseline.yaml --override training.loss.name=dice --run-name loss_dice

# Pure BCE Loss
python scripts/train.py --config configs/unet_baseline.yaml --override training.loss.name=bce --run-name loss_bce

# BCE + Dice (equal weights)
python scripts/train.py --config configs/unet_baseline.yaml --override training.loss.dice_weight=0.5 training.loss.bce_weight=0.5 --run-name loss_bce_dice_equal

# BCE + Dice (dice weighted)
python scripts/train.py --config configs/unet_baseline.yaml --override training.loss.dice_weight=0.7 training.loss.bce_weight=0.3 --run-name loss_dice_weighted
```

**Expected Results**:
- Pure Dice: Good for class imbalance, may be unstable
- Pure BCE: Stable training, may struggle with imbalance
- Combined: Best overall performance and stability

### D. Data Augmentation Impact

Test the effect of different augmentation strategies.

**Commands**:
```bash
# No augmentation
python scripts/train.py --config configs/unet_baseline.yaml --override data.augmentation.enabled=false --run-name no_aug

# Basic augmentation (flips only)
python scripts/train.py --config configs/unet_baseline.yaml --override data.augmentation.rotation=false data.augmentation.scale=false --run-name basic_aug

# Full augmentation (default)
python scripts/train.py --config configs/unet_baseline.yaml --run-name full_aug

# Heavy augmentation
python scripts/train.py --config configs/unet_baseline.yaml --override data.augmentation.noise_std=0.1 data.augmentation.blur_prob=0.3 --run-name heavy_aug
```

**Expected Results**:
- No augmentation: Overfitting on toy data
- Basic augmentation: Good baseline performance
- Full augmentation: Best generalization
- Heavy augmentation: May hurt performance on simple toy data

### E. Learning Rate Sensitivity

Analyze the impact of different learning rates on convergence.

**Commands**:
```bash
# Low learning rate
python scripts/train.py --config configs/unet_baseline.yaml --override training.learning_rate=1e-5 --run-name lr_1e5

# Default learning rate
python scripts/train.py --config configs/unet_baseline.yaml --run-name lr_1e4

# High learning rate
python scripts/train.py --config configs/unet_baseline.yaml --override training.learning_rate=1e-3 --run-name lr_1e3

# Very high learning rate
python scripts/train.py --config configs/unet_baseline.yaml --override training.learning_rate=1e-2 --run-name lr_1e2
```

**Expected Results**:
- 1e-5: Slow convergence, may not reach optimal performance
- 1e-4: Good default choice
- 1e-3: Faster initial convergence, may be unstable
- 1e-2: Likely unstable or divergent training

## Dataset Complexity Ablations

### F. Shape Complexity

Test model performance on different levels of shape complexity.

**Commands**:
```bash
# Simple shapes (circles, squares only)
python scripts/train.py --config configs/unet_baseline.yaml --override data.shape_types=['circle','square'] --run-name simple_shapes

# Medium complexity (add triangles)
python scripts/train.py --config configs/unet_baseline.yaml --override data.shape_types=['circle','square','triangle'] --run-name medium_shapes

# High complexity (all shapes)
python scripts/train.py --config configs/unet_baseline.yaml --run-name complex_shapes

# With occlusion
python scripts/train.py --config configs/unet_baseline.yaml --override data.occlusion_prob=0.3 --run-name with_occlusion
```

### G. Noise Robustness

Evaluate model robustness to different noise levels.

**Commands**:
```bash
# Clean data
python scripts/train.py --config configs/unet_baseline.yaml --override data.noise_std=0.0 --run-name clean_data

# Low noise
python scripts/train.py --config configs/unet_baseline.yaml --override data.noise_std=0.05 --run-name low_noise

# Medium noise (default)
python scripts/train.py --config configs/unet_baseline.yaml --run-name medium_noise

# High noise
python scripts/train.py --config configs/unet_baseline.yaml --override data.noise_std=0.15 --run-name high_noise
```

## Evaluation and Comparison

### Running Evaluations

After training, evaluate all models:

```bash
# Evaluate specific model
python scripts/eval.py --checkpoint runs/unet_baseline/checkpoints/best.pt --run-name eval_baseline

# Evaluate all models in runs directory
python scripts/eval_all.py --runs-dir runs/ --output results/evaluation_summary.csv
```

### Generating Comparison Reports

Create comprehensive comparison reports:

```bash
# Generate summary of all experiments
python scripts/summarize.py --runs-dir runs/ --output results/

# Create visualization plots
python scripts/plot_results.py --results results/summary.csv --output results/plots/

# Generate markdown report
python scripts/generate_report.py --results results/summary.csv --output results/ablation_report.md
```

### Expected Output Files

After running ablations, you should have:

```
results/
├── summary.csv              # Tabular results for all experiments
├── summary.md               # Markdown summary with key findings
├── plots/
│   ├── loss_curves.png      # Training/validation loss curves
│   ├── metric_comparison.png # IoU/Dice score comparisons
│   └── training_time.png    # Training time vs performance
└── ablation_report.md       # Detailed analysis and recommendations
```

## Key Findings Template

Based on the ablation studies, typical findings include:

### Model Architecture
- **Winner**: U-Net + Transformer (4 layers, 8 heads)
- **Performance**: +2-3% IoU improvement over baseline
- **Trade-off**: 50% longer training time, 50% more memory

### Loss Functions
- **Winner**: BCE + Dice (0.5/0.5 weighting)
- **Performance**: Most stable training with best final metrics
- **Alternative**: Dice-weighted (0.7/0.3) for class imbalanced datasets

### Data Augmentation
- **Winner**: Full augmentation suite
- **Performance**: +1-2% IoU improvement through better generalization
- **Note**: Heavy augmentation may hurt performance on simple synthetic data

### Learning Rate
- **Winner**: 1e-4 with cosine annealing
- **Performance**: Good convergence speed and stability
- **Alternative**: 1e-3 with careful monitoring for faster experiments

## Reproducibility Commands

To reproduce the exact results from the paper/documentation:

```bash
# Set up environment
python scripts/setup_env.py --seed 42

# Run core comparison
bash scripts/run_core_ablations.sh

# Generate final report
python scripts/final_report.py --experiments core_ablations --output final_results/
```

All commands include proper seed setting and deterministic training for reproducible results.