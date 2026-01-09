# Ablation Studies and Model Comparisons

This document provides exact commands and configurations for comparing different model variants and conducting ablation studies for the Paraformer ASR system.

## Model Variants

### 1. Encoder-Only Baseline
Simple transformer encoder with direct token prediction (no predictor/decoder separation).

**Configuration**: `configs/encoder_only.yaml`

**Training Command**:
```bash
python scripts/train.py --config configs/encoder_only.yaml --run-name encoder_only
```

**Expected Performance**:
- Token Accuracy: ~0.90 on toy sequences
- Training time: ~10 minutes (100 epochs, GPU)
- Memory usage: ~3GB GPU memory

### 2. Full Paraformer Architecture
Complete encoder-predictor-decoder architecture with alignment modeling.

**Configuration**: `configs/paraformer_base.yaml`

**Training Command**:
```bash
python scripts/train.py --config configs/paraformer_base.yaml --run-name paraformer_full
```

**Expected Performance**:
- Token Accuracy: ~0.95 on toy sequences
- Training time: ~15 minutes (100 epochs, GPU)
- Memory usage: ~4GB GPU memory

## Systematic Ablation Studies

### A. Encoder Architecture Comparison

Test different encoder architectures and depths.

**Commands**:
```bash
# 3-layer encoder
python scripts/train.py --config configs/paraformer_base.yaml --override model.encoder.num_layers=3 --run-name encoder_3layers

# 6-layer encoder (default)
python scripts/train.py --config configs/paraformer_base.yaml --run-name encoder_6layers

# 12-layer encoder
python scripts/train.py --config configs/paraformer_base.yaml --override model.encoder.num_layers=12 --run-name encoder_12layers

# Different hidden dimensions
python scripts/train.py --config configs/paraformer_base.yaml --override model.encoder.hidden_dim=256 --run-name encoder_dim256
python scripts/train.py --config configs/paraformer_base.yaml --override model.encoder.hidden_dim=1024 --run-name encoder_dim1024
```

**Expected Results**:
- 3 layers: Faster training, may underfit complex sequences
- 6 layers: Good balance of performance and efficiency
- 12 layers: Marginal improvement, much slower training
- Smaller hidden_dim: Faster but lower capacity
- Larger hidden_dim: Better performance, more memory usage

### B. Attention Head Analysis

Evaluate the impact of different numbers of attention heads.

**Commands**:
```bash
# 4 attention heads
python scripts/train.py --config configs/paraformer_base.yaml --override model.encoder.num_heads=4 model.decoder.num_heads=4 --run-name heads_4

# 8 attention heads (default)
python scripts/train.py --config configs/paraformer_base.yaml --run-name heads_8

# 16 attention heads
python scripts/train.py --config configs/paraformer_base.yaml --override model.encoder.num_heads=16 model.decoder.num_heads=16 --run-name heads_16
```

**Expected Results**:
- 4 heads: Slightly lower performance, faster training
- 8 heads: Good balance for most sequence lengths
- 16 heads: Marginal improvement, higher memory usage

### C. Predictor Module Ablation

Test the impact of the predictor module on alignment and performance.

**Commands**:
```bash
# No predictor (direct encoder-decoder)
python scripts/train.py --config configs/paraformer_base.yaml --override model.use_predictor=false --run-name no_predictor

# Simple predictor (1 layer)
python scripts/train.py --config configs/paraformer_base.yaml --override model.predictor.num_layers=1 --run-name predictor_1layer

# Default predictor (2 layers)
python scripts/train.py --config configs/paraformer_base.yaml --run-name predictor_2layers

# Complex predictor (4 layers)
python scripts/train.py --config configs/paraformer_base.yaml --override model.predictor.num_layers=4 --run-name predictor_4layers
```

**Expected Results**:
- No predictor: Lower performance on alignment-sensitive tasks
- 1 layer: Good baseline with minimal overhead
- 2 layers: Optimal balance for most cases
- 4 layers: Diminishing returns, increased complexity

### D. Sequence Length Sensitivity

Analyze performance across different sequence lengths and complexity.

**Commands**:
```bash
# Short sequences (max 100 frames, 20 tokens)
python scripts/train.py --config configs/paraformer_base.yaml --override data.max_feat_len=100 data.max_token_len=20 --run-name short_sequences

# Medium sequences (default: 300 frames, 60 tokens)
python scripts/train.py --config configs/paraformer_base.yaml --run-name medium_sequences

# Long sequences (500 frames, 100 tokens)
python scripts/train.py --config configs/paraformer_base.yaml --override data.max_feat_len=500 data.max_token_len=100 --run-name long_sequences

# Variable length training
python scripts/train.py --config configs/paraformer_base.yaml --override data.variable_length=true --run-name variable_length
```

**Expected Results**:
- Short sequences: Fast training, may not capture long-range dependencies
- Medium sequences: Good balance for most applications
- Long sequences: Better modeling of complex patterns, slower training
- Variable length: Most realistic, requires careful batching

### E. Loss Function Comparison

Compare different loss function combinations and weighting strategies.

**Commands**:
```bash
# Pure cross-entropy loss
python scripts/train.py --config configs/paraformer_base.yaml --override training.loss.predictor_weight=0.0 --run-name loss_ce_only

# Cross-entropy + predictor loss (equal weights)
python scripts/train.py --config configs/paraformer_base.yaml --override training.loss.predictor_weight=1.0 --run-name loss_equal_weights

# Cross-entropy + predictor loss (predictor weighted)
python scripts/train.py --config configs/paraformer_base.yaml --override training.loss.predictor_weight=2.0 --run-name loss_predictor_weighted

# With label smoothing
python scripts/train.py --config configs/paraformer_base.yaml --override training.loss.label_smoothing=0.1 --run-name loss_label_smoothing
```

**Expected Results**:
- CE only: Good baseline, may struggle with alignment
- Equal weights: Balanced training of both components
- Predictor weighted: Better alignment, may sacrifice token accuracy
- Label smoothing: More robust predictions, slower convergence

### F. Optimizer and Learning Rate Analysis

Test different optimization strategies and learning rates.

**Commands**:
```bash
# Adam optimizer
python scripts/train.py --config configs/paraformer_base.yaml --override training.optimizer=adam --run-name opt_adam

# AdamW optimizer (default)
python scripts/train.py --config configs/paraformer_base.yaml --run-name opt_adamw

# Different learning rates
python scripts/train.py --config configs/paraformer_base.yaml --override training.learning_rate=1e-5 --run-name lr_1e5
python scripts/train.py --config configs/paraformer_base.yaml --override training.learning_rate=1e-3 --run-name lr_1e3

# Learning rate scheduling
python scripts/train.py --config configs/paraformer_base.yaml --override training.scheduler=cosine --run-name sched_cosine
python scripts/train.py --config configs/paraformer_base.yaml --override training.scheduler=linear_warmup --run-name sched_warmup
```

**Expected Results**:
- Adam: Faster initial convergence, may be less stable
- AdamW: Better generalization with weight decay
- Low LR (1e-5): Slow but stable convergence
- High LR (1e-3): Fast initial progress, may be unstable
- Cosine scheduling: Smooth convergence to optimum
- Linear warmup: Stable training start, good for large models

## Dataset Complexity Ablations

### G. Synthetic Data Complexity

Test model performance on different levels of synthetic data complexity.

**Commands**:
```bash
# Simple patterns (short, regular sequences)
python scripts/train.py --config configs/paraformer_base.yaml --override data.complexity=simple --run-name data_simple

# Medium complexity (variable length, some noise)
python scripts/train.py --config configs/paraformer_base.yaml --override data.complexity=medium --run-name data_medium

# High complexity (long sequences, high variability)
python scripts/train.py --config configs/paraformer_base.yaml --override data.complexity=high --run-name data_high

# With feature noise
python scripts/train.py --config configs/paraformer_base.yaml --override data.feature_noise=0.1 --run-name data_noisy
```

### H. Vocabulary Size Impact

Evaluate model performance with different vocabulary sizes.

**Commands**:
```bash
# Small vocabulary (26 characters)
python scripts/train.py --config configs/paraformer_base.yaml --override data.vocab_size=26 --run-name vocab_26

# Medium vocabulary (100 characters, default)
python scripts/train.py --config configs/paraformer_base.yaml --run-name vocab_100

# Large vocabulary (1000 characters)
python scripts/train.py --config configs/paraformer_base.yaml --override data.vocab_size=1000 --run-name vocab_1000
```

**Expected Results**:
- Small vocab: Fast training, limited expressiveness
- Medium vocab: Good balance for character-level modeling
- Large vocab: More expressive, slower training, may overfit

## Decoding Strategy Comparison

### I. Decoding Algorithm Analysis

Compare different decoding strategies for inference.

**Commands**:
```bash
# Greedy decoding (default)
python scripts/eval.py --checkpoint runs/paraformer_full/checkpoints/best.pt --decode-method greedy --run-name decode_greedy

# Beam search (beam size 4)
python scripts/eval.py --checkpoint runs/paraformer_full/checkpoints/best.pt --decode-method beam --beam-size 4 --run-name decode_beam4

# Beam search (beam size 8)
python scripts/eval.py --checkpoint runs/paraformer_full/checkpoints/best.pt --decode-method beam --beam-size 8 --run-name decode_beam8

# With length normalization
python scripts/eval.py --checkpoint runs/paraformer_full/checkpoints/best.pt --decode-method beam --beam-size 4 --length-normalize --run-name decode_beam_norm
```

**Expected Results**:
- Greedy: Fast inference, may miss optimal sequences
- Beam search: Better quality, slower inference
- Larger beam: Marginal improvement, much slower
- Length normalization: Better handling of variable-length outputs

## Evaluation and Comparison

### Running Comprehensive Evaluations

After training, evaluate all models:

```bash
# Evaluate specific model
python scripts/eval.py --checkpoint runs/paraformer_full/checkpoints/best.pt --run-name eval_full

# Evaluate all models in runs directory
python scripts/eval_all.py --runs-dir runs/ --output results/evaluation_summary.csv

# Test on different sequence lengths
python scripts/eval_lengths.py --checkpoint runs/paraformer_full/checkpoints/best.pt --output results/length_analysis.csv
```

### Generating Comparison Reports

Create comprehensive comparison reports:

```bash
# Generate summary of all experiments
python scripts/summarize.py --runs-dir runs/ --output results/

# Create visualization plots
python scripts/plot_results.py --results results/summary.csv --output results/plots/

# Generate attention visualization
python scripts/visualize_attention.py --checkpoint runs/paraformer_full/checkpoints/best.pt --output results/attention/

# Generate alignment analysis
python scripts/analyze_alignment.py --checkpoint runs/paraformer_full/checkpoints/best.pt --output results/alignment/
```

### Expected Output Files

After running ablations, you should have:

```
results/
├── summary.csv              # Tabular results for all experiments
├── summary.md               # Markdown summary with key findings
├── plots/
│   ├── loss_curves.png      # Training/validation loss curves
│   ├── accuracy_comparison.png # Token/sequence accuracy comparisons
│   ├── training_time.png    # Training time vs performance
│   └── memory_usage.png     # Memory usage analysis
├── attention/
│   ├── attention_heads.png  # Attention head visualizations
│   └── attention_patterns.png # Attention pattern analysis
└── alignment/
    ├── predictor_analysis.png # Predictor output analysis
    └── alignment_quality.png  # Alignment quality metrics
```

## Key Findings Template

Based on the ablation studies, typical findings include:

### Model Architecture
- **Winner**: Full Paraformer (6 encoder layers, 6 decoder layers, 2 predictor layers)
- **Performance**: +5% token accuracy over encoder-only baseline
- **Trade-off**: 50% longer training time, 33% more memory

### Attention Configuration
- **Winner**: 8 attention heads with 512 hidden dimension
- **Performance**: Good balance of expressiveness and efficiency
- **Alternative**: 4 heads for faster training with minimal performance loss

### Loss Function Strategy
- **Winner**: Cross-entropy + predictor loss (1:1 weighting)
- **Performance**: Best alignment quality with good token accuracy
- **Alternative**: Predictor-weighted (1:2) for alignment-critical applications

### Learning Rate and Optimization
- **Winner**: AdamW with 1e-4 learning rate and cosine scheduling
- **Performance**: Stable convergence with good generalization
- **Alternative**: Linear warmup for very large models or datasets

### Sequence Length Handling
- **Winner**: Variable-length training with proper masking
- **Performance**: Best real-world applicability
- **Trade-off**: More complex batching and slightly slower training

## Reproducibility Commands

To reproduce the exact results from the documentation:

```bash
# Set up environment
python scripts/setup_env.py --seed 42

# Run core comparison
bash scripts/run_core_ablations.sh

# Run extended ablations
bash scripts/run_extended_ablations.sh

# Generate final report
python scripts/final_report.py --experiments all_ablations --output final_results/
```

## Performance Benchmarks

### Training Speed Benchmarks
```bash
# Benchmark training speed across configurations
python scripts/benchmark_training.py --configs configs/ --output results/training_benchmarks.csv

# Memory profiling
python scripts/profile_memory.py --config configs/paraformer_base.yaml --output results/memory_profile.json
```

### Inference Speed Benchmarks
```bash
# Benchmark inference speed
python scripts/benchmark_inference.py --checkpoint runs/paraformer_full/checkpoints/best.pt --output results/inference_benchmarks.csv

# Service performance testing
python scripts/test_service_performance.py --checkpoint runs/paraformer_full/checkpoints/best.pt --port 8000 --output results/service_benchmarks.csv
```

All commands include proper seed setting and deterministic training for reproducible results.