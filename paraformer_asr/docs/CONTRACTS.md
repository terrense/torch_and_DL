# Tensor Contracts Documentation

This document defines explicit tensor shapes, dtypes, and value ranges for all major components in the Paraformer ASR project.

## Notation

- `B` = Batch size
- `T` = Sequence length (variable)
- `F` = Feature dimension (input features)
- `D` = Hidden dimension
- `V` = Vocabulary size
- `H` = Number of attention heads
- `L` = Number of layers

## Data Pipeline Contracts

### ToySeq2SeqDataset

| Method | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `__getitem__` | index | scalar | int | [0, len) | features | [T_feat, F] | float32 | [-2, 2] | Speech-like features |
| | | | | | tokens | [T_token] | long | [0, V-1] | Token sequence |
| | | | | | feat_len | scalar | int | [1, T_feat] | Actual feature length |
| | | | | | token_len | scalar | int | [1, T_token] | Actual token length |
| `generate_sequence` | length | int | - | [10, 500] | features | [T, F] | float32 | [-2, 2] | Correlated features |
| | | | | | tokens | [S] | long | [0, V-1] | Corresponding tokens |

### Tokenizer

| Method | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `encode` | text | str | - | - | tokens | [T] | long | [0, V-1] | Includes special tokens |
| `decode` | tokens | [T] | long | [0, V-1] | text | str | - | - | Removes special tokens |
| `batch_encode` | texts | List[str] | - | - | tokens | [B, T] | long | [0, V-1] | Padded sequences |
| | | | | | lengths | [B] | int | [1, T] | Actual lengths |

### DataLoader Collation

| Function | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `collate_fn` | features | List[[T_i, F]] | float32 | [-2, 2] | batch | [B, T_max, F] | float32 | [-2, 2] | Padded to max length |
| | tokens | List[[S_i]] | long | [0, V-1] | batch | [B, S_max] | long | [0, V-1] | Padded with pad_token |
| | feat_lens | List[int] | int | [1, T_i] | lengths | [B] | int | [1, T_max] | Actual lengths |
| | token_lens | List[int] | int | [1, S_i] | lengths | [B] | int | [1, S_max] | Actual lengths |

## Model Architecture Contracts

### Transformer Components

| Component | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|-----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `MultiHeadAttention` | query | [B, T, D] | float32 | any | output | [B, T, D] | float32 | any | Self or cross attention |
| | key | [B, S, D] | float32 | any | | | | | S=T for self-attention |
| | value | [B, S, D] | float32 | any | | | | | Same as key |
| | mask | [B, T, S] | bool | {0, 1} | | | | | Optional attention mask |
| `FeedForward` | x | [B, T, D] | float32 | any | x | [B, T, D] | float32 | any | Linear→ReLU→Linear |
| `TransformerBlock` | x | [B, T, D] | float32 | any | x | [B, T, D] | float32 | any | MHA + FFN + residual |
| | mask | [B, T, T] | bool | {0, 1} | | | | | Optional self-attention mask |
| `PositionalEncoding` | x | [B, T, D] | float32 | any | x | [B, T, D] | float32 | any | Adds position embeddings |

### Encoder Architecture

| Method | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `forward` | features | [B, T, F] | float32 | [-2, 2] | encoded | [B, T, D] | float32 | any | Full encoding |
| | mask | [B, T] | bool | {0, 1} | | | | | Padding mask |
| `encode_layer` | x | [B, T, D] | float32 | any | x | [B, T, D] | float32 | any | Single layer |
| | mask | [B, T, T] | bool | {0, 1} | | | | | Self-attention mask |

### Predictor Module

| Method | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `forward` | encoder_features | [B, T, D] | float32 | any | predictions | [B, T, 1] | float32 | any | Raw predictions |
| | padding_mask | [B, T] | bool | {0, 1} | probabilities | [B, T, 1] | float32 | [0, 1] | Sigmoid/softplus probs |
| `compute_alignment_loss` | predictions | [B, T, 1] | float32 | any | loss | scalar | float32 | [0, ∞) | Alignment supervision |
| | target_alignments | [B, T] or [B, T, 1] | float32 | [0, 1] | | | | | Ground truth alignment |
| | padding_mask | [B, T] | bool | {0, 1} | | | | | Valid position mask |
| `extract_token_positions` | probabilities | [B, T, 1] | float32 | [0, 1] | positions | List[List[int]] | int | [0, T-1] | Token boundary positions |
| | padding_mask | [B, T] | bool | {0, 1} | | | | | Valid position mask |
| | threshold | scalar | float32 | [0, 1] | | | | | Minimum boundary prob |
| `generate_alignment_targets` | token_positions | List[List[int]] | int | [0, T-1] | targets | [B, T] | float32 | [0, 1] | Training targets |
| | sequence_length | scalar | int | [1, ∞) | | | | | Feature sequence length |
| | method | str | - | - | | | | | 'boundary' or 'gaussian' |
| `visualize_alignment` | probabilities | [B, T, 1] | float32 | [0, 1] | viz_data | Dict | - | - | Visualization data |
| | token_positions | List[List[int]] | int | [0, T-1] | | | | | Optional ground truth |
| | feature_lengths | [B] | int | [1, T] | | | | | Valid sequence lengths |

### Decoder Architecture

| Method | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `forward` | encoder_out | [B, T_enc, D] | float32 | any | logits | [B, T_dec, V] | float32 | any | Token logits |
| | predictor_out | [B, T_enc, 1] | float32 | [0, ∞) | | | | | Alignment info |
| | target | [B, T_dec] | long | [0, V-1] | | | | | Target tokens (training) |
| | enc_mask | [B, T_enc] | bool | {0, 1} | | | | | Encoder padding mask |
| | dec_mask | [B, T_dec] | bool | {0, 1} | | | | | Decoder padding mask |
| `decode_step` | encoder_out | [B, T_enc, D] | float32 | any | logits | [B, 1, V] | float32 | any | Single decoding step |
| | prev_tokens | [B, T_prev] | long | [0, V-1] | | | | | Previous tokens |
| | step | int | - | [0, T_dec) | | | | | Current step index |

### Complete Paraformer Model

| Method | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `forward` | features | [B, T_feat, F] | float32 | [-2, 2] | logits | [B, T_token, V] | float32 | any | Training forward pass |
| | tokens | [B, T_token] | long | [0, V-1] | | | | | Target tokens |
| | feat_mask | [B, T_feat] | bool | {0, 1} | | | | | Feature padding mask |
| | token_mask | [B, T_token] | bool | {0, 1} | | | | | Token padding mask |
| `encode` | features | [B, T_feat, F] | float32 | [-2, 2] | encoded | [B, T_feat, D] | float32 | any | Encoder only |
| | mask | [B, T_feat] | bool | {0, 1} | predictor | [B, T_feat, 1] | float32 | [0, ∞) | With predictor |
| `decode_greedy` | encoder_out | [B, T_feat, D] | float32 | any | tokens | [B, T_out] | long | [0, V-1] | Greedy decoding |
| | predictor_out | [B, T_feat, 1] | float32 | [0, ∞) | | | | | Alignment guidance |
| | max_len | int | - | [1, ∞) | | | | | Maximum output length |

## Loss Functions Contracts

| Loss Function | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|---------------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `MaskedCrossEntropy` | logits | [B, T, V] | float32 | any | loss | scalar | float32 | [0, ∞) | Ignores padded positions |
| | targets | [B, T] | long | [0, V-1] | | | | | Target token IDs |
| | mask | [B, T] | bool | {0, 1} | | | | | Valid position mask |
| `PredictorLoss` | predictions | [B, T, 1] | float32 | [0, ∞) | loss | scalar | float32 | [0, ∞) | Duration prediction loss |
| | targets | [B, T] | float32 | [0, ∞) | | | | | Target durations |
| | mask | [B, T] | bool | {0, 1} | | | | | Valid position mask |

## Metrics Contracts

| Metric | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|--------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `TokenAccuracy` | logits | [B, T, V] | float32 | any | accuracy | scalar | float32 | [0, 1] | Per-token accuracy |
| | targets | [B, T] | long | [0, V-1] | | | | | Target tokens |
| | mask | [B, T] | bool | {0, 1} | | | | | Valid positions |
| `SequenceAccuracy` | predictions | [B, T] | long | [0, V-1] | accuracy | scalar | float32 | [0, 1] | Exact sequence match |
| | targets | [B, T] | long | [0, V-1] | | | | | Target sequences |
| | pred_lens | [B] | int | [1, T] | | | | | Prediction lengths |
| | target_lens | [B] | int | [1, T] | | | | | Target lengths |

## Decoding Contracts

| Function | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `greedy_decode` | model | - | - | - | tokens | [B, T_out] | long | [0, V-1] | Greedy decoding |
| | features | [B, T_feat, F] | float32 | [-2, 2] | | | | | Input features |
| | max_len | int | - | [1, ∞) | | | | | Maximum output length |
| | feat_mask | [B, T_feat] | bool | {0, 1} | | | | | Feature padding mask |
| `beam_search` | model | - | - | - | tokens | [B, beam_size, T_out] | long | [0, V-1] | Beam search results |
| | features | [B, T_feat, F] | float32 | [-2, 2] | scores | [B, beam_size] | float32 | (-∞, 0] | Log probabilities |
| | beam_size | int | - | [1, ∞) | | | | | Number of beams |
| | max_len | int | - | [1, ∞) | | | | | Maximum output length |

## Training Loop Contracts

| Component | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|-----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `train_step` | batch | ([B,T_f,F], [B,T_t], [B], [B]) | (float32, long, int, int) | ([-2,2], [0,V-1], [1,T_f], [1,T_t]) | loss | scalar | float32 | [0, ∞) | Single training step |
| `val_step` | batch | ([B,T_f,F], [B,T_t], [B], [B]) | (float32, long, int, int) | ([-2,2], [0,V-1], [1,T_f], [1,T_t]) | metrics | dict | float32 | [0, 1] | Validation metrics |
| `predict` | features | [T_feat, F] | float32 | [-2, 2] | text | str | - | - | Single sequence prediction |

## Utility Functions Contracts

| Function | Input | Shape | Dtype | Range | Output | Shape | Dtype | Range | Notes |
|----------|-------|-------|-------|-------|--------|-------|-------|-------|-------|
| `create_padding_mask` | lengths | [B] | int | [1, T] | mask | [B, T] | bool | {0, 1} | True for valid positions |
| | max_len | int | - | [1, ∞) | | | | | Maximum sequence length |
| `create_causal_mask` | size | int | - | [1, ∞) | mask | [size, size] | bool | {0, 1} | Lower triangular mask |
| `apply_mask` | tensor | [B, T, ...] | float32 | any | tensor | [B, T, ...] | float32 | any | Sets masked positions to 0 |
| | mask | [B, T] | bool | {0, 1} | | | | | Mask to apply |
| `assert_shape` | tensor | any | any | any | None | - | - | - | Raises on mismatch |
| | expected | str | - | - | | | | | Pattern like "B,T,D" |

## Common Shape Patterns

### Sequence Lengths
- **Training Features**: 100-500 frames (variable)
- **Training Tokens**: 20-100 tokens (variable)
- **Inference**: Variable length, up to model maximum

### Batch Sizes
- **Training**: 16-32 (depending on sequence lengths and GPU memory)
- **Validation**: 32-64 (larger batches for efficiency)
- **Inference**: 1-8 (depending on service requirements)

### Model Dimensions
- **Feature Dim (F)**: 80 (mel-spectrogram features)
- **Hidden Dim (D)**: 512 (configurable)
- **Vocab Size (V)**: 100-1000 (character-level tokenizer)
- **Num Heads (H)**: 8 (hidden_dim must be divisible)

### Attention Patterns
- **Self-Attention**: [B, T, T] masks for encoder
- **Cross-Attention**: [B, T_dec, T_enc] masks for decoder
- **Causal Attention**: Lower triangular masks for autoregressive decoding

## Validation Guidelines

### Runtime Assertions
All major functions include shape assertions:
```python
assert_shape(features, "B,T,F", "input features")
assert_shape(tokens, "B,S", "target tokens")
assert_shape(mask, "B,T", "padding mask")
```

### Mask Validation
Attention masks must be properly formatted:
```python
assert mask.dtype == torch.bool, "Mask must be boolean"
assert mask.shape[-2:] == (T, S), "Mask shape mismatch"
```

### Sequence Length Checks
Variable-length sequences require length validation:
```python
assert all(length <= max_len for length in lengths), "Length exceeds maximum"
assert all(length > 0 for length in lengths), "Empty sequences not allowed"
```

### Memory Monitoring
Track GPU memory usage during training:
- Peak memory should not exceed available GPU memory
- Sequence length scaling affects memory quadratically (attention)
- Batch size adjustments based on memory constraints
- Gradient accumulation for large effective batch sizes

## Trou
bleshooting Guide

### Common Shape Mismatches

#### Problem: "Expected shape [B,T,F] but got [B,F,T]"
**Cause**: Feature tensor has time and feature dimensions swapped

**Solution**:
```python
# Transpose to correct format
if features.shape[1] == feature_dim and features.shape[2] > feature_dim:
    features = features.transpose(1, 2)  # [B,F,T] -> [B,T,F]
```

**Prevention**: Always maintain [B, T, F] format where T is time/sequence dimension.

#### Problem: "Sequence length mismatch between encoder output and decoder input"
**Cause**: Predictor output not properly aligned with decoder expectations

**Solution**:
```python
# Ensure predictor output matches encoder sequence length
encoder_out = encoder(features, feat_mask)  # [B, T_enc, D]
predictor_out = predictor(encoder_out)      # [B, T_enc, 1]

# Decoder should handle encoder sequence length
decoder_out = decoder(encoder_out, predictor_out, target_tokens)  # [B, T_dec, V]
```

**Prevention**: Validate sequence length consistency across all model components.

#### Problem: "RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)"
**Cause**: Padding mask dimensions don't match sequence length

**Solution**:
```python
# Create padding mask with correct dimensions
def create_padding_mask(lengths, max_len):
    batch_size = len(lengths)
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    return mask  # [B, T]

# Validate mask shape
assert mask.shape == (batch_size, seq_len), \
    f"Mask shape {mask.shape} doesn't match expected ({batch_size}, {seq_len})"
```

**Prevention**: Always create masks based on actual sequence lengths.

### Common Mask Issues

#### Problem: "Attention produces NaN values with masking"
**Cause**: Mask applied incorrectly or all positions masked out

**Solution**:
```python
# Proper mask application in attention
def apply_attention_mask(scores, mask):
    # mask: [B, T] or [B, T, T], True for valid positions
    if mask is not None:
        # Convert to attention mask format (False for valid positions)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Check if any sequence is fully masked
    if mask is not None and (~mask).all(dim=-1).any():
        print("Warning: Some sequences are fully masked!")
    
    return scores
```

**Prevention**: Ensure at least one valid position per sequence and use correct mask polarity.

#### Problem: "Padding tokens affect loss calculation"
**Cause**: Loss computed over padded positions

**Solution**:
```python
# Masked cross-entropy loss
def masked_cross_entropy(logits, targets, mask):
    # logits: [B, T, V], targets: [B, T], mask: [B, T]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='none'
    )
    loss = loss.reshape(targets.shape)
    
    # Apply mask and normalize by valid tokens
    loss = (loss * mask).sum() / mask.sum()
    return loss
```

**Prevention**: Always mask loss computation for variable-length sequences.

#### Problem: "Causal mask prevents model from seeing current token"
**Cause**: Incorrect causal mask construction

**Solution**:
```python
# Correct causal mask (lower triangular including diagonal)
def create_causal_mask(size):
    mask = torch.tril(torch.ones(size, size)).bool()
    return mask  # [T, T], True for valid positions

# Incorrect: torch.tril(torch.ones(size, size), diagonal=-1)  # Excludes diagonal
```

**Prevention**: Include diagonal in causal mask to allow attending to current position.

### Sequence-Specific Issues

#### Problem: "Token accuracy is 0% or very low"
**Cause**: Vocabulary mismatch between tokenizer and model, or incorrect decoding

**Solution**:
```python
# Verify vocabulary consistency
assert model.vocab_size == tokenizer.vocab_size, \
    f"Model vocab size {model.vocab_size} != tokenizer vocab size {tokenizer.vocab_size}"

# Check special token handling
print(f"PAD token: {tokenizer.pad_token_id}")
print(f"BOS token: {tokenizer.bos_token_id}")
print(f"EOS token: {tokenizer.eos_token_id}")

# Verify decoding
sample_tokens = torch.tensor([[1, 5, 10, 15, 2]])  # BOS, tokens, EOS
decoded = tokenizer.decode(sample_tokens[0])
print(f"Decoded: {decoded}")
```

**Prevention**: Initialize model and tokenizer with same vocabulary configuration.

#### Problem: "Greedy decoding produces empty or truncated sequences"
**Cause**: EOS token generated too early or max_length too small

**Solution**:
```python
# Adjust decoding parameters
def greedy_decode(model, features, max_len=100, min_len=5):
    tokens = [tokenizer.bos_token_id]
    
    for step in range(max_len):
        logits = model.decode_step(features, torch.tensor([tokens]))
        next_token = logits.argmax(dim=-1).item()
        
        # Prevent early EOS
        if next_token == tokenizer.eos_token_id and step < min_len:
            # Get second best token
            logits[0, -1, tokenizer.eos_token_id] = float('-inf')
            next_token = logits.argmax(dim=-1).item()
        
        tokens.append(next_token)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    return tokens
```

**Prevention**: Set appropriate min_len and max_len based on expected sequence lengths.

#### Problem: "Variable-length batching causes errors"
**Cause**: Improper collation or missing length tracking

**Solution**:
```python
# Proper collation function
def collate_fn(batch):
    features, tokens, feat_lens, token_lens = zip(*batch)
    
    # Pad features
    max_feat_len = max(feat_lens)
    padded_features = torch.zeros(len(batch), max_feat_len, features[0].shape[-1])
    for i, (feat, length) in enumerate(zip(features, feat_lens)):
        padded_features[i, :length] = feat
    
    # Pad tokens
    max_token_len = max(token_lens)
    padded_tokens = torch.full((len(batch), max_token_len), tokenizer.pad_token_id)
    for i, (tok, length) in enumerate(zip(tokens, token_lens)):
        padded_tokens[i, :length] = tok
    
    return (
        padded_features,
        padded_tokens,
        torch.tensor(feat_lens),
        torch.tensor(token_lens)
    )
```

**Prevention**: Always track and return sequence lengths in collation.

### Predictor-Specific Issues

#### Problem: "Predictor outputs are all zeros or ones"
**Cause**: Predictor not learning meaningful alignments

**Solution**:
```python
# Check predictor loss weight
if predictor_loss_weight == 0:
    print("Warning: Predictor loss weight is 0, predictor won't learn!")

# Visualize predictor outputs
def visualize_predictor(predictor_out, token_positions=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(predictor_out[0].detach().cpu().numpy())
    if token_positions:
        for pos in token_positions[0]:
            plt.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Frame')
    plt.ylabel('Boundary Probability')
    plt.title('Predictor Output')
    plt.savefig('predictor_debug.png')
    plt.close()

# Use during training
visualize_predictor(predictor_out, token_positions)
```

**Prevention**: Use non-zero predictor loss weight and monitor predictor outputs.

#### Problem: "Predictor loss is NaN"
**Cause**: Invalid target alignments or division by zero

**Solution**:
```python
# Robust predictor loss
def predictor_loss(predictions, targets, mask, epsilon=1e-7):
    # predictions: [B, T, 1], targets: [B, T], mask: [B, T]
    predictions = predictions.squeeze(-1)
    
    # Ensure valid range
    predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
    targets = torch.clamp(targets, 0, 1)
    
    # Binary cross-entropy with mask
    loss = F.binary_cross_entropy(predictions, targets, reduction='none')
    loss = (loss * mask).sum() / (mask.sum() + epsilon)
    
    # Check for NaN
    if torch.isnan(loss):
        print(f"NaN in predictor loss! Predictions: [{predictions.min()}, {predictions.max()}]")
        return torch.tensor(0.0, device=loss.device)
    
    return loss
```

**Prevention**: Clamp values and add epsilon to prevent numerical instability.

### Training Issues

#### Problem: "Loss oscillates wildly during training"
**Cause**: Learning rate too high or batch size too small

**Solution**:
```python
# Reduce learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Instead of 1e-3

# Increase effective batch size with gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = train_step(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

# Use learning rate warmup
def get_lr_schedule(optimizer, warmup_steps=1000):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Prevention**: Start with conservative learning rates and use warmup.

#### Problem: "Model doesn't learn on toy dataset"
**Cause**: Dataset too easy/hard or model capacity mismatch

**Solution**:
```python
# Verify dataset difficulty
dataset = ToySeq2SeqDataset(num_samples=100)
for i in range(5):
    features, tokens, feat_len, token_len = dataset[i]
    print(f"Sample {i}: feat_len={feat_len}, token_len={token_len}")
    print(f"  Tokens: {tokenizer.decode(tokens[:token_len])}")

# Adjust model capacity
if dataset is too easy:
    # Reduce model size
    model = Paraformer(hidden_dim=128, num_layers=2)
elif dataset is too hard:
    # Increase model size
    model = Paraformer(hidden_dim=512, num_layers=6)
```

**Prevention**: Match model capacity to dataset complexity.

#### Problem: "Validation accuracy much lower than training"
**Cause**: Overfitting or train/val distribution mismatch

**Solution**:
```python
# Add dropout
model = Paraformer(dropout=0.2)

# Use weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Verify train/val split
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Train avg length: {train_dataset.get_avg_length()}")
print(f"Val avg length: {val_dataset.get_avg_length()}")

# Early stopping
best_val_acc = 0
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_acc = validate(model, val_loader)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        save_checkpoint(model, 'best.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
```

**Prevention**: Use regularization and monitor train/val gap.

### Memory Issues

#### Problem: "CUDA out of memory with long sequences"
**Cause**: Attention memory scales quadratically with sequence length

**Solution**:
```python
# Reduce batch size for long sequences
def get_dynamic_batch_size(seq_len):
    if seq_len < 100:
        return 32
    elif seq_len < 300:
        return 16
    else:
        return 8

# Use gradient checkpointing
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # ... initialize layers
    
    def forward(self, x, mask=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, mask)
        return self._forward(x, mask)

# Truncate very long sequences
max_seq_len = 500
if features.shape[1] > max_seq_len:
    features = features[:, :max_seq_len]
    feat_lens = torch.clamp(feat_lens, max=max_seq_len)
```

**Prevention**: Profile memory usage and adjust batch size based on sequence length.

#### Problem: "Memory leak during training"
**Cause**: Retaining computation graphs or accumulating tensors

**Solution**:
```python
# Detach metrics from computation graph
metrics = {
    'loss': loss.item(),  # Use .item() not .detach()
    'accuracy': accuracy.item()
}

# Clear cache periodically
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()

# Use context manager for validation
with torch.no_grad():
    val_metrics = validate(model, val_loader)

# Don't accumulate tensors in lists
# Bad: all_losses.append(loss)
# Good: all_losses.append(loss.item())
```

**Prevention**: Always detach tensors when not needed for backpropagation.

### Debugging Tips

#### Enable Detailed Error Messages
```python
# Set environment variables
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Add backward hooks for gradient debugging
def gradient_hook(grad):
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print(f"NaN/Inf gradient detected! Shape: {grad.shape}")
        print(f"  Min: {grad.min()}, Max: {grad.max()}")
    return grad

for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(gradient_hook)
```

#### Add Comprehensive Logging
```python
# Log tensor statistics
def log_tensor_stats(tensor, name):
    if tensor is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
    if tensor.dtype in [torch.float32, torch.float16]:
        print(f"  min={tensor.min():.4f}, max={tensor.max():.4f}, "
              f"mean={tensor.mean():.4f}, std={tensor.std():.4f}")
        print(f"  nan={torch.isnan(tensor).sum()}, inf={torch.isinf(tensor).sum()}")

# Use in training loop
log_tensor_stats(features, "features")
log_tensor_stats(encoder_out, "encoder_out")
log_tensor_stats(predictor_out, "predictor_out")
log_tensor_stats(logits, "logits")
log_tensor_stats(loss, "loss")
```

#### Visualize Attention Patterns
```python
# Extract and visualize attention weights
def visualize_attention(attention_weights, save_path):
    import matplotlib.pyplot as plt
    # attention_weights: [num_heads, T, T]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < attention_weights.shape[0]:
            im = ax.imshow(attention_weights[i].detach().cpu().numpy(), cmap='viridis')
            ax.set_title(f'Head {i}')
            plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Use during debugging
attention_weights = model.encoder.layers[0].attention.get_attention_weights()
visualize_attention(attention_weights, 'attention_debug.png')
```

#### Test Individual Components
```python
# Unit test for encoder
def test_encoder():
    encoder = Encoder(feature_dim=80, hidden_dim=512, num_layers=6)
    features = torch.randn(2, 100, 80)
    mask = torch.ones(2, 100).bool()
    
    output = encoder(features, mask)
    assert output.shape == (2, 100, 512), f"Unexpected shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in encoder output"
    print("Encoder test passed!")

# Unit test for predictor
def test_predictor():
    predictor = Predictor(hidden_dim=512)
    encoder_out = torch.randn(2, 100, 512)
    
    predictions = predictor(encoder_out)
    assert predictions.shape == (2, 100, 1), f"Unexpected shape: {predictions.shape}"
    assert (predictions >= 0).all() and (predictions <= 1).all(), "Predictions out of range"
    print("Predictor test passed!")

# Run tests
test_encoder()
test_predictor()
```
