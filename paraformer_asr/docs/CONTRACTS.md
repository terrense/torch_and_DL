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
| `forward` | encoder_out | [B, T, D] | float32 | any | predictions | [B, T, 1] | float32 | [0, ∞) | Duration predictions |
| | mask | [B, T] | bool | {0, 1} | | | | | Optional padding mask |
| `predict_alignment` | encoder_out | [B, T, D] | float32 | any | alignment | [B, T] | float32 | [0, 1] | Normalized alignment |
| | target_len | int | - | [1, ∞) | | | | | Target sequence length |

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