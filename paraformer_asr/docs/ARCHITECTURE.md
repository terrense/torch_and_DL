# Paraformer ASR Architecture

This document provides a detailed explanation of the Paraformer-style automatic speech recognition architecture.

## Overview

Paraformer is a non-autoregressive ASR model that consists of three main components:
- **Encoder**: Processes input features and extracts contextual representations
- **Predictor**: Estimates token boundaries and alignment information
- **Decoder**: Generates output tokens using encoder features and predictor guidance

## High-Level Architecture

```
Input Features [B, T_feat, F]
    ↓
Encoder (Transformer/Conformer)
    ├─ Multi-layer self-attention
    ├─ Feed-forward networks
    └─ Positional encoding
    ↓
Encoder Output [B, T_feat, D]
    ├─────────────────┐
    ↓                 ↓
Predictor         Decoder
    ├─ Boundary      ├─ Cross-attention
    ├─ Estimation    ├─ Self-attention
    └─ Alignment     └─ Token generation
    ↓                 ↓
Predictor Out    Token Logits
[B, T_feat, 1]   [B, T_token, V]
```

## Component Details

### 1. Encoder

The encoder transforms input acoustic features into contextual representations.


#### Encoder Structure

```
Input Features [B, T, F]
    ↓
Linear Projection [B, T, D]
    ↓
Positional Encoding [B, T, D]
    ↓
Transformer Layer 1
    ├─ Multi-Head Self-Attention
    ├─ Add & Norm
    ├─ Feed-Forward Network
    └─ Add & Norm
    ↓
Transformer Layer 2...N
    ↓
Encoder Output [B, T, D]
```

#### Multi-Head Self-Attention

```python
# For each attention head h:
Q = W_q @ x  # Query: [B, T, D/H]
K = W_k @ x  # Key: [B, T, D/H]
V = W_v @ x  # Value: [B, T, D/H]

# Scaled dot-product attention
scores = (Q @ K.T) / sqrt(D/H)  # [B, T, T]

# Apply padding mask
if mask is not None:
    scores = scores.masked_fill(~mask, -inf)

attention = softmax(scores, dim=-1)  # [B, T, T]
output = attention @ V  # [B, T, D/H]

# Concatenate all heads
multi_head_output = concat([head_1, ..., head_H])  # [B, T, D]
```

**Key Features:**
- **Bidirectional attention**: Each position attends to all positions
- **Padding mask**: Prevents attention to padded positions
- **Multiple heads**: Different heads capture different patterns

### 2. Predictor Module

The predictor estimates token boundaries in the feature sequence.


#### Predictor Purpose

The predictor solves the alignment problem:
- **Input**: Continuous feature sequence of length T_feat
- **Output**: Discrete token sequence of length T_token (where T_token << T_feat)
- **Challenge**: How to map from features to tokens?

#### Predictor Architecture

```
Encoder Output [B, T, D]
    ↓
Linear Layer 1 [B, T, D/2]
    ↓
ReLU Activation
    ↓
Linear Layer 2 [B, T, 1]
    ↓
Sigmoid/Softplus [B, T, 1]
```

#### Predictor Output Interpretation

The predictor outputs a probability for each frame indicating token boundaries:

```python
# Example predictor output for "HELLO"
# Frame:  0   1   2   3   4   5   6   7   8   9  10  11  12
# Prob:  0.1 0.9 0.1 0.1 0.8 0.1 0.9 0.1 0.9 0.1 0.8 0.1 0.1
#         ^   H   ^   ^   E   ^   L   ^   L   ^   O   ^   ^
# Peaks indicate token boundaries
```

**Training the Predictor:**
- Supervised with ground-truth alignments
- Loss: Binary cross-entropy or MSE
- Target: Gaussian peaks at token boundaries

### 3. Decoder

The decoder generates output tokens using encoder features and predictor guidance.

#### Decoder Structure

```
Encoder Output [B, T_enc, D]
Predictor Output [B, T_enc, 1]
Target Tokens [B, T_dec] (training only)
    ↓
Token Embedding [B, T_dec, D]
    ↓
Positional Encoding [B, T_dec, D]
    ↓
Decoder Layer 1
    ├─ Masked Self-Attention (causal)
    ├─ Add & Norm
    ├─ Cross-Attention (to encoder)
    ├─ Add & Norm
    ├─ Feed-Forward Network
    └─ Add & Norm
    ↓
Decoder Layer 2...N
    ↓
Linear Projection [B, T_dec, V]
    ↓
Token Logits [B, T_dec, V]
```


#### Cross-Attention Mechanism

Cross-attention allows decoder to attend to encoder features:

```python
# Decoder queries encoder
Q = W_q @ decoder_hidden  # [B, T_dec, D]
K = W_k @ encoder_output  # [B, T_enc, D]
V = W_v @ encoder_output  # [B, T_enc, D]

# Attention scores
scores = (Q @ K.T) / sqrt(D)  # [B, T_dec, T_enc]

# Use predictor to guide attention
if predictor_out is not None:
    # Predictor provides soft alignment hints
    alignment_bias = predictor_out.squeeze(-1)  # [B, T_enc]
    scores = scores + alignment_bias.unsqueeze(1)  # Broadcast to [B, T_dec, T_enc]

attention = softmax(scores, dim=-1)
output = attention @ V  # [B, T_dec, D]
```

## Training Process

### Loss Functions

#### 1. Token Prediction Loss (Main)

```python
def token_loss(logits, targets, mask):
    # logits: [B, T, V]
    # targets: [B, T]
    # mask: [B, T] - True for valid positions
    
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction='none'
    )
    loss = loss.reshape(targets.shape)
    
    # Apply mask
    loss = (loss * mask).sum() / mask.sum()
    return loss
```

#### 2. Predictor Loss (Auxiliary)

```python
def predictor_loss(predictions, targets, mask):
    # predictions: [B, T, 1]
    # targets: [B, T] - Gaussian peaks at boundaries
    # mask: [B, T]
    
    loss = F.mse_loss(
        predictions.squeeze(-1),
        targets,
        reduction='none'
    )
    
    # Apply mask
    loss = (loss * mask).sum() / mask.sum()
    return loss
```

#### 3. Combined Loss

```python
total_loss = token_loss + lambda_pred * predictor_loss
# Typical: lambda_pred = 1.0
```

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        features, tokens, feat_lens, token_lens = batch
        
        # Forward pass
        logits = model(features, tokens, feat_lens, token_lens)
        
        # Compute losses
        token_loss = compute_token_loss(logits, tokens, token_lens)
        pred_loss = compute_predictor_loss(model.predictor_out, alignments, feat_lens)
        
        total_loss = token_loss + pred_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

## Inference Process

### Greedy Decoding

```python
def greedy_decode(model, features, max_len=100):
    # Encode
    encoder_out, predictor_out = model.encode(features)
    
    # Initialize with BOS token
    tokens = [BOS_TOKEN_ID]
    
    for step in range(max_len):
        # Decode one step
        logits = model.decode_step(encoder_out, predictor_out, tokens)
        
        # Get next token
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        tokens.append(next_token)
        
        # Stop if EOS
        if next_token == EOS_TOKEN_ID:
            break
    
    return tokens
```

### Beam Search Decoding

```python
def beam_search(model, features, beam_size=4, max_len=100):
    encoder_out, predictor_out = model.encode(features)
    
    # Initialize beams
    beams = [(0.0, [BOS_TOKEN_ID])]  # (score, tokens)
    
    for step in range(max_len):
        candidates = []
        
        for score, tokens in beams:
            if tokens[-1] == EOS_TOKEN_ID:
                candidates.append((score, tokens))
                continue
            
            # Get next token probabilities
            logits = model.decode_step(encoder_out, predictor_out, tokens)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            
            # Expand beam
            top_k_probs, top_k_ids = log_probs.topk(beam_size)
            for prob, token_id in zip(top_k_probs[0], top_k_ids[0]):
                new_score = score + prob.item()
                new_tokens = tokens + [token_id.item()]
                candidates.append((new_score, new_tokens))
        
        # Keep top beams
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
    
    return beams[0][1]  # Return best sequence
```

## Performance Optimization

### Memory Efficiency

1. **Gradient Checkpointing**:
```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    return checkpoint(self.transformer_layer, x)
```

2. **Mixed Precision Training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits = model(features, tokens)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Speed Optimization

1. **Efficient Attention**:
```python
# Use Flash Attention for faster computation
from flash_attn import flash_attn_func

attention_output = flash_attn_func(q, k, v, causal=True)
```

2. **Batch Processing**:
```python
# Process multiple sequences in parallel
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)
```

## Extending the Architecture

### Adding Conformer Blocks

Replace transformer layers with conformer blocks for better speech modeling:

```python
class ConformerBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ffn1 = FeedForward(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim)
        self.conv = ConvModule(hidden_dim)
        self.ffn2 = FeedForward(hidden_dim)
    
    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attention(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return x
```

### Multi-Task Learning

Add auxiliary tasks for better representations:

```python
class MultiTaskParaformer(nn.Module):
    def forward(self, features):
        encoder_out = self.encoder(features)
        
        # Main task: ASR
        asr_logits = self.decoder(encoder_out)
        
        # Auxiliary task: CTC
        ctc_logits = self.ctc_head(encoder_out)
        
        # Auxiliary task: Language modeling
        lm_logits = self.lm_head(encoder_out)
        
        return asr_logits, ctc_logits, lm_logits
```

## References

- Paraformer: Gao et al., "Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition" (2022)
- Transformer: Vaswani et al., "Attention Is All You Need" (2017)
- Conformer: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020)
