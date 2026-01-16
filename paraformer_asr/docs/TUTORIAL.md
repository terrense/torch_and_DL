# Paraformer ASR Tutorial

This tutorial walks you through using and extending the Paraformer ASR project.

## Getting Started

### Installation

```bash
cd paraformer_asr
pip install -r requirements.txt
```

### Quick Test

```bash
# Run a quick smoke test
python -m pytest tests/test_smoke.py -v
```

## Tutorial 1: Training Your First Model

### Step 1: Understand the Configuration

Open `configs/paraformer_base.yaml`:

```yaml
model:
  encoder:
    num_layers: 6
    hidden_dim: 512
    num_heads: 8
  predictor:
    num_layers: 2
    hidden_dim: 256
  decoder:
    num_layers: 6
    hidden_dim: 512

data:
  vocab_size: 100
  feature_dim: 80
  max_feat_len: 300
  max_token_len: 60

training:
  batch_size: 16
  learning_rate: 0.0001
  num_epochs: 100
```

### Step 2: Train the Model

```bash
python scripts/train.py --config configs/paraformer_base.yaml --run-name my_first_asr
```

Watch the training:
- Token accuracy should increase
- Loss should decrease
- Check `runs/my_first_asr/` for logs

### Step 3: Evaluate

```bash
python scripts/evaluate.py --checkpoint runs/my_first_asr/checkpoints/best.pt
```

### Step 4: Run Inference

```bash
python scripts/inference.py \
    --checkpoint runs/my_first_asr/checkpoints/best.pt \
    --text "hello world"
```

## Tutorial 2: Understanding the Code

### Tokenizer

```python
from src.data.tokenizer import CharTokenizer

# Initialize tokenizer
tokenizer = CharTokenizer(vocab_size=100)

# Encode text
text = "hello"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")  # [BOS, h_id, e_id, l_id, l_id, o_id, EOS]

# Decode tokens
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")  # "hello"
```

### Dataset

```python
from src.data.toy_seq2seq import ToySeq2SeqDataset

dataset = ToySeq2SeqDataset(
    num_samples=100,
    tokenizer=tokenizer,
    max_feat_len=300,
    max_token_len=60
)

features, tokens, feat_len, token_len = dataset[0]
print(f"Features: {features.shape}")  # [T_feat, 80]
print(f"Tokens: {tokens.shape}")      # [T_token]
```

### Model Forward Pass

```python
from src.models.paraformer import Paraformer

model = Paraformer(
    vocab_size=tokenizer.vocab_size,
    feature_dim=80,
    hidden_dim=512
)

# Training forward pass
logits = model(features, tokens, feat_lens, token_lens)
print(f"Logits: {logits.shape}")  # [B, T_token, vocab_size]

# Inference
encoder_out, predictor_out = model.encode(features, feat_lens)
decoded_tokens = model.decode_greedy(encoder_out, predictor_out, max_len=100)
```

## Tutorial 3: Experimenting with Architecture

### Experiment 1: Encoder Depth

```bash
# Shallow encoder
python scripts/train.py --config configs/paraformer_base.yaml \
    --override model.encoder.num_layers=3 \
    --run-name encoder_3layers

# Deep encoder
python scripts/train.py --config configs/paraformer_base.yaml \
    --override model.encoder.num_layers=12 \
    --run-name encoder_12layers
```

### Experiment 2: Hidden Dimension

```bash
# Small model
python scripts/train.py --config configs/paraformer_base.yaml \
    --override model.encoder.hidden_dim=256 \
    --override model.decoder.hidden_dim=256 \
    --run-name small_model

# Large model
python scripts/train.py --config configs/paraformer_base.yaml \
    --override model.encoder.hidden_dim=1024 \
    --override model.decoder.hidden_dim=1024 \
    --run-name large_model
```

### Experiment 3: Predictor Impact

```bash
# Without predictor
python scripts/train.py --config configs/paraformer_base.yaml \
    --override model.use_predictor=false \
    --run-name no_predictor

# With predictor (default)
python scripts/train.py --config configs/paraformer_base.yaml \
    --run-name with_predictor
```

## Tutorial 4: Custom Components

### Custom Attention Mechanism

Create `src/models/custom_attention.py`:

```python
import torch
import torch.nn as nn

class LocalAttention(nn.Module):
    """Attention with local window."""
    def __init__(self, hidden_dim, num_heads, window_size=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, mask=None):
        B, T, D = x.shape
        
        # Project
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        
        # Local attention (only attend to nearby positions)
        output = []
        for i in range(T):
            start = max(0, i - self.window_size // 2)
            end = min(T, i + self.window_size // 2)
            
            q_i = q[:, i:i+1]  # [B, 1, H, D/H]
            k_local = k[:, start:end]  # [B, W, H, D/H]
            v_local = v[:, start:end]  # [B, W, H, D/H]
            
            # Attention
            scores = torch.einsum('bqhd,bkhd->bhqk', q_i, k_local) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out_i = torch.einsum('bhqk,bkhd->bqhd', attn, v_local)
            output.append(out_i)
        
        output = torch.cat(output, dim=1)  # [B, T, H, D/H]
        output = output.reshape(B, T, D)
        return self.out_proj(output)
```

### Custom Decoding Strategy

Create `src/decode/nucleus_sampling.py`:

```python
import torch
import torch.nn.functional as F

def nucleus_sampling(model, features, top_p=0.9, temperature=1.0, max_len=100):
    """Nucleus (top-p) sampling for diverse outputs."""
    encoder_out, predictor_out = model.encode(features)
    
    tokens = [model.tokenizer.bos_token_id]
    
    for step in range(max_len):
        logits = model.decode_step(encoder_out, predictor_out, tokens)
        logits = logits[:, -1, :] / temperature
        
        # Sort probabilities
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # Compute cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Zero out removed indices
        sorted_probs[sorted_indices_to_remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()
        
        # Sample from filtered distribution
        next_token = torch.multinomial(sorted_probs, 1).item()
        next_token = sorted_indices[0, next_token].item()
        
        tokens.append(next_token)
        
        if next_token == model.tokenizer.eos_token_id:
            break
    
    return tokens
```

## Tutorial 5: Working with Real Data

### Preparing Audio Features

```python
import torchaudio
import torch

def extract_mel_features(audio_path, n_mels=80):
    """Extract mel-spectrogram features from audio."""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Extract mel-spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)
    
    # Convert to log scale
    log_mel = torch.log(mel_spec + 1e-9)
    
    # Transpose to [T, F]
    features = log_mel.squeeze(0).transpose(0, 1)
    
    return features
```

### Custom Dataset for Real Audio

```python
from pathlib import Path
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_dir, transcript_file, tokenizer):
        self.audio_dir = Path(audio_dir)
        self.tokenizer = tokenizer
        
        # Load transcripts
        self.samples = []
        with open(transcript_file, 'r') as f:
            for line in f:
                audio_file, transcript = line.strip().split('\t')
                self.samples.append((audio_file, transcript))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_file, transcript = self.samples[idx]
        
        # Extract features
        audio_path = self.audio_dir / audio_file
        features = extract_mel_features(audio_path)
        
        # Tokenize transcript
        tokens = self.tokenizer.encode(transcript)
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return features, tokens, features.shape[0], len(tokens)
```

## Tutorial 6: Debugging and Optimization

### Visualizing Attention

```python
import matplotlib.pyplot as plt

def visualize_attention(model, features):
    """Visualize attention patterns."""
    model.eval()
    with torch.no_grad():
        encoder_out, _ = model.encode(features.unsqueeze(0))
        
        # Get attention weights from first layer
        attention_weights = model.encoder.layers[0].attention.get_weights()
        
        # Plot
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, ax in enumerate(axes.flat):
            if i < attention_weights.shape[1]:
                ax.imshow(attention_weights[0, i].cpu().numpy())
                ax.set_title(f'Head {i}')
        plt.savefig('attention_visualization.png')
```

### Profiling Performance

```bash
# Profile memory and speed
python scripts/profile_performance.py \
    --config configs/paraformer_base.yaml \
    --output results/profile.json
```

### Debugging Predictor

```python
def debug_predictor(model, features, tokens):
    """Debug predictor outputs."""
    model.eval()
    with torch.no_grad():
        encoder_out, predictor_out = model.encode(features.unsqueeze(0))
        
        # Visualize predictor output
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(predictor_out[0].squeeze().cpu().numpy())
        plt.xlabel('Frame')
        plt.ylabel('Boundary Probability')
        plt.title('Predictor Output')
        
        # Mark actual token boundaries
        for i in range(len(tokens)):
            plt.axvline(x=i * (len(predictor_out[0]) // len(tokens)), 
                       color='r', linestyle='--', alpha=0.5)
        
        plt.savefig('predictor_debug.png')
```

## Next Steps

1. **Read Architecture Guide**: `docs/ARCHITECTURE.md`
2. **Study Tensor Contracts**: `docs/CONTRACTS.md`
3. **Run Ablation Studies**: `docs/ABLATIONS.md`
4. **Explore the Code**: Start with `src/models/paraformer.py`
5. **Experiment**: Try different configurations and share results!
