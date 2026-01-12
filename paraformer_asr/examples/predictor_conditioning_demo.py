#!/usr/bin/env python3
"""
Demonstration of Predictor Conditioning in Paraformer ASR

This script shows how the predictor module estimates token boundaries
and conditions the decoder processing for better alignment.

Key concepts demonstrated:
1. How predictor estimates alignment from encoder features
2. How predictor output conditions decoder attention
3. Training with alignment supervision
4. Visualization of alignment predictions
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.paraformer import ParaformerASR
from data.toy_seq2seq import ToySeq2SeqDataset
from data.tokenizer import CharacterTokenizer


def create_sample_data(batch_size: int = 2, max_feature_len: int = 100, vocab_size: int = 50):
    """Create sample data for demonstration."""
    
    # Create toy dataset and tokenizer
    dataset = ToySeq2SeqDataset(
        vocab_size=vocab_size,
        min_seq_len=10,
        max_seq_len=30,
        feature_dim=80,
        max_feature_len=max_feature_len,
        correlation_strength=0.8
    )
    
    tokenizer = CharacterTokenizer(vocab_size=vocab_size)
    
    # Generate batch of samples
    features_list = []
    tokens_list = []
    feature_lengths = []
    token_lengths = []
    
    for _ in range(batch_size):
        features, tokens, feat_len, token_len = dataset[0]  # Generate random sample
        features_list.append(features)
        tokens_list.append(tokens)
        feature_lengths.append(feat_len)
        token_lengths.append(token_len)
    
    # Pad and stack
    max_feat_len = max(feature_lengths)
    max_token_len = max(token_lengths)
    
    # Pad features
    padded_features = torch.zeros(batch_size, max_feat_len, 80)
    for i, feat in enumerate(features_list):
        padded_features[i, :feature_lengths[i]] = feat
    
    # Pad tokens
    padded_tokens = torch.zeros(batch_size, max_token_len, dtype=torch.long)
    for i, tokens in enumerate(tokens_list):
        padded_tokens[i, :token_lengths[i]] = tokens
    
    return {
        'features': padded_features,
        'feature_lengths': torch.tensor(feature_lengths),
        'tokens': padded_tokens,
        'token_lengths': torch.tensor(token_lengths),
        'tokenizer': tokenizer
    }


def demonstrate_predictor_conditioning():
    """Demonstrate how predictor conditions decoder processing."""
    
    print("=== Paraformer Predictor Conditioning Demonstration ===\n")
    
    # 1. Create model
    print("1. Creating Paraformer model...")
    model = ParaformerASR(
        input_dim=80,
        encoder_dim=256,
        encoder_layers=4,
        encoder_heads=8,
        vocab_size=50,
        predictor_layers=2,
        predictor_hidden_dim=128,
        decoder_layers=4,
        decoder_heads=8,
        predictor_integration='concat',  # Show how predictor conditions decoder
        predictor_loss_weight=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 2. Create sample data
    print("\n2. Creating sample data...")
    data = create_sample_data(batch_size=2, max_feature_len=80, vocab_size=50)
    
    features = data['features']
    feature_lengths = data['feature_lengths']
    tokens = data['tokens']
    token_lengths = data['token_lengths']
    
    print(f"Features shape: {features.shape}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"Feature lengths: {feature_lengths}")
    print(f"Token lengths: {token_lengths}")
    
    # 3. Forward pass - show how components work together
    print("\n3. Forward pass through model components...")
    
    model.eval()
    with torch.no_grad():
        # Step 3a: Encoder processes features
        print("   a) Encoder processing features...")
        encoder_features, padding_mask = model.encoder(features, feature_lengths)
        print(f"      Encoder output shape: {encoder_features.shape}")
        print(f"      Padding mask shape: {padding_mask.shape}")
        
        # Step 3b: Predictor estimates alignment
        print("   b) Predictor estimating alignment...")
        predictor_predictions, predictor_probabilities = model.predictor(
            encoder_features, padding_mask
        )
        print(f"      Predictor predictions shape: {predictor_predictions.shape}")
        print(f"      Predictor probabilities shape: {predictor_probabilities.shape}")
        
        # Show alignment statistics
        for i in range(features.shape[0]):
            valid_len = feature_lengths[i].item()
            probs = predictor_probabilities[i, :valid_len, 0]
            max_prob = probs.max().item()
            mean_prob = probs.mean().item()
            print(f"      Batch {i}: max_prob={max_prob:.3f}, mean_prob={mean_prob:.3f}")
        
        # Step 3c: Decoder uses predictor output for conditioning
        print("   c) Decoder generating tokens with predictor conditioning...")
        
        # Show how predictor output is integrated
        integration = model.decoder.predictor_integration
        integrated_features = integration(encoder_features, predictor_probabilities)
        print(f"      Integrated features shape: {integrated_features.shape}")
        print(f"      Integration method: {integration.integration_type}")
        
        # Generate tokens
        generated_tokens = model.generate(
            features=features,
            feature_lengths=feature_lengths,
            max_length=30
        )
        print(f"      Generated tokens shape: {generated_tokens.shape}")
    
    # 4. Training mode - show loss computation
    print("\n4. Training mode with alignment supervision...")
    
    model.train()
    
    # Create synthetic alignment targets (normally from forced alignment)
    alignment_targets = create_alignment_targets(token_lengths, features.shape[1])
    
    # Forward pass in training mode
    outputs = model(
        features=features,
        feature_lengths=feature_lengths,
        target_tokens=tokens,
        target_lengths=token_lengths,
        target_alignments=alignment_targets
    )
    
    # Compute losses
    losses = model.compute_loss(
        outputs=outputs,
        target_tokens=tokens,
        target_alignments=alignment_targets,
        target_lengths=token_lengths,
        label_smoothing=0.1
    )
    
    print(f"   Total loss: {losses['total_loss'].item():.4f}")
    print(f"   Decoder loss: {losses['decoder_loss'].item():.4f}")
    print(f"   Predictor loss: {losses['predictor_loss'].item():.4f}")
    
    # 5. Demonstrate alignment extraction
    print("\n5. Extracting token alignment positions...")
    
    model.eval()
    with torch.no_grad():
        alignment_probs, token_positions = model.extract_alignment(
            features=features,
            feature_lengths=feature_lengths,
            threshold=0.3,
            min_distance=2
        )
        
        for i, positions in enumerate(token_positions):
            print(f"   Batch {i} predicted positions: {positions}")
            print(f"   Batch {i} token length: {token_lengths[i].item()}")
    
    # 6. Visualization (if matplotlib available)
    print("\n6. Creating alignment visualization...")
    try:
        visualize_alignment(
            alignment_probs=alignment_probs,
            token_positions=token_positions,
            feature_lengths=feature_lengths,
            batch_idx=0
        )
        print("   Alignment plot saved as 'alignment_demo.png'")
    except ImportError:
        print("   Matplotlib not available, skipping visualization")
    
    print("\n=== Demonstration Complete ===")
    
    return {
        'model': model,
        'data': data,
        'outputs': outputs,
        'losses': losses,
        'alignment_probs': alignment_probs,
        'token_positions': token_positions
    }


def create_alignment_targets(token_lengths: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Create synthetic alignment targets for demonstration."""
    batch_size = token_lengths.shape[0]
    targets = torch.zeros(batch_size, sequence_length)
    
    for i, token_len in enumerate(token_lengths):
        # Create evenly spaced token boundaries
        if token_len > 1:
            positions = torch.linspace(0, sequence_length - 1, token_len.item()).long()
            targets[i, positions] = 1.0
    
    return targets


def visualize_alignment(
    alignment_probs: torch.Tensor,
    token_positions: list,
    feature_lengths: torch.Tensor,
    batch_idx: int = 0
):
    """Visualize alignment predictions."""
    
    # Extract data for visualization
    probs = alignment_probs[batch_idx, :, 0].cpu().numpy()
    valid_len = feature_lengths[batch_idx].item()
    probs = probs[:valid_len]
    
    predicted_pos = token_positions[batch_idx]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot alignment probabilities
    time_axis = np.arange(valid_len)
    plt.plot(time_axis, probs, 'b-', linewidth=2, label='Alignment Probability')
    
    # Mark predicted positions
    for pos in predicted_pos:
        if pos < valid_len:
            plt.axvline(x=pos, color='red', linestyle='--', alpha=0.7, label='Predicted Boundary' if pos == predicted_pos[0] else '')
    
    plt.xlabel('Feature Time Steps')
    plt.ylabel('Boundary Probability')
    plt.title(f'Token Alignment Prediction (Batch {batch_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('alignment_demo.png', dpi=150, bbox_inches='tight')
    plt.close()


def explain_conditioning_mechanism():
    """Explain how predictor conditioning works."""
    
    print("\n=== How Predictor Conditioning Works ===")
    print("""
The predictor module conditions decoder processing in several key ways:

1. ALIGNMENT ESTIMATION:
   - Predictor takes encoder features [B, T, D] 
   - Outputs boundary probabilities [B, T, 1]
   - Estimates where tokens should be aligned in the audio

2. FEATURE INTEGRATION:
   - Decoder integrates predictor output with encoder features
   - Methods: concat, add, or gate
   - Creates enhanced features [B, T, D] with alignment info

3. ATTENTION CONDITIONING:
   - Decoder cross-attention uses integrated features
   - Predictor signals guide where to attend for each token
   - Improves alignment between audio and text

4. TRAINING SUPERVISION:
   - Predictor trained with alignment targets (from forced alignment)
   - Joint training with decoder improves both components
   - Predictor loss weighted and combined with decoder loss

5. INFERENCE BENEFITS:
   - Better alignment leads to more accurate transcriptions
   - Faster convergence during training
   - More robust to audio variations

This conditioning mechanism is what makes Paraformer effective for
non-autoregressive speech recognition.
""")


if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_predictor_conditioning()
    
    # Explain the mechanism
    explain_conditioning_mechanism()
    
    print(f"\nDemo completed successfully!")
    print(f"Check 'alignment_demo.png' for visualization (if matplotlib available)")