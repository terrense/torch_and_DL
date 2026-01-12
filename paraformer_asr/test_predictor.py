#!/usr/bin/env python3
"""
Simple test script for the predictor module implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from models.paraformer import ParaformerASR


def test_predictor_implementation():
    """Test the predictor module implementation."""
    
    print("Testing Paraformer ASR with Predictor Module...")
    
    # Create model
    model = ParaformerASR(
        input_dim=80,
        encoder_dim=256,
        vocab_size=100,
        predictor_layers=2,
        decoder_layers=2,
        encoder_layers=2,
        predictor_type='boundary'
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create test data
    batch_size = 2
    feature_length = 50
    token_length = 20
    
    features = torch.randn(batch_size, feature_length, 80)
    feature_lengths = torch.tensor([45, 50])
    tokens = torch.randint(0, 100, (batch_size, token_length))
    token_lengths = torch.tensor([18, 20])
    
    print(f"Test data shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Tokens: {tokens.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            features=features,
            feature_lengths=feature_lengths,
            target_tokens=tokens,
            target_lengths=token_lengths
        )
    
    print(f"Forward pass successful!")
    print(f"  Predictor predictions: {outputs['predictor_predictions'].shape}")
    print(f"  Predictor probabilities: {outputs['predictor_probabilities'].shape}")
    print(f"  Decoder logits: {outputs['logits'].shape}")
    
    # Test predictor conditioning
    predictor_probs = outputs['predictor_probabilities']
    print(f"  Predictor probability range: [{predictor_probs.min():.3f}, {predictor_probs.max():.3f}]")
    
    # Test alignment extraction
    alignment_probs, token_positions = model.extract_alignment(
        features=features,
        feature_lengths=feature_lengths,
        threshold=0.3
    )
    
    print(f"Alignment extraction successful!")
    for i, positions in enumerate(token_positions):
        print(f"  Batch {i} predicted positions: {positions}")
    
    # Test loss computation
    model.train()
    
    # Create synthetic alignment targets
    alignment_targets = torch.zeros(batch_size, feature_length)
    for i, token_len in enumerate(token_lengths):
        # Create evenly spaced boundaries
        if token_len > 1:
            positions = torch.linspace(0, feature_length - 1, token_len.item()).long()
            alignment_targets[i, positions] = 1.0
    
    outputs = model(
        features=features,
        feature_lengths=feature_lengths,
        target_tokens=tokens,
        target_lengths=token_lengths,
        target_alignments=alignment_targets
    )
    
    losses = model.compute_loss(
        outputs=outputs,
        target_tokens=tokens,
        target_alignments=alignment_targets,
        target_lengths=token_lengths
    )
    
    print(f"Loss computation successful!")
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Decoder loss: {losses['decoder_loss'].item():.4f}")
    print(f"  Predictor loss: {losses['predictor_loss'].item():.4f}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        generated_tokens = model.generate(
            features=features,
            feature_lengths=feature_lengths,
            max_length=25
        )
    
    print(f"Generation successful!")
    print(f"  Generated tokens shape: {generated_tokens.shape}")
    
    print("\n✓ All predictor tests passed!")
    
    return True


if __name__ == "__main__":
    try:
        test_predictor_implementation()
        print("\n🎉 Predictor module implementation is complete and working!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()