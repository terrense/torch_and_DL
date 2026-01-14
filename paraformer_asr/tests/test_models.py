"""
Unit tests for model components in Paraformer ASR.

Tests model architectures, shapes, gradient flow, and tensor contracts.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.transformer import MultiHeadAttention, TransformerBlock, PositionalEncoding
from src.models.encoder import TransformerEncoder
from src.models.predictor import Predictor
from src.models.decoder import Decoder
from src.models.paraformer import Paraformer
from src.data.tokenizer import CharTokenizer
from src.data.utils import create_attention_mask


class TestTransformerComponents:
    """Test transformer building blocks."""
    
    def test_multi_head_attention(self):
        """Test MultiHeadAttention implementation."""
        d_model = 256
        num_heads = 8
        seq_len = 50
        batch_size = 2
        
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        # Test self-attention
        x = torch.randn(batch_size, seq_len, d_model)
        out = mha(x, x, x)
        
        assert out.shape == (batch_size, seq_len, d_model)
        assert out.dtype == torch.float32
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        
        # Test with attention mask
        mask = torch.tril(torch.ones(seq_len, seq_len))  # Causal mask
        out_masked = mha(x, x, x, mask=mask)
        
        assert out_masked.shape == (batch_size, seq_len, d_model)
        assert not torch.allclose(out, out_masked, atol=1e-6)  # Should be different
        
        # Test gradient flow
        loss = out.sum()
        loss.backward()
        
        assert mha.query_proj.weight.grad is not None
        assert mha.key_proj.weight.grad is not None
        assert mha.value_proj.weight.grad is not None
        assert mha.out_proj.weight.grad is not None
    
    def test_transformer_block(self):
        """Test complete TransformerBlock."""
        d_model = 256
        num_heads = 8
        d_ff = 1024
        seq_len = 50
        batch_size = 2
        
        block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=0.1
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        out = block(x)
        
        assert out.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        
        # Test with mask
        mask = create_attention_mask(
            torch.tensor([seq_len, seq_len-10]), 
            seq_len
        )
        out_masked = block(x, mask=mask)
        
        assert out_masked.shape == (batch_size, seq_len, d_model)
        
        # Test gradient flow
        loss = out.sum()
        loss.backward()
        
        assert block.attention.query_proj.weight.grad is not None
        assert block.feed_forward[0].weight.grad is not None
    
    def test_positional_encoding(self):
        """Test PositionalEncoding."""
        d_model = 256
        max_len = 1000
        
        pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        seq_len = 100
        batch_size = 2
        x = torch.randn(batch_size, seq_len, d_model)
        
        out = pos_enc(x)
        
        assert out.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        
        # Test that positional encoding is added (not just passed through)
        assert not torch.allclose(x, out, atol=1e-6)
        
        # Test with different sequence lengths
        x_short = torch.randn(batch_size, 20, d_model)
        out_short = pos_enc(x_short)
        assert out_short.shape == (batch_size, 20, d_model)


class TestEncoder:
    """Test Transformer encoder."""
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = TransformerEncoder(
            input_dim=80,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            dropout=0.1
        )
        
        batch_size = 2
        seq_len = 100
        input_dim = 80
        
        # Test forward pass
        features = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([seq_len, seq_len-20])
        
        out = encoder(features, lengths)
        
        assert out.shape == (batch_size, seq_len, 256)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        
        # Test gradient flow
        loss = out.sum()
        loss.backward()
        
        assert encoder.input_projection.weight.grad is not None
        assert encoder.layers[0].attention.query_proj.weight.grad is not None
    
    def test_encoder_with_padding(self):
        """Test encoder with padded sequences."""
        encoder = TransformerEncoder(
            input_dim=80,
            d_model=128,
            num_heads=4,
            num_layers=2
        )
        
        batch_size = 3
        max_len = 50
        
        # Create sequences with different lengths
        features = torch.randn(batch_size, max_len, 80)
        lengths = torch.tensor([50, 30, 40])
        
        out = encoder(features, lengths)
        
        assert out.shape == (batch_size, max_len, 128)
        
        # Check that padded positions are handled properly
        # (exact behavior depends on implementation, but should not be NaN/Inf)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_encoder_different_sizes(self):
        """Test encoder with different input sizes."""
        encoder = TransformerEncoder(
            input_dim=40,
            d_model=128,
            num_heads=4,
            num_layers=2
        )
        
        # Test different sequence lengths
        test_lengths = [10, 50, 100]
        
        for seq_len in test_lengths:
            features = torch.randn(1, seq_len, 40)
            lengths = torch.tensor([seq_len])
            
            out = encoder(features, lengths)
            
            assert out.shape == (1, seq_len, 128)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


class TestPredictor:
    """Test Predictor module."""
    
    def test_predictor_forward(self):
        """Test predictor forward pass."""
        predictor = Predictor(
            input_dim=256,
            hidden_dim=128,
            num_layers=2
        )
        
        batch_size = 2
        seq_len = 100
        input_dim = 256
        
        encoder_out = torch.randn(batch_size, seq_len, input_dim)
        lengths = torch.tensor([seq_len, seq_len-20])
        
        predictions = predictor(encoder_out, lengths)
        
        assert predictions.shape == (batch_size, seq_len, 1)
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
        
        # Predictions should be positive (representing durations/alignments)
        assert (predictions >= 0).all()
        
        # Test gradient flow
        loss = predictions.sum()
        loss.backward()
        
        assert predictor.layers[0].weight.grad is not None
        assert predictor.output_proj.weight.grad is not None
    
    def test_predictor_output_range(self):
        """Test that predictor outputs are in reasonable range."""
        predictor = Predictor(input_dim=128, hidden_dim=64, num_layers=1)
        
        # Test with various inputs
        for _ in range(5):
            encoder_out = torch.randn(1, 50, 128)
            lengths = torch.tensor([50])
            
            predictions = predictor(encoder_out, lengths)
            
            # Should be non-negative and not too large
            assert (predictions >= 0).all()
            assert (predictions <= 10.0).all()  # Reasonable upper bound
    
    def test_predictor_with_masking(self):
        """Test predictor with sequence masking."""
        predictor = Predictor(input_dim=128, hidden_dim=64, num_layers=1)
        
        batch_size = 2
        max_len = 50
        
        encoder_out = torch.randn(batch_size, max_len, 128)
        lengths = torch.tensor([50, 30])  # Second sequence is shorter
        
        predictions = predictor(encoder_out, lengths)
        
        assert predictions.shape == (batch_size, max_len, 1)
        
        # Predictions for padded positions should be handled appropriately
        # (exact behavior depends on implementation)
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()


class TestDecoder:
    """Test Decoder module."""
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        vocab_size = 100
        decoder = Decoder(
            encoder_dim=256,
            vocab_size=vocab_size,
            hidden_dim=128,
            num_layers=2
        )
        
        batch_size = 2
        seq_len = 100
        
        encoder_out = torch.randn(batch_size, seq_len, 256)
        predictor_out = torch.randn(batch_size, seq_len, 1)
        lengths = torch.tensor([seq_len, seq_len-20])
        
        logits = decoder(encoder_out, predictor_out, lengths)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        
        # Test gradient flow
        loss = logits.sum()
        loss.backward()
        
        assert decoder.attention.query_proj.weight.grad is not None
        assert decoder.output_proj.weight.grad is not None
    
    def test_decoder_attention_mechanism(self):
        """Test decoder attention over encoder outputs."""
        vocab_size = 50
        decoder = Decoder(
            encoder_dim=128,
            vocab_size=vocab_size,
            hidden_dim=64,
            num_layers=1
        )
        
        batch_size = 1
        seq_len = 20
        
        # Create encoder output with distinct patterns
        encoder_out = torch.zeros(batch_size, seq_len, 128)
        encoder_out[:, :10, :] = 1.0  # First half has different pattern
        encoder_out[:, 10:, :] = -1.0  # Second half has different pattern
        
        predictor_out = torch.ones(batch_size, seq_len, 1) * 0.5
        lengths = torch.tensor([seq_len])
        
        logits = decoder(encoder_out, predictor_out, lengths)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        
        # Logits should vary across sequence (due to attention over different encoder states)
        logits_first_half = logits[:, :10, :].mean()
        logits_second_half = logits[:, 10:, :].mean()
        
        # Should be different (though exact difference depends on initialization)
        assert not torch.allclose(logits_first_half, logits_second_half, atol=1e-3)
    
    def test_decoder_with_predictor_conditioning(self):
        """Test that decoder uses predictor information."""
        vocab_size = 30
        decoder = Decoder(
            encoder_dim=64,
            vocab_size=vocab_size,
            hidden_dim=32,
            num_layers=1
        )
        
        batch_size = 1
        seq_len = 10
        
        encoder_out = torch.randn(batch_size, seq_len, 64)
        lengths = torch.tensor([seq_len])
        
        # Test with different predictor outputs
        predictor_out1 = torch.ones(batch_size, seq_len, 1) * 0.1
        predictor_out2 = torch.ones(batch_size, seq_len, 1) * 0.9
        
        logits1 = decoder(encoder_out, predictor_out1, lengths)
        logits2 = decoder(encoder_out, predictor_out2, lengths)
        
        # Outputs should be different when predictor outputs are different
        assert not torch.allclose(logits1, logits2, atol=1e-3)


class TestParaformerModel:
    """Test complete Paraformer model."""
    
    def test_paraformer_forward(self):
        """Test Paraformer forward pass."""
        vocab_size = 100
        model = Paraformer(
            input_dim=80,
            vocab_size=vocab_size,
            encoder_dim=256,
            encoder_layers=4,
            encoder_heads=8,
            predictor_dim=128,
            decoder_dim=128
        )
        
        batch_size = 2
        seq_len = 100
        
        features = torch.randn(batch_size, seq_len, 80)
        lengths = torch.tensor([seq_len, seq_len-20])
        
        outputs = model(features, lengths)
        
        # Check output structure
        assert 'logits' in outputs
        assert 'predictor_out' in outputs
        assert 'encoder_out' in outputs
        
        # Check shapes
        logits = outputs['logits']
        predictor_out = outputs['predictor_out']
        encoder_out = outputs['encoder_out']
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert predictor_out.shape == (batch_size, seq_len, 1)
        assert encoder_out.shape == (batch_size, seq_len, 256)
        
        # Check for NaN/Inf
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert not torch.isnan(predictor_out).any()
        assert not torch.isinf(predictor_out).any()
    
    def test_paraformer_gradient_flow(self):
        """Test gradient flow through complete model."""
        vocab_size = 50
        model = Paraformer(
            input_dim=40,
            vocab_size=vocab_size,
            encoder_dim=128,
            encoder_layers=2,
            encoder_heads=4,
            predictor_dim=64,
            decoder_dim=64
        )
        
        batch_size = 1
        seq_len = 50
        
        features = torch.randn(batch_size, seq_len, 40, requires_grad=True)
        lengths = torch.tensor([seq_len])
        
        outputs = model(features, lengths)
        
        # Compute loss and backpropagate
        loss = outputs['logits'].sum()
        loss.backward()
        
        # Check input gradients
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        assert not torch.isinf(features.grad).any()
        
        # Check that all model components have gradients
        # Encoder
        assert model.encoder.input_projection.weight.grad is not None
        assert model.encoder.layers[0].attention.query_proj.weight.grad is not None
        
        # Predictor
        assert model.predictor.layers[0].weight.grad is not None
        
        # Decoder
        assert model.decoder.attention.query_proj.weight.grad is not None
        assert model.decoder.output_proj.weight.grad is not None
    
    def test_paraformer_different_lengths(self):
        """Test Paraformer with different sequence lengths."""
        vocab_size = 30
        model = Paraformer(
            input_dim=40,
            vocab_size=vocab_size,
            encoder_dim=64,
            encoder_layers=1,
            encoder_heads=2,
            predictor_dim=32,
            decoder_dim=32
        )
        
        # Test different lengths
        test_lengths = [10, 25, 50]
        
        for seq_len in test_lengths:
            features = torch.randn(1, seq_len, 40)
            lengths = torch.tensor([seq_len])
            
            outputs = model(features, lengths)
            
            assert outputs['logits'].shape == (1, seq_len, vocab_size)
            assert outputs['predictor_out'].shape == (1, seq_len, 1)
            assert outputs['encoder_out'].shape == (1, seq_len, 64)
            
            assert not torch.isnan(outputs['logits']).any()
            assert not torch.isinf(outputs['logits']).any()
    
    def test_paraformer_batch_processing(self):
        """Test Paraformer with batched inputs of different lengths."""
        vocab_size = 40
        model = Paraformer(
            input_dim=80,
            vocab_size=vocab_size,
            encoder_dim=128,
            encoder_layers=2,
            encoder_heads=4,
            predictor_dim=64,
            decoder_dim=64
        )
        
        batch_size = 3
        max_len = 60
        
        features = torch.randn(batch_size, max_len, 80)
        lengths = torch.tensor([60, 40, 50])  # Different lengths
        
        outputs = model(features, lengths)
        
        assert outputs['logits'].shape == (batch_size, max_len, vocab_size)
        assert outputs['predictor_out'].shape == (batch_size, max_len, 1)
        assert outputs['encoder_out'].shape == (batch_size, max_len, 128)
        
        # All outputs should be valid (no NaN/Inf)
        for key, tensor in outputs.items():
            assert not torch.isnan(tensor).any(), f"NaN found in {key}"
            assert not torch.isinf(tensor).any(), f"Inf found in {key}"
    
    def test_paraformer_parameter_count(self):
        """Test that Paraformer has reasonable parameter count."""
        vocab_size = 100
        model = Paraformer(
            input_dim=80,
            vocab_size=vocab_size,
            encoder_dim=256,
            encoder_layers=4,
            encoder_heads=8,
            predictor_dim=128,
            decoder_dim=128
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable
        assert total_params < 100_000_000  # Reasonable upper bound
        
        print(f"Paraformer parameter count: {total_params:,}")


class TestModelIntegration:
    """Test model integration with tokenizer and data."""
    
    def test_model_with_tokenizer(self):
        """Test model integration with tokenizer."""
        # Create tokenizer
        tokenizer = CharTokenizer(vocab_size=50)
        
        # Create model
        model = Paraformer(
            input_dim=40,
            vocab_size=tokenizer.get_vocab_size(),
            encoder_dim=128,
            encoder_layers=2,
            encoder_heads=4,
            predictor_dim=64,
            decoder_dim=64
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 30
        
        features = torch.randn(batch_size, seq_len, 40)
        lengths = torch.tensor([seq_len, seq_len-10])
        
        outputs = model(features, lengths)
        logits = outputs['logits']
        
        # Test that logits can be used for token prediction
        predicted_tokens = torch.argmax(logits, dim=-1)
        
        assert predicted_tokens.shape == (batch_size, seq_len)
        assert predicted_tokens.min() >= 0
        assert predicted_tokens.max() < tokenizer.get_vocab_size()
        
        # Test decoding
        for i in range(batch_size):
            length = lengths[i].item()
            tokens = predicted_tokens[i, :length]
            decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Should produce some text (even if nonsensical for untrained model)
            assert isinstance(decoded_text, str)
    
    def test_model_training_step(self):
        """Test a single training step."""
        tokenizer = CharTokenizer(vocab_size=30)
        
        model = Paraformer(
            input_dim=40,
            vocab_size=tokenizer.get_vocab_size(),
            encoder_dim=64,
            encoder_layers=1,
            encoder_heads=2,
            predictor_dim=32,
            decoder_dim=32
        )
        
        # Create sample batch
        batch_size = 2
        feat_len = 20
        token_len = 10
        
        features = torch.randn(batch_size, feat_len, 40)
        target_tokens = torch.randint(0, tokenizer.get_vocab_size(), (batch_size, token_len))
        feature_lengths = torch.tensor([feat_len, feat_len-5])
        token_lengths = torch.tensor([token_len, token_len-3])
        
        # Forward pass
        outputs = model(features, feature_lengths)
        logits = outputs['logits']
        
        # Compute loss (simplified)
        # In practice, would use proper sequence loss with masking
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, tokenizer.get_vocab_size()),
            target_tokens.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are reasonable
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"


if __name__ == "__main__":
    # Run basic tests
    print("Running ASR model tests...")
    
    # Test transformer components
    test_transformer = TestTransformerComponents()
    test_transformer.test_multi_head_attention()
    test_transformer.test_transformer_block()
    test_transformer.test_positional_encoding()
    print("✓ Transformer component tests passed")
    
    # Test encoder
    test_encoder = TestEncoder()
    test_encoder.test_encoder_forward()
    test_encoder.test_encoder_with_padding()
    print("✓ Encoder tests passed")
    
    # Test predictor
    test_predictor = TestPredictor()
    test_predictor.test_predictor_forward()
    test_predictor.test_predictor_output_range()
    print("✓ Predictor tests passed")
    
    # Test decoder
    test_decoder = TestDecoder()
    test_decoder.test_decoder_forward()
    test_decoder.test_decoder_attention_mechanism()
    print("✓ Decoder tests passed")
    
    # Test complete model
    test_paraformer = TestParaformerModel()
    test_paraformer.test_paraformer_forward()
    test_paraformer.test_paraformer_gradient_flow()
    test_paraformer.test_paraformer_different_lengths()
    print("✓ Paraformer model tests passed")
    
    # Test integration
    test_integration = TestModelIntegration()
    test_integration.test_model_with_tokenizer()
    test_integration.test_model_training_step()
    print("✓ Model integration tests passed")
    
    print("All ASR model tests completed successfully!")