"""
Unit tests for model components in U-Net Transformer Segmentation.

Tests model architectures, shapes, gradient flow, and tensor contracts.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.blocks import ConvBlock, DownBlock, UpBlock
from src.models.transformer import MultiHeadAttention, TransformerBlock, PositionalEncoding
from src.models.unet import UNet
from src.models.unet_transformer import UNetTransformer
from src.models.registry import ModelRegistry
from src.utils.tensor_utils import assert_shape, check_nan_inf
from src.config import ModelConfig


class TestBasicBlocks:
    """Test basic building blocks."""
    
    def test_conv_block(self):
        """Test ConvBlock functionality."""
        block = ConvBlock(in_channels=64, out_channels=128, kernel_size=3)
        
        # Test forward pass
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        
        assert_shape(out, "2,128,32,32", "conv_block_output")
        check_nan_inf(out, "conv_block_output")
        
        # Test gradient flow
        loss = out.sum()
        loss.backward()
        
        assert block.conv.weight.grad is not None
        check_nan_inf(block.conv.weight.grad, "conv_block_grad")
    
    def test_down_block(self):
        """Test DownBlock with skip connections."""
        block = DownBlock(in_channels=64, out_channels=128)
        
        x = torch.randn(2, 64, 64, 64)
        out, skip = block(x)
        
        # Output should be downsampled
        assert_shape(out, "2,128,32,32", "down_block_output")
        # Skip should maintain input resolution
        assert_shape(skip, "2,128,64,64", "down_block_skip")
        
        check_nan_inf(out, "down_block_output")
        check_nan_inf(skip, "down_block_skip")
    
    def test_up_block(self):
        """Test UpBlock with skip connection fusion."""
        block = UpBlock(in_channels=128, skip_channels=64, out_channels=64)
        
        x = torch.randn(2, 128, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        
        out = block(x, skip)
        
        # Output should match skip resolution
        assert_shape(out, "2,64,32,32", "up_block_output")
        check_nan_inf(out, "up_block_output")
        
        # Test gradient flow
        loss = out.sum()
        loss.backward()
        
        assert block.conv.conv.weight.grad is not None


class TestTransformerComponents:
    """Test transformer components."""
    
    def test_multi_head_attention(self):
        """Test MultiHeadAttention implementation."""
        d_model = 256
        num_heads = 8
        seq_len = 64
        batch_size = 2
        
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        # Test self-attention
        x = torch.randn(batch_size, seq_len, d_model)
        out = mha(x, x, x)
        
        assert_shape(out, f"{batch_size},{seq_len},{d_model}", "mha_output")
        check_nan_inf(out, "mha_output")
        
        # Test with attention mask
        mask = torch.tril(torch.ones(seq_len, seq_len))  # Causal mask
        out_masked = mha(x, x, x, mask=mask)
        
        assert_shape(out_masked, f"{batch_size},{seq_len},{d_model}", "mha_masked_output")
        check_nan_inf(out_masked, "mha_masked_output")
        
        # Outputs should be different with mask
        assert not torch.allclose(out, out_masked, atol=1e-6)
    
    def test_transformer_block(self):
        """Test complete TransformerBlock."""
        d_model = 256
        num_heads = 8
        d_ff = 1024
        seq_len = 64
        batch_size = 2
        
        block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=0.1
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        out = block(x)
        
        assert_shape(out, f"{batch_size},{seq_len},{d_model}", "transformer_block_output")
        check_nan_inf(out, "transformer_block_output")
        
        # Test gradient flow
        loss = out.sum()
        loss.backward()
        
        assert block.attention.query_proj.weight.grad is not None
        assert block.feed_forward[0].weight.grad is not None
    
    def test_positional_encoding(self):
        """Test PositionalEncoding for spatial tokens."""
        d_model = 256
        max_len = 1024
        
        pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        # Test 1D positional encoding
        seq_len = 100
        batch_size = 2
        x = torch.randn(batch_size, seq_len, d_model)
        
        out = pos_enc(x)
        assert_shape(out, f"{batch_size},{seq_len},{d_model}", "pos_enc_output")
        check_nan_inf(out, "pos_enc_output")
        
        # Test 2D positional encoding
        h, w = 16, 16
        x_2d = torch.randn(batch_size, h * w, d_model)
        
        out_2d = pos_enc.add_2d_pos_encoding(x_2d, h, w)
        assert_shape(out_2d, f"{batch_size},{h * w},{d_model}", "pos_enc_2d_output")
        check_nan_inf(out_2d, "pos_enc_2d_output")


class TestUNetModel:
    """Test U-Net model architecture."""
    
    def test_unet_forward(self):
        """Test U-Net forward pass."""
        model = UNet(
            in_channels=3,
            num_classes=4,
            base_channels=64,
            depth=3
        )
        
        batch_size = 2
        h, w = 128, 128
        x = torch.randn(batch_size, 3, h, w)
        
        out = model(x)
        
        assert_shape(out, f"{batch_size},4,{h},{w}", "unet_output")
        check_nan_inf(out, "unet_output")
        
        # Test gradient flow
        loss = out.sum()
        loss.backward()
        
        # Check that gradients exist for key components
        assert model.encoder[0].conv.weight.grad is not None
        assert model.decoder[0].conv.conv.weight.grad is not None
        assert model.final_conv.weight.grad is not None
    
    def test_unet_different_sizes(self):
        """Test U-Net with different input sizes."""
        model = UNet(in_channels=3, num_classes=2, base_channels=32, depth=2)
        
        # Test multiple sizes
        sizes = [(64, 64), (128, 128), (256, 256)]
        
        for h, w in sizes:
            x = torch.randn(1, 3, h, w)
            out = model(x)
            
            assert_shape(out, f"1,2,{h},{w}", f"unet_output_{h}x{w}")
            check_nan_inf(out, f"unet_output_{h}x{w}")
    
    def test_unet_parameter_count(self):
        """Test that U-Net has reasonable parameter count."""
        model = UNet(in_channels=3, num_classes=4, base_channels=64, depth=3)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable
        assert total_params < 50_000_000  # Reasonable upper bound


class TestUNetTransformer:
    """Test U-Net + Transformer hybrid model."""
    
    def test_unet_transformer_forward(self):
        """Test U-Net Transformer forward pass."""
        model = UNetTransformer(
            in_channels=3,
            num_classes=4,
            base_channels=64,
            depth=3,
            transformer_layers=2,
            num_heads=8
        )
        
        batch_size = 2
        h, w = 128, 128
        x = torch.randn(batch_size, 3, h, w)
        
        out = model(x)
        
        assert_shape(out, f"{batch_size},4,{h},{w}", "unet_transformer_output")
        check_nan_inf(out, "unet_transformer_output")
        
        # Test gradient flow through transformer components
        loss = out.sum()
        loss.backward()
        
        # Check gradients in transformer layers
        assert model.transformer_layers[0].attention.query_proj.weight.grad is not None
        assert model.transformer_layers[0].feed_forward[0].weight.grad is not None
    
    def test_tensor_reshaping(self):
        """Test tensor reshaping between CNN and Transformer formats."""
        model = UNetTransformer(
            in_channels=3,
            num_classes=2,
            base_channels=32,
            depth=2,
            transformer_layers=1,
            num_heads=4
        )
        
        # Test the internal reshaping methods
        batch_size = 2
        channels = 128
        h, w = 16, 16
        
        # Test CNN to Transformer format
        cnn_features = torch.randn(batch_size, channels, h, w)
        transformer_features = model._cnn_to_transformer(cnn_features)
        
        expected_seq_len = h * w
        assert_shape(
            transformer_features, 
            f"{batch_size},{expected_seq_len},{channels}", 
            "cnn_to_transformer"
        )
        
        # Test Transformer to CNN format
        reconstructed = model._transformer_to_cnn(transformer_features, h, w)
        assert_shape(
            reconstructed, 
            f"{batch_size},{channels},{h},{w}", 
            "transformer_to_cnn"
        )
        
        # Should be able to reconstruct original (approximately)
        assert torch.allclose(cnn_features, reconstructed, atol=1e-6)
    
    def test_comparison_with_baseline(self):
        """Test that hybrid model produces different outputs than baseline."""
        # Create baseline U-Net
        unet = UNet(in_channels=3, num_classes=4, base_channels=32, depth=2)
        
        # Create hybrid model with same base architecture
        unet_transformer = UNetTransformer(
            in_channels=3,
            num_classes=4,
            base_channels=32,
            depth=2,
            transformer_layers=1,
            num_heads=4
        )
        
        x = torch.randn(1, 3, 64, 64)
        
        out_unet = unet(x)
        out_hybrid = unet_transformer(x)
        
        # Outputs should have same shape
        assert out_unet.shape == out_hybrid.shape
        
        # But should be different (since architectures are different)
        assert not torch.allclose(out_unet, out_hybrid, atol=1e-3)


class TestModelRegistry:
    """Test model registry and configuration-based building."""
    
    def test_model_registration(self):
        """Test model registration and retrieval."""
        registry = ModelRegistry()
        
        # Test that default models are registered
        assert 'unet' in registry.list_models()
        assert 'unet_transformer' in registry.list_models()
    
    def test_model_building_from_config(self):
        """Test building models from configuration."""
        registry = ModelRegistry()
        
        # Test U-Net config
        unet_config = ModelConfig(
            name='unet',
            in_channels=3,
            num_classes=5,
            base_channels=64,
            depth=3
        )
        
        model = registry.build_model(unet_config)
        assert isinstance(model, UNet)
        
        # Test with sample input
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert_shape(out, "1,5,128,128", "registry_unet_output")
        
        # Test U-Net Transformer config
        transformer_config = ModelConfig(
            name='unet_transformer',
            in_channels=3,
            num_classes=4,
            base_channels=32,
            depth=2,
            transformer_layers=2,
            num_heads=8
        )
        
        model = registry.build_model(transformer_config)
        assert isinstance(model, UNetTransformer)
        
        out = model(x)
        assert_shape(out, "1,4,128,128", "registry_transformer_output")
    
    def test_invalid_model_config(self):
        """Test handling of invalid model configurations."""
        registry = ModelRegistry()
        
        # Test unknown model name
        invalid_config = ModelConfig(name='unknown_model')
        
        with pytest.raises(ValueError, match="Unknown model"):
            registry.build_model(invalid_config)


class TestGradientFlow:
    """Test gradient flow through models."""
    
    def test_unet_gradient_flow(self):
        """Test that gradients flow properly through U-Net."""
        model = UNet(in_channels=3, num_classes=2, base_channels=32, depth=2)
        
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        
        # Compute loss and backpropagate
        loss = out.mean()
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        check_nan_inf(x.grad, "input_gradients")
        
        # Check that all model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            check_nan_inf(param.grad, f"gradient_{name}")
    
    def test_transformer_gradient_flow(self):
        """Test gradient flow through transformer components."""
        model = UNetTransformer(
            in_channels=3,
            num_classes=2,
            base_channels=32,
            depth=2,
            transformer_layers=1,
            num_heads=4
        )
        
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        
        loss = out.mean()
        loss.backward()
        
        # Check input gradients
        assert x.grad is not None
        check_nan_inf(x.grad, "transformer_input_gradients")
        
        # Check transformer-specific gradients
        transformer_layer = model.transformer_layers[0]
        assert transformer_layer.attention.query_proj.weight.grad is not None
        assert transformer_layer.attention.key_proj.weight.grad is not None
        assert transformer_layer.attention.value_proj.weight.grad is not None
        assert transformer_layer.feed_forward[0].weight.grad is not None


if __name__ == "__main__":
    # Run basic tests
    print("Running model tests...")
    
    # Test basic blocks
    test_blocks = TestBasicBlocks()
    test_blocks.test_conv_block()
    test_blocks.test_down_block()
    test_blocks.test_up_block()
    print("✓ Basic block tests passed")
    
    # Test transformer components
    test_transformer = TestTransformerComponents()
    test_transformer.test_multi_head_attention()
    test_transformer.test_transformer_block()
    test_transformer.test_positional_encoding()
    print("✓ Transformer component tests passed")
    
    # Test U-Net model
    test_unet = TestUNetModel()
    test_unet.test_unet_forward()
    test_unet.test_unet_different_sizes()
    print("✓ U-Net model tests passed")
    
    # Test U-Net Transformer
    test_hybrid = TestUNetTransformer()
    test_hybrid.test_unet_transformer_forward()
    test_hybrid.test_tensor_reshaping()
    print("✓ U-Net Transformer tests passed")
    
    # Test model registry
    test_registry = TestModelRegistry()
    test_registry.test_model_registration()
    test_registry.test_model_building_from_config()
    print("✓ Model registry tests passed")
    
    # Test gradient flow
    test_gradients = TestGradientFlow()
    test_gradients.test_unet_gradient_flow()
    test_gradients.test_transformer_gradient_flow()
    print("✓ Gradient flow tests passed")
    
    print("All model tests completed successfully!")