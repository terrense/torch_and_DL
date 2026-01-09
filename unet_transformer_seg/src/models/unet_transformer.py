"""U-Net + Transformer hybrid model implementation."""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .blocks import DownBlock, UpBlock, ConvBlock
from .transformer import TransformerBlock, PositionalEncoding2D
from .registry import register_model
from ..config import ModelConfig
from ..utils.tensor_utils import validate_tensor, assert_shape


class TransformerBottleneck(nn.Module):
    """
    Transformer-based bottleneck for U-Net hybrid model.
    
    Tensor Contract:
    - Input: [B, C, H, W]
    - Output: [B, C, H, W]
    - Internal processing: [B, C, H, W] -> [B, H*W, C] -> [B, H*W, C] -> [B, C, H, W]
    """
    
    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        num_heads: int = 8,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_height: int = 64,
        max_width: int = 64
    ):
        """
        Initialize TransformerBottleneck.
        
        Args:
            channels: Number of input/output channels
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension for feed-forward (defaults to 4 * channels)
            dropout: Dropout probability
            max_height: Maximum height for positional encoding
            max_width: Maximum width for positional encoding
        """
        super().__init__()
        
        self.channels = channels
        self.num_layers = num_layers
        
        # Positional encoding for spatial tokens
        self.pos_encoding = PositionalEncoding2D(
            embed_dim=channels,
            max_height=max_height,
            max_width=max_width
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=channels,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation="gelu",
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization for final output
        self.final_norm = nn.LayerNorm(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer bottleneck.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Validate input
        assert_shape(x, f"B,{self.channels},H,W", "input")
        
        # Reshape from [B, C, H, W] to [B, H*W, C]
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        
        # Add positional encoding
        x_pos = self.pos_encoding(x_flat, height=H, width=W)
        
        # Apply transformer layers
        current = x_pos
        for transformer_layer in self.transformer_layers:
            current, _ = transformer_layer(current)
        
        # Apply final normalization
        current = self.final_norm(current)
        
        # Reshape back to [B, C, H, W]
        output = current.transpose(1, 2).view(B, C, H, W)
        
        # Validate output
        assert_shape(output, f"B,{self.channels},H,W", "output")
        
        return output


@register_model("unet_transformer")
class UNetTransformer(nn.Module):
    """
    U-Net + Transformer hybrid model for image segmentation.
    
    Architecture:
    - Encoder: Series of downsampling blocks (same as pure U-Net)
    - Bottleneck: Transformer-based processing with spatial attention
    - Decoder: Series of upsampling blocks (same as pure U-Net)
    - Output: Final classification layer
    
    Tensor Contract:
    - Input: [B, input_channels, H, W]
    - Output: [B, num_classes, H, W]
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize U-Net + Transformer hybrid model.
        
        Args:
            config: Model configuration containing architecture parameters
        """
        super().__init__()
        
        self.config = config
        self.input_channels = config.input_channels
        self.num_classes = config.num_classes
        self.encoder_channels = config.encoder_channels
        self.decoder_channels = config.decoder_channels
        
        # Transformer-specific parameters
        self.use_transformer = config.use_transformer
        self.transformer_layers = config.transformer_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        
        # Validate configuration
        self._validate_config()
        
        # Build encoder path
        self.encoder = self._build_encoder()
        
        # Build transformer bottleneck
        self.bottleneck = self._build_transformer_bottleneck()
        
        # Build decoder path
        self.decoder = self._build_decoder()
        
        # Final output layer
        self.output_conv = nn.Conv2d(
            in_channels=self.decoder_channels[-1],
            out_channels=self.num_classes,
            kernel_size=1,
            padding=0
        )
        
        # Initialize weights
        self._init_weights()
    
    def _validate_config(self):
        """Validate model configuration."""
        if len(self.encoder_channels) == 0:
            raise ValueError("encoder_channels cannot be empty")
        
        if len(self.decoder_channels) == 0:
            raise ValueError("decoder_channels cannot be empty")
        
        if len(self.encoder_channels) != len(self.decoder_channels):
            raise ValueError(
                f"encoder_channels ({len(self.encoder_channels)}) and "
                f"decoder_channels ({len(self.decoder_channels)}) must have same length"
            )
        
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {self.input_channels}")
        
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        
        if self.transformer_layers <= 0:
            raise ValueError(f"transformer_layers must be positive, got {self.transformer_layers}")
        
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        
        # Check if bottleneck channels are divisible by num_heads
        bottleneck_channels = self.encoder_channels[-1] * 2
        if bottleneck_channels % self.num_heads != 0:
            raise ValueError(
                f"Bottleneck channels ({bottleneck_channels}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
    
    def _build_encoder(self) -> nn.ModuleList:
        """Build encoder path with downsampling blocks."""
        encoder = nn.ModuleList()
        
        # First block: input_channels -> encoder_channels[0]
        in_channels = self.input_channels
        
        for out_channels in self.encoder_channels:
            encoder.append(
                DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    downsample_type="maxpool",
                    norm_type="batch",
                    activation="relu",
                    dropout=0.0
                )
            )
            in_channels = out_channels
        
        return encoder
    
    def _build_transformer_bottleneck(self) -> nn.Module:
        """Build transformer-based bottleneck."""
        # Input channels from last encoder block
        in_channels = self.encoder_channels[-1]
        
        # Expand channels for bottleneck processing
        bottleneck_channels = in_channels * 2
        
        # Input projection to expand channels
        input_proj = ConvBlock(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            padding=1,
            norm_type="batch",
            activation="relu",
            dropout=self.dropout
        )
        
        # Transformer bottleneck
        transformer_bottleneck = TransformerBottleneck(
            channels=bottleneck_channels,
            num_layers=self.transformer_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            max_height=64,  # Reasonable maximum for bottleneck
            max_width=64
        )
        
        # Output projection to maintain channel consistency
        output_proj = ConvBlock(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            padding=1,
            norm_type="batch",
            activation="relu",
            dropout=self.dropout
        )
        
        return nn.Sequential(
            input_proj,
            transformer_bottleneck,
            output_proj
        )
    
    def _build_decoder(self) -> nn.ModuleList:
        """Build decoder path with upsampling blocks."""
        decoder = nn.ModuleList()
        
        # Start from bottleneck channels
        bottleneck_channels = self.encoder_channels[-1] * 2
        in_channels = bottleneck_channels
        
        # Build decoder blocks in reverse order
        for i, out_channels in enumerate(self.decoder_channels):
            # Skip connection comes from corresponding encoder level
            skip_channels = self.encoder_channels[-(i+1)]
            
            decoder.append(
                UpBlock(
                    in_channels=in_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    upsample_type="transpose",
                    norm_type="batch",
                    activation="relu",
                    dropout=0.0
                )
            )
            in_channels = out_channels
        
        return decoder
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net + Transformer hybrid.
        
        Args:
            x: Input tensor [B, input_channels, H, W]
            
        Returns:
            Output tensor [B, num_classes, H, W]
        """
        # Validate input
        assert_shape(x, f"B,{self.input_channels},H,W", "input")
        
        # Store original spatial dimensions
        original_height, original_width = x.shape[2], x.shape[3]
        
        # Encoder path - collect skip connections
        skip_connections = []
        current = x
        
        for encoder_block in self.encoder:
            current, skip = encoder_block(current)
            skip_connections.append(skip)
        
        # Transformer bottleneck
        current = self.bottleneck(current)
        
        # Decoder path - use skip connections in reverse order
        skip_connections = skip_connections[::-1]  # Reverse order
        
        for i, decoder_block in enumerate(self.decoder):
            skip = skip_connections[i]
            current = decoder_block(current, skip)
        
        # Final output layer
        output = self.output_conv(current)
        
        # Ensure output matches input spatial dimensions
        if output.shape[2:] != (original_height, original_width):
            output = torch.nn.functional.interpolate(
                output,
                size=(original_height, original_width),
                mode='bilinear',
                align_corners=False
            )
        
        # Validate output
        assert_shape(output, f"B,{self.num_classes},{original_height},{original_width}", "output")
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> dict:
        """
        Get attention maps from transformer bottleneck.
        
        Args:
            x: Input tensor [B, input_channels, H, W]
            
        Returns:
            Dictionary containing attention maps from each transformer layer
        """
        # Run through encoder to get to bottleneck
        current = x
        for encoder_block in self.encoder:
            current, _ = encoder_block(current)
        
        # Get bottleneck input
        bottleneck_input = self.bottleneck[0](current)  # Input projection
        
        # Process through transformer layers and collect attention
        B, C, H, W = bottleneck_input.shape
        x_flat = bottleneck_input.view(B, C, H * W).transpose(1, 2)
        
        # Add positional encoding
        transformer_bottleneck = self.bottleneck[1]  # TransformerBottleneck
        x_pos = transformer_bottleneck.pos_encoding(x_flat, height=H, width=W)
        
        # Collect attention maps from each layer
        attention_maps = {}
        current_x = x_pos
        
        for i, transformer_layer in enumerate(transformer_bottleneck.transformer_layers):
            current_x, attn_weights = transformer_layer(current_x, return_attention=True)
            if attn_weights is not None:
                # Reshape attention weights for visualization
                # attn_weights: [B, H, T, T] -> [B, H, H_spatial, W_spatial, H_spatial, W_spatial]
                attn_reshaped = attn_weights.view(B, transformer_bottleneck.num_heads, H, W, H, W)
                attention_maps[f'layer_{i}'] = attn_reshaped
        
        return attention_maps
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Get intermediate feature maps for visualization/analysis.
        
        Args:
            x: Input tensor [B, input_channels, H, W]
            
        Returns:
            Dictionary containing feature maps at different levels
        """
        features = {'input': x}
        
        # Encoder features
        skip_connections = []
        current = x
        
        for i, encoder_block in enumerate(self.encoder):
            current, skip = encoder_block(current)
            skip_connections.append(skip)
            features[f'encoder_{i}'] = skip
            features[f'encoder_{i}_down'] = current
        
        # Bottleneck features
        bottleneck_input = self.bottleneck[0](current)  # Input projection
        features['bottleneck_input'] = bottleneck_input
        
        transformer_output = self.bottleneck[1](bottleneck_input)  # Transformer
        features['bottleneck_transformer'] = transformer_output
        
        bottleneck_output = self.bottleneck[2](transformer_output)  # Output projection
        features['bottleneck_output'] = bottleneck_output
        
        current = bottleneck_output
        
        # Decoder features
        skip_connections = skip_connections[::-1]
        
        for i, decoder_block in enumerate(self.decoder):
            skip = skip_connections[i]
            current = decoder_block(current, skip)
            features[f'decoder_{i}'] = current
        
        # Final output
        output = self.output_conv(current)
        features['output'] = output
        
        return features
    
    def count_parameters(self) -> dict:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count parameters by component
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        bottleneck_params = sum(p.numel() for p in self.bottleneck.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        output_params = sum(p.numel() for p in self.output_conv.parameters())
        
        # Break down bottleneck parameters
        input_proj_params = sum(p.numel() for p in self.bottleneck[0].parameters())
        transformer_params = sum(p.numel() for p in self.bottleneck[1].parameters())
        output_proj_params = sum(p.numel() for p in self.bottleneck[2].parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': encoder_params,
            'bottleneck': bottleneck_params,
            'bottleneck_input_proj': input_proj_params,
            'bottleneck_transformer': transformer_params,
            'bottleneck_output_proj': output_proj_params,
            'decoder': decoder_params,
            'output': output_params
        }
    
    def get_model_info(self) -> dict:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        param_counts = self.count_parameters()
        
        return {
            'model_type': 'UNetTransformer',
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'transformer_layers': self.transformer_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'num_encoder_blocks': len(self.encoder),
            'num_decoder_blocks': len(self.decoder),
            'parameters': param_counts
        }


def create_unet_transformer(config: ModelConfig) -> UNetTransformer:
    """
    Factory function to create U-Net + Transformer hybrid model.
    
    Args:
        config: Model configuration
        
    Returns:
        U-Net + Transformer hybrid model instance
    """
    return UNetTransformer(config)