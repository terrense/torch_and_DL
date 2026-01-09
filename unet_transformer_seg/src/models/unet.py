"""Pure U-Net baseline model implementation."""

import torch
import torch.nn as nn
from typing import List, Optional

from .blocks import DownBlock, UpBlock, BottleneckBlock, ConvBlock
from .registry import register_model
from ..config import ModelConfig
from ..utils.tensor_utils import validate_tensor, assert_shape


@register_model("unet_baseline")
class UNet(nn.Module):
    """
    Pure U-Net model for image segmentation.
    
    Architecture:
    - Encoder: Series of downsampling blocks with skip connections
    - Bottleneck: Central processing block
    - Decoder: Series of upsampling blocks with skip connection fusion
    - Output: Final classification layer
    
    Tensor Contract:
    - Input: [B, input_channels, H, W]
    - Output: [B, num_classes, H, W]
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize U-Net model.
        
        Args:
            config: Model configuration containing architecture parameters
        """
        super().__init__()
        
        self.config = config
        self.input_channels = config.input_channels
        self.num_classes = config.num_classes
        self.encoder_channels = config.encoder_channels
        self.decoder_channels = config.decoder_channels
        
        # Validate configuration
        self._validate_config()
        
        # Build encoder path
        self.encoder = self._build_encoder()
        
        # Build bottleneck
        self.bottleneck = self._build_bottleneck()
        
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
    
    def _build_bottleneck(self) -> BottleneckBlock:
        """Build bottleneck block."""
        bottleneck_channels = self.encoder_channels[-1] * 2
        
        return BottleneckBlock(
            in_channels=self.encoder_channels[-1],
            out_channels=bottleneck_channels,
            norm_type="batch",
            activation="relu",
            dropout=0.1
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
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
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
        
        # Bottleneck
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
        current = self.bottleneck(current)
        features['bottleneck'] = current
        
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
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': encoder_params,
            'bottleneck': bottleneck_params,
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
            'model_type': 'UNet',
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'num_encoder_blocks': len(self.encoder),
            'num_decoder_blocks': len(self.decoder),
            'parameters': param_counts
        }


def create_unet_baseline(config: ModelConfig) -> UNet:
    """
    Factory function to create U-Net baseline model.
    
    Args:
        config: Model configuration
        
    Returns:
        U-Net model instance
    """
    return UNet(config)