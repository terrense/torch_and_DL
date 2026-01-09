"""Basic building blocks for U-Net architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..utils.tensor_utils import validate_tensor, assert_shape


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv->Norm->Activation pattern.
    
    Tensor Contract:
    - Input: [B, in_channels, H, W]
    - Output: [B, out_channels, H, W] (same spatial dimensions)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize ConvBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            stride: Stride size
            norm_type: Normalization type ('batch', 'instance', 'group', 'none')
            activation: Activation function ('relu', 'leaky_relu', 'gelu', 'none')
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=(norm_type == "none")  # Use bias only if no normalization
        )
        
        # Normalization layer
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == "group":
            # Use 8 groups by default, but ensure it divides out_channels
            num_groups = min(8, out_channels)
            while out_channels % num_groups != 0:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups, out_channels)
        elif norm_type == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation type: {activation}")
        
        # Dropout layer
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ConvBlock.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        # Validate input
        assert_shape(x, f"B,{self.in_channels},H,W", "input")
        
        # Apply layers in sequence
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Validate output
        assert_shape(x, f"B,{self.out_channels},H,W", "output")
        
        return x


class DoubleConvBlock(nn.Module):
    """
    Double convolution block commonly used in U-Net.
    
    Tensor Contract:
    - Input: [B, in_channels, H, W]
    - Output: [B, out_channels, H, W]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize DoubleConvBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels (defaults to out_channels)
            norm_type: Normalization type
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout
        )
        
        self.conv2 = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DoubleConvBlock."""
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder.
    
    Tensor Contract:
    - Input: [B, in_channels, H, W]
    - Output: [B, out_channels, H//2, W//2]
    - Skip connection: [B, out_channels, H, W] (before downsampling)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_type: str = "maxpool",
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize DownBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            downsample_type: Downsampling method ('maxpool', 'conv', 'avgpool')
            norm_type: Normalization type
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Double convolution block
        self.conv_block = DoubleConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout
        )
        
        # Downsampling layer
        if downsample_type == "maxpool":
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample_type == "avgpool":
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        elif downsample_type == "conv":
            self.downsample = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2
            )
        else:
            raise ValueError(f"Unknown downsample type: {downsample_type}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DownBlock.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            
        Returns:
            Tuple of (downsampled_output, skip_connection)
            - downsampled_output: [B, out_channels, H//2, W//2]
            - skip_connection: [B, out_channels, H, W]
        """
        # Validate input
        assert_shape(x, f"B,{self.in_channels},H,W", "input")
        
        # Apply convolution block
        skip = self.conv_block(x)
        
        # Apply downsampling
        down = self.downsample(skip)
        
        # Validate outputs
        B, C, H, W = skip.shape
        assert_shape(skip, f"B,{self.out_channels},H,W", "skip_connection")
        assert_shape(down, f"B,{self.out_channels},{H//2},{W//2}", "downsampled_output")
        
        return down, skip


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder with skip connections.
    
    Tensor Contract:
    - Input: [B, in_channels, H, W]
    - Skip connection: [B, skip_channels, H*2, W*2]
    - Output: [B, out_channels, H*2, W*2]
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_type: str = "transpose",
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        """
        Initialize UpBlock.
        
        Args:
            in_channels: Number of input channels
            skip_channels: Number of skip connection channels
            out_channels: Number of output channels
            upsample_type: Upsampling method ('transpose', 'bilinear', 'nearest')
            norm_type: Normalization type
            activation: Activation function
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        
        # Upsampling layer
        if upsample_type == "transpose":
            self.upsample = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=2,
                stride=2
            )
        elif upsample_type in ["bilinear", "nearest"]:
            self.upsample = nn.Upsample(
                scale_factor=2,
                mode=upsample_type,
                align_corners=False if upsample_type == "bilinear" else None
            )
        else:
            raise ValueError(f"Unknown upsample type: {upsample_type}")
        
        # Convolution block after concatenation
        concat_channels = in_channels + skip_channels
        self.conv_block = DoubleConvBlock(
            in_channels=concat_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UpBlock.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            skip: Skip connection tensor [B, skip_channels, H*2, W*2]
            
        Returns:
            Output tensor [B, out_channels, H*2, W*2]
        """
        # Validate inputs
        assert_shape(x, f"B,{self.in_channels},H,W", "input")
        assert_shape(skip, f"B,{self.skip_channels},H2,W2", "skip_connection")
        
        # Upsample input
        x_up = self.upsample(x)
        
        # Ensure spatial dimensions match for concatenation
        if x_up.shape[2:] != skip.shape[2:]:
            # Resize upsampled tensor to match skip connection
            x_up = F.interpolate(
                x_up,
                size=skip.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate along channel dimension
        x_concat = torch.cat([x_up, skip], dim=1)
        
        # Apply convolution block
        output = self.conv_block(x_concat)
        
        # Validate output
        assert_shape(output, f"B,{self.out_channels},H2,W2", "output")
        
        return output


class BottleneckBlock(nn.Module):
    """
    Bottleneck block for the center of U-Net.
    
    Tensor Contract:
    - Input: [B, in_channels, H, W]
    - Output: [B, out_channels, H, W]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.1
    ):
        """
        Initialize BottleneckBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            norm_type: Normalization type
            activation: Activation function
            dropout: Dropout probability (higher default for bottleneck)
        """
        super().__init__()
        
        self.conv_block = DoubleConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through BottleneckBlock."""
        return self.conv_block(x)