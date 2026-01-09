"""
U-Net + Transformer Hybrid Model Implementation for Image Segmentation

This module implements a state-of-the-art hybrid architecture that combines the spatial 
hierarchical feature extraction capabilities of U-Net with the global context modeling 
power of Vision Transformers. This approach addresses key limitations in traditional 
CNN-based segmentation models by incorporating self-attention mechanisms at the bottleneck.

Key Deep Learning Concepts:
1. Hierarchical Feature Learning: U-Net encoder-decoder structure captures multi-scale features
2. Global Context Modeling: Transformer bottleneck captures long-range spatial dependencies
3. Skip Connections: Preserve fine-grained spatial information across scales
4. Self-Attention: Enables each spatial location to attend to all other locations
5. Positional Encoding: Provides spatial awareness to transformer layers

Architecture Innovation:
- Combines CNN's translation equivariance with Transformer's global receptive field
- Addresses the quadratic complexity of pure Vision Transformers by applying attention 
  only at the compressed bottleneck representation
- Maintains U-Net's proven effectiveness for dense prediction tasks while adding 
  global context understanding

References:
- U-Net: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Vision Transformer: Dosovitskiy et al. "An Image is Worth 16x16 Words"
- TransUNet: Chen et al. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"
"""

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
    Transformer-based Bottleneck for U-Net Hybrid Architecture
    
    This module serves as the critical bridge between the CNN encoder and decoder,
    replacing traditional convolutional bottleneck with self-attention mechanisms.
    
    Deep Learning Rationale:
    - CNNs have limited receptive fields, especially at deeper layers
    - Self-attention provides global receptive field with O(n²) complexity
    - Applied at bottleneck where spatial resolution is lowest (computational efficiency)
    - Enables modeling of long-range spatial dependencies crucial for segmentation
    
    Technical Implementation:
    - Converts 2D feature maps to sequence of spatial tokens
    - Applies learnable 2D positional encoding to preserve spatial relationships
    - Processes through multiple transformer layers with residual connections
    - Converts back to 2D feature maps for decoder processing
    
    Attention Mechanism Benefits:
    - Each pixel can attend to all other pixels in the bottleneck representation
    - Learns to focus on relevant spatial regions for segmentation task
    - Captures global context that pure CNNs struggle with
    - Provides interpretable attention maps for model analysis
    
    Tensor Contract:
    - Input: [B, C, H, W] - Batch, Channels, Height, Width
    - Output: [B, C, H, W] - Same spatial dimensions preserved
    - Internal: [B, C, H, W] → [B, H*W, C] → [B, H*W, C] → [B, C, H, W]
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
        Initialize Transformer Bottleneck with Multi-Head Self-Attention
        
        Args:
            channels: Number of input/output channels (embedding dimension)
            num_layers: Number of transformer layers (depth of attention processing)
            num_heads: Number of attention heads (parallel attention computations)
            hidden_dim: Hidden dimension for feed-forward network (defaults to 4 * channels)
            dropout: Dropout probability for regularization (prevents overfitting)
            max_height: Maximum height for positional encoding (spatial awareness)
            max_width: Maximum width for positional encoding (spatial awareness)
            
        Deep Learning Design Choices:
        - num_heads=8: Standard choice balancing computational cost and representation capacity
        - hidden_dim=4*channels: Common transformer scaling factor for FFN expansion
        - dropout=0.1: Moderate regularization to prevent overfitting in attention layers
        - Layer normalization: Pre-norm architecture for stable training (norm_first=True)
        """
        super().__init__()
        
        self.channels = channels
        self.num_layers = num_layers
        
        # Positional encoding provides spatial awareness to transformer
        # Critical for maintaining spatial relationships in 2D feature maps
        self.pos_encoding = PositionalEncoding2D(
            embed_dim=channels,
            max_height=max_height,
            max_width=max_width
        )
        
        # Stack of transformer layers for deep feature processing
        # Each layer applies: LayerNorm → MultiHeadAttention → LayerNorm → FFN
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=channels,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation="gelu",  # GELU activation for smoother gradients
                norm_first=True     # Pre-norm for training stability
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization for output stabilization
        self.final_norm = nn.LayerNorm(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass Through Transformer Bottleneck
        
        This method implements the core transformation from CNN feature maps to 
        self-attention processed representations and back to spatial format.
        
        Processing Pipeline:
        1. Spatial Flattening: Convert 2D feature maps to sequence of spatial tokens
        2. Positional Encoding: Add learnable spatial position information
        3. Self-Attention: Apply multi-layer transformer processing
        4. Spatial Reconstruction: Convert back to 2D feature maps
        
        Args:
            x: Input feature tensor [B, C, H, W]
            
        Returns:
            Output feature tensor [B, C, H, W] with global context integration
            
        Deep Learning Operations:
        - Reshape operation preserves all spatial information as sequence
        - Positional encoding maintains spatial relationships in attention
        - Multi-head attention captures diverse spatial interaction patterns
        - Residual connections ensure gradient flow and feature preservation
        """
        B, C, H, W = x.shape
        
        # Validate input tensor dimensions and channel count
        assert_shape(x, f"B,{self.channels},H,W", "input")
        
        # Convert 2D spatial feature maps to sequence of spatial tokens
        # [B, C, H, W] → [B, H*W, C] where each spatial location becomes a token
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        
        # Add learnable 2D positional encoding to preserve spatial relationships
        # Critical for transformer to understand spatial structure of image
        x_pos = self.pos_encoding(x_flat, height=H, width=W)
        
        # Apply sequential transformer layers with residual connections
        # Each layer refines the global context understanding
        current = x_pos
        for transformer_layer in self.transformer_layers:
            # Self-attention mechanism: each token attends to all other tokens
            # Captures long-range spatial dependencies impossible with local convolutions
            current, _ = transformer_layer(current)
        
        # Apply final layer normalization for output stability
        current = self.final_norm(current)
        
        # Reconstruct 2D spatial feature maps from processed token sequence
        # [B, H*W, C] → [B, C, H, W] maintaining original spatial dimensions
        output = current.transpose(1, 2).view(B, C, H, W)
        
        # Validate output maintains expected dimensions
        assert_shape(output, f"B,{self.channels},H,W", "output")
        
        return output


@register_model("unet_transformer")
class UNetTransformer(nn.Module):
    """
    U-Net + Transformer Hybrid Architecture for Semantic Segmentation
    
    This model represents a significant advancement in medical and natural image segmentation
    by combining the proven hierarchical feature extraction of U-Net with the global context
    modeling capabilities of Vision Transformers.
    
    Architectural Innovation:
    1. CNN Encoder: Hierarchical feature extraction with increasing receptive fields
    2. Transformer Bottleneck: Global context modeling via self-attention
    3. CNN Decoder: Progressive spatial resolution recovery with skip connections
    4. Hybrid Design: Leverages strengths of both CNN and Transformer paradigms
    
    Key Deep Learning Principles:
    - Multi-Scale Feature Learning: Encoder captures features at multiple resolutions
    - Global Context Integration: Transformer processes global spatial relationships
    - Information Preservation: Skip connections maintain fine-grained details
    - Progressive Reconstruction: Decoder gradually recovers spatial resolution
    
    Advantages over Pure CNN U-Net:
    - Global receptive field at bottleneck (vs. limited CNN receptive field)
    - Long-range spatial dependency modeling
    - Attention-based feature refinement
    - Better handling of large objects and global structure
    
    Advantages over Pure Vision Transformer:
    - Computational efficiency (attention only at bottleneck)
    - Preserved spatial inductive biases from CNN components
    - Better fine-grained detail preservation via skip connections
    - Proven U-Net architecture for dense prediction tasks
    
    Tensor Flow Architecture:
    Input [B, C_in, H, W] 
    → Encoder [B, C_enc, H/2^n, W/2^n] 
    → Transformer Bottleneck [B, C_enc*2, H/2^n, W/2^n]
    → Decoder [B, C_dec, H, W] 
    → Output [B, num_classes, H, W]
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize U-Net + Transformer Hybrid Model
        
        This constructor builds the complete hybrid architecture by combining
        CNN-based encoder/decoder with transformer-based bottleneck processing.
        
        Args:
            config: Model configuration containing all architectural parameters
            
        Architecture Construction Process:
        1. Configuration Validation: Ensure architectural consistency
        2. Encoder Building: Create hierarchical feature extraction pathway
        3. Transformer Bottleneck: Build global context processing module
        4. Decoder Building: Create progressive upsampling pathway
        5. Output Layer: Final pixel-wise classification layer
        6. Weight Initialization: Apply appropriate initialization schemes
        
        Deep Learning Design Considerations:
        - Channel progression ensures smooth information flow
        - Skip connection compatibility between encoder/decoder levels
        - Transformer bottleneck channel expansion for richer representations
        - Proper weight initialization for stable training convergence
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
        """
        Build CNN Encoder Path for Hierarchical Feature Extraction
        
        The encoder implements the contracting path of U-Net, progressively
        reducing spatial resolution while increasing feature complexity.
        
        Deep Learning Principles:
        - Hierarchical Feature Learning: Each level captures features at different scales
        - Receptive Field Growth: Deeper layers see larger spatial contexts
        - Feature Abstraction: Lower levels detect edges/textures, higher levels detect objects
        - Information Bottleneck: Spatial compression forces learning of compact representations
        
        Architecture Pattern:
        Level 0: [B, C_in, H, W] → [B, C_enc[0], H/2, W/2]
        Level 1: [B, C_enc[0], H/2, W/2] → [B, C_enc[1], H/4, W/4]
        ...
        Level n: [B, C_enc[n-1], H/2^n, W/2^n] → [B, C_enc[n], H/2^(n+1), W/2^(n+1)]
        
        Returns:
            ModuleList of DownBlock modules for progressive downsampling
        """
        encoder = nn.ModuleList()
        
        # Progressive channel expansion with spatial compression
        in_channels = self.input_channels
        
        for out_channels in self.encoder_channels:
            encoder.append(
                DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    downsample_type="maxpool",  # Max pooling preserves important features
                    norm_type="batch",          # Batch normalization for training stability
                    activation="relu",          # ReLU for non-linearity and sparsity
                    dropout=0.0                 # No dropout in encoder (preserve information)
                )
            )
            in_channels = out_channels
        
        return encoder
    
    def _build_transformer_bottleneck(self) -> nn.Module:
        """
        Build Transformer-Based Bottleneck for Global Context Processing
        
        The bottleneck is the critical innovation that differentiates this hybrid
        architecture from pure CNN U-Net. It replaces traditional convolutional
        bottleneck with self-attention mechanisms for global context modeling.
        
        Design Rationale:
        - Applied at lowest spatial resolution (computational efficiency)
        - Channel expansion provides richer representation space for attention
        - Global receptive field addresses CNN's limited spatial context
        - Self-attention learns task-relevant spatial relationships
        
        Three-Stage Processing:
        1. Input Projection: Expand channels for richer transformer processing
        2. Transformer Processing: Multi-layer self-attention with global context
        3. Output Projection: Maintain architectural consistency for decoder
        
        Mathematical Flow:
        [B, C_enc, H_min, W_min] 
        → Input Proj → [B, 2*C_enc, H_min, W_min]
        → Transformer → [B, 2*C_enc, H_min, W_min] (with global context)
        → Output Proj → [B, 2*C_enc, H_min, W_min]
        
        Returns:
            Sequential module containing the complete bottleneck processing pipeline
        """
        # Input channels from deepest encoder level
        in_channels = self.encoder_channels[-1]
        
        # Channel expansion for richer transformer representation
        # Factor of 2 is common practice balancing capacity and efficiency
        bottleneck_channels = in_channels * 2
        
        # Stage 1: Input projection to expand representational capacity
        input_proj = ConvBlock(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            padding=1,
            norm_type="batch",
            activation="relu",
            dropout=self.dropout
        )
        
        # Stage 2: Transformer bottleneck for global context modeling
        transformer_bottleneck = TransformerBottleneck(
            channels=bottleneck_channels,
            num_layers=self.transformer_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            max_height=64,  # Reasonable maximum for typical bottleneck sizes
            max_width=64
        )
        
        # Stage 3: Output projection to maintain channel consistency
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
        Forward Pass Through Complete U-Net + Transformer Hybrid Architecture
        
        This method implements the full forward propagation through the hybrid model,
        combining hierarchical CNN feature extraction with global transformer processing.
        
        Processing Pipeline:
        1. Encoder Path: Progressive spatial downsampling with feature extraction
        2. Skip Connection Storage: Preserve multi-scale features for decoder
        3. Transformer Bottleneck: Global context processing via self-attention
        4. Decoder Path: Progressive spatial upsampling with skip connection fusion
        5. Output Generation: Final pixel-wise classification
        
        Args:
            x: Input image tensor [B, input_channels, H, W]
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
            
        Deep Learning Flow Analysis:
        - Information Compression: Encoder reduces spatial dimensions, increases channels
        - Global Processing: Transformer captures long-range spatial dependencies
        - Information Recovery: Decoder reconstructs spatial details using skip connections
        - Multi-Scale Fusion: Skip connections preserve fine-grained spatial information
        - Dense Prediction: Every pixel receives a classification score
        
        Key Innovations:
        - Hybrid CNN-Transformer processing for optimal feature extraction
        - Global context integration at computationally efficient bottleneck
        - Preserved spatial detail through U-Net skip connections
        - End-to-end differentiable architecture for segmentation tasks
        """
        # Validate input tensor format and dimensions
        assert_shape(x, f"B,{self.input_channels},H,W", "input")
        
        # Store original spatial dimensions for final output validation
        original_height, original_width = x.shape[2], x.shape[3]
        
        # ENCODER PATH: Hierarchical feature extraction with spatial compression
        skip_connections = []  # Store features for decoder skip connections
        current = x
        
        for encoder_block in self.encoder:
            # Each encoder block: convolution → normalization → activation → downsample
            # Returns: downsampled features + skip connection features
            current, skip = encoder_block(current)
            skip_connections.append(skip)  # Preserve for decoder
        
        # TRANSFORMER BOTTLENECK: Global context processing
        # Apply self-attention at lowest spatial resolution for efficiency
        # Captures long-range spatial dependencies impossible with local convolutions
        current = self.bottleneck(current)
        
        # DECODER PATH: Progressive spatial reconstruction with skip connection fusion
        skip_connections = skip_connections[::-1]  # Reverse for symmetric processing
        
        for i, decoder_block in enumerate(self.decoder):
            # Retrieve corresponding skip connection from encoder
            skip = skip_connections[i]
            # Fuse upsampled features with skip connection via concatenation
            # Combines global context (from transformer) with local details (from encoder)
            current = decoder_block(current, skip)
        
        # FINAL OUTPUT: Pixel-wise classification layer
        # 1x1 convolution maps feature channels to class probabilities
        output = self.output_conv(current)
        
        # Ensure output matches input spatial dimensions (handle any size mismatches)
        if output.shape[2:] != (original_height, original_width):
            output = torch.nn.functional.interpolate(
                output,
                size=(original_height, original_width),
                mode='bilinear',        # Smooth interpolation for segmentation
                align_corners=False     # PyTorch best practice for segmentation
            )
        
        # Validate final output dimensions
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

# It centralizes creation decisions and hides construction details from the caller. ———————— factory function 

"""
Why not just call the constructor directly?

If your project is tiny and you will never change the creation logic, calling User() directly is fine.

A factory becomes valuable when at least one of these is true:

1) You may choose between multiple implementations

Analogy: You ask for “a car,” not “a 2022 Toyota Corolla with these exact options.”

                def make_cache(env: str):
                    if env == "prod":
                        return RedisCache()
                    return InMemoryCache()


Caller code stays stable even if you later switch Redis → Memcached.

2) Construction is non-trivial (more than “new + defaults”)

Maybe you must pass config, inject dependencies, set up internal state, validate arguments, etc.

            def make_http_client(config):
                return HttpClient(
                    base_url=config.base_url,
                    timeout=config.timeout,
                    retries=3,
                )


The caller no longer repeats these knobs everywhere.

3) You want one place to enforce rules/invariants

Example rules:

        “Timeout must be within [1, 30] seconds.”

        “In production, TLS must be enabled.”

        “This object must be created with these dependencies.”

        Factories are a good place to enforce policy.

4) You want easier testing / swapping (without rewriting caller code)

In tests, you can replace the factory result:

                def make_repo():
                    return RealRepo()

# in tests, patch make_repo() to return FakeRepo()


This is often simpler than hunting down 30 places that do RealRepo().

5) You want lazy creation (create only when needed)
            _repo = None
            def get_repo():
                global _repo
                if _repo is None:
                    _repo = RealRepo()
                return _repo


This “factory + cache” is a common Singleton alternative.

So when is it “just a wrapper”?

If your factory is always:

        return SomeClass()

        no branching

        no extra parameters or validation

        no policy

        no dependency wiring

        no future flexibility needed

…then yes, it’s mostly a wrapper, and it may be unnecessary.

But many teams still keep such wrappers deliberately because they expect growth:

Today: return Model()

Tomorrow: load weights, set device, configure batching, attach metrics, etc.

That wrapper becomes the single edit point later.

"""