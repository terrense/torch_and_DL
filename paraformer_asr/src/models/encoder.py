"""
Paraformer Encoder Stack

Implements multi-layer transformer/conformer-style encoder with:
- Bidirectional self-attention with proper masking
- Clear tensor contracts and shape documentation
- Configurable depth and dimensions
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

from .transformer import TransformerLayer, create_padding_mask
from ..utils.tensor_utils import assert_shape, check_nan_inf


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs.
    
    Tensor Contracts:
    - Input: [B, T, D] where B=batch, T=sequence_length, D=model_dim
    - Output: [B, T, D] same shape as input with positional encoding added
    """
    
    def __init__(self, model_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * 
                           (-math.log(10000.0) / model_dim))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, model_dim]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: [B, T, D] input tensor
            
        Returns:
            output: [B, T, D] input with positional encoding added
        """
        assert_shape(x, "B,T,D", "pos_encoding_input")
        
        B, T, D = x.shape
        assert D == self.model_dim, f"Input dim {D} must match model_dim {self.model_dim}"
        
        # Add positional encoding
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


class InputProjection(nn.Module):
    """
    Projects input features to model dimension with optional layer norm.
    
    Tensor Contracts:
    - Input: [B, T, F] where F=input_feature_dim
    - Output: [B, T, D] where D=model_dim
    """
    
    def __init__(self, input_dim: int, model_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        self.projection = nn.Linear(input_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input features to model dimension.
        
        Args:
            x: [B, T, F] input features
            
        Returns:
            output: [B, T, D] projected features
        """
        assert_shape(x, f"B,T,{self.input_dim}", "input_projection")
        check_nan_inf(x, "input_features")
        
        # Project to model dimension
        x = self.projection(x)  # [B, T, model_dim]
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        assert_shape(x, f"B,T,{self.model_dim}", "projected_features")
        check_nan_inf(x, "projected_features")
        
        return x


class ParaformerEncoder(nn.Module):
    """
    Multi-layer transformer encoder for Paraformer ASR.
    
    Tensor Contracts:
    - Input features: [B, T, F] where F=input_feature_dim
    - Input lengths: [B] sequence lengths for each batch item
    - Output: [B, T, D] where D=model_dim, encoded representations
    - Output mask: [B, T] boolean mask (True for valid positions)
    
    Architecture:
    1. Input projection: [B, T, F] -> [B, T, D]
    2. Positional encoding: Add sinusoidal position embeddings
    3. N transformer layers with bidirectional self-attention
    4. Optional final layer normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        max_position: int = 5000,
        layer_norm_eps: float = 1e-5,
        final_layer_norm: bool = True
    ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = InputProjection(input_dim, model_dim, dropout)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(model_dim, max_position, dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])
        
        # Optional final layer normalization
        self.final_layer_norm = None
        if final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(model_dim, eps=layer_norm_eps)
    
    def forward(
        self, 
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Paraformer encoder.
        
        Args:
            features: [B, T, F] input feature sequences
            lengths: [B] sequence lengths (optional, defaults to full length)
            
        Returns:
            encoded: [B, T, D] encoded representations
            padding_mask: [B, T] boolean mask (True for valid positions)
        """
        # Input validation
        assert_shape(features, "B,T,F", "encoder_input_features")
        check_nan_inf(features, "encoder_input_features")
        
        B, T, F = features.shape
        assert F == self.input_dim, f"Feature dim {F} must match input_dim {self.input_dim}"
        
        # Create padding mask from lengths
        if lengths is not None:
            assert_shape(lengths, "B", "sequence_lengths")
            assert torch.all(lengths <= T), f"All lengths must be <= max_len {T}"
            assert torch.all(lengths > 0), "All lengths must be > 0"
            padding_mask = create_padding_mask(lengths, T)
        else:
            # No padding, all positions are valid
            padding_mask = torch.ones(B, T, dtype=torch.bool, device=features.device)
        
        # 1. Input projection: [B, T, F] -> [B, T, D]
        x = self.input_projection(features)
        
        # 2. Add positional encoding
        x = self.pos_encoding(x)
        
        # 3. Pass through transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, padding_mask=padding_mask)
            
            # Intermediate validation for debugging
            check_nan_inf(x, f"encoder_layer_{i}_output")
        
        # 4. Optional final layer normalization
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        
        # Final validation
        assert_shape(x, f"B,T,{self.model_dim}", "encoder_output")
        assert_shape(padding_mask, "B,T", "encoder_padding_mask")
        check_nan_inf(x, "encoder_final_output")
        
        return x, padding_mask
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self.model_dim
    
    def get_config(self) -> dict:
        """Get encoder configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'model_dim': self.model_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.layers[0].feed_forward.ff_dim,
            'dropout': self.layers[0].dropout.p,
            'max_position': self.pos_encoding.pe.shape[1],
            'final_layer_norm': self.final_layer_norm is not None
        }


class ConformerFeedForward(nn.Module):
    """
    Conformer-style feed-forward with Swish activation and different structure.
    Alternative to standard transformer feed-forward for conformer variants.
    
    Tensor Contracts:
    - Input: [B, T, D] 
    - Output: [B, T, D] same shape as input
    """
    
    def __init__(self, model_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        
        # Conformer uses Swish activation and different layer structure
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.swish = lambda x: x * torch.sigmoid(x)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conformer feed-forward forward pass."""
        assert_shape(x, "B,T,D", "conformer_ff_input")
        
        # First linear + Swish + dropout
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        
        # Second linear + dropout
        x = self.linear2(x)
        x = self.dropout2(x)
        
        assert_shape(x, f"B,T,{self.model_dim}", "conformer_ff_output")
        return x


def create_encoder_from_config(config: dict) -> ParaformerEncoder:
    """
    Create encoder from configuration dictionary.
    
    Args:
        config: Dictionary with encoder parameters
        
    Returns:
        encoder: Configured ParaformerEncoder instance
    """
    return ParaformerEncoder(
        input_dim=config['input_dim'],
        model_dim=config.get('model_dim', 512),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        ff_dim=config.get('ff_dim', 2048),
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'relu'),
        max_position=config.get('max_position', 5000),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        final_layer_norm=config.get('final_layer_norm', True)
    )