"""Transformer components implemented from scratch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..utils.tensor_utils import validate_tensor, assert_shape, check_nan_inf


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism implemented from scratch.
    
    Tensor Contract:
    - Input: [B, T, D]
    - Output: [B, T, D]
    - Attention weights: [B, H, T, T] (if return_attention=True)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        scale_factor: Optional[float] = None
    ):
        """
        Initialize MultiHeadAttention.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            scale_factor: Custom scaling factor (defaults to 1/sqrt(head_dim))
        """
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = scale_factor if scale_factor is not None else (self.head_dim ** -0.5)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor [B, T, D]
            key: Key tensor [B, S, D] (defaults to query for self-attention)
            value: Value tensor [B, S, D] (defaults to key)
            attention_mask: Attention mask [B, T, S] or [T, S] (1 for valid, 0 for masked)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
            - output: [B, T, D]
            - attention_weights: [B, H, T, S] if return_attention=True, else None
        """
        # Default to self-attention
        if key is None:
            key = query
        if value is None:
            value = key
        
        B, T, D = query.shape
        S = key.shape[1]
        
        # Validate inputs
        assert_shape(query, f"B,T,{self.embed_dim}", "query")
        assert_shape(key, f"B,S,{self.embed_dim}", "key")
        assert_shape(value, f"B,S,{self.embed_dim}", "value")
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [B, T, D]
        K = self.k_proj(key)    # [B, S, D]
        V = self.v_proj(value)  # [B, S, D]
        
        # Reshape for multi-head attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, head_dim]
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T, S]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Broadcast [T, S] to [B, H, T, S]
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                # Broadcast [B, T, S] to [B, H, T, S]
                attention_mask = attention_mask.unsqueeze(1)
            
            # Apply mask (0 for masked positions)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, T, S]
        attn_weights = self.dropout(attn_weights)
        
        # Check for NaN in attention weights
        check_nan_inf(attn_weights, "attention_weights")
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T, head_dim]
        
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        # Validate output
        assert_shape(output, f"B,T,{self.embed_dim}", "output")
        
        if return_attention:
            return output, attn_weights
        else:
            return output, None


class FeedForward(nn.Module):
    """
    Feed-forward network with residual connections and layer normalization.
    
    Tensor Contract:
    - Input: [B, T, D]
    - Output: [B, T, D]
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize FeedForward network.
        
        Args:
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension (defaults to 4 * embed_dim)
            activation: Activation function ('gelu', 'relu', 'swish')
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 4 * embed_dim
        
        # Linear layers
        self.linear1 = nn.Linear(embed_dim, self.hidden_dim, bias=bias)
        self.linear2 = nn.Linear(self.hidden_dim, embed_dim, bias=bias)
        
        # Activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "swish":
            self.activation = nn.SiLU()  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.constant_(self.linear1.bias, 0.0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor [B, T, D]
            
        Returns:
            Output tensor [B, T, D]
        """
        # Validate input
        assert_shape(x, f"B,T,{self.embed_dim}", "input")
        
        # Apply feed-forward layers
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # Validate output
        assert_shape(x, f"B,T,{self.embed_dim}", "output")
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network.
    
    Tensor Contract:
    - Input: [B, T, D]
    - Output: [B, T, D]
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True
    ):
        """
        Initialize TransformerBlock.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension for feed-forward (defaults to 4 * embed_dim)
            dropout: Dropout probability
            activation: Activation function for feed-forward
            norm_first: Whether to apply layer norm before attention/ffn (Pre-LN) or after (Post-LN)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.norm_first = norm_first
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [B, T, D]
            attention_mask: Attention mask [B, T, T] or [T, T]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
            - output: [B, T, D]
            - attention_weights: [B, H, T, T] if return_attention=True, else None
        """
        # Validate input
        assert_shape(x, f"B,T,{self.embed_dim}", "input")
        
        if self.norm_first:
            # Pre-LN: LayerNorm -> Attention -> Residual
            attn_input = self.norm1(x)
            attn_output, attn_weights = self.self_attn(
                attn_input,
                attention_mask=attention_mask,
                return_attention=return_attention
            )
            x = x + self.dropout(attn_output)
            
            # Pre-LN: LayerNorm -> FFN -> Residual
            ffn_input = self.norm2(x)
            ffn_output = self.ffn(ffn_input)
            x = x + self.dropout(ffn_output)
        else:
            # Post-LN: Attention -> Residual -> LayerNorm
            attn_output, attn_weights = self.self_attn(
                x,
                attention_mask=attention_mask,
                return_attention=return_attention
            )
            x = self.norm1(x + self.dropout(attn_output))
            
            # Post-LN: FFN -> Residual -> LayerNorm
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_output))
        
        # Validate output
        assert_shape(x, f"B,T,{self.embed_dim}", "output")
        
        return x, attn_weights


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for spatial tokens.
    
    Tensor Contract:
    - Input: [B, H*W, D] or [B, C, H, W]
    - Output: [B, H*W, D] (same as input if already flattened)
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_height: int = 256,
        max_width: int = 256,
        temperature: float = 10000.0
    ):
        """
        Initialize 2D positional encoding.
        
        Args:
            embed_dim: Embedding dimension (must be even)
            max_height: Maximum height for precomputed encodings
            max_width: Maximum width for precomputed encodings
            temperature: Temperature parameter for sinusoidal encoding
        """
        super().__init__()
        
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be even for 2D positional encoding")
        
        self.embed_dim = embed_dim
        self.max_height = max_height
        self.max_width = max_width
        self.temperature = temperature
        
        # Precompute positional encodings
        self.register_buffer('pos_encoding', self._create_pos_encoding())
    
    def _create_pos_encoding(self) -> torch.Tensor:
        """Create 2D positional encoding table."""
        # Create coordinate grids
        y_pos = torch.arange(self.max_height, dtype=torch.float32).unsqueeze(1)  # [H, 1]
        x_pos = torch.arange(self.max_width, dtype=torch.float32).unsqueeze(0)   # [1, W]
        
        # Create dimension indices
        dim_t = torch.arange(self.embed_dim // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t / self.embed_dim)
        
        # Compute positional encodings
        pos_x = x_pos.unsqueeze(-1) / dim_t  # [1, W, D//2]
        pos_y = y_pos.unsqueeze(-1) / dim_t  # [H, 1, D//2]
        
        # Apply sin/cos
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)  # [1, W, D//2]
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)  # [H, 1, D//2]
        
        # Broadcast and concatenate
        pos_x = pos_x.expand(self.max_height, -1, -1)  # [H, W, D//2]
        pos_y = pos_y.expand(-1, self.max_width, -1)   # [H, W, D//2]
        
        pos_encoding = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, D]
        pos_encoding = pos_encoding.view(-1, self.embed_dim)  # [H*W, D]
        
        return pos_encoding
    
    def forward(self, x: torch.Tensor, height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:
        """
        Add 2D positional encoding to input.
        
        Args:
            x: Input tensor [B, H*W, D] or [B, C, H, W]
            height: Height dimension (required if input is [B, H*W, D])
            width: Width dimension (required if input is [B, H*W, D])
            
        Returns:
            Output tensor [B, H*W, D] with positional encoding added
        """
        if x.dim() == 4:
            # Input is [B, C, H, W] - reshape to [B, H*W, C]
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
            assert_shape(x, f"B,{H*W},{C}", "reshaped_input")
        else:
            # Input is [B, H*W, D]
            B, HW, D = x.shape
            if height is None or width is None:
                raise ValueError("height and width must be provided for flattened input")
            H, W = height, width
            if H * W != HW:
                raise ValueError(f"height * width ({H * W}) must equal sequence length ({HW})")
        
        # Validate dimensions
        if H > self.max_height or W > self.max_width:
            raise ValueError(
                f"Input dimensions ({H}, {W}) exceed maximum ({self.max_height}, {self.max_width})"
            )
        
        # Get positional encoding for current dimensions
        pos_encoding = self.pos_encoding[:H*W, :].unsqueeze(0)  # [1, H*W, D]
        
        # Add positional encoding
        x = x + pos_encoding
        
        return x