"""
Transformer Components Implemented from Scratch for Computer Vision

This module provides a complete implementation of transformer components specifically
designed for computer vision tasks, particularly for integration with CNN architectures
in hybrid models like U-Net + Transformer.

Key Deep Learning Concepts Implemented:
1. Multi-Head Self-Attention: Enables modeling of long-range dependencies
2. Scaled Dot-Product Attention: Core attention mechanism with learnable relationships
3. Feed-Forward Networks: Non-linear transformations for feature refinement
4. Residual Connections: Gradient flow and training stability
5. Layer Normalization: Input normalization for stable training
6. 2D Positional Encoding: Spatial awareness for image processing

Architectural Innovations:
- Custom 2D positional encoding for spatial token sequences
- Flexible attention masking for various vision tasks
- Pre-norm vs Post-norm transformer blocks for training stability
- Efficient implementation optimized for computer vision applications

Mathematical Foundation:
- Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Multi-head allows parallel attention computations with different learned projections
- Positional encoding provides spatial inductive bias for 2D image data
- Layer normalization stabilizes training in deep transformer networks

References:
- "Attention Is All You Need" - Vaswani et al. (Original Transformer)
- "An Image is Worth 16x16 Words" - Dosovitskiy et al. (Vision Transformer)
- "On Layer Normalization in the Transformer Architecture" - Xiong et al. (Pre-norm)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..utils.tensor_utils import validate_tensor, assert_shape, check_nan_inf


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism for Vision Transformers
    
    This implementation provides the core attention mechanism that enables transformers
    to model long-range dependencies in spatial data. Multi-head attention allows the
    model to jointly attend to information from different representation subspaces.
    
    Deep Learning Principles:
    1. Scaled Dot-Product Attention: Computes attention weights via query-key similarity
    2. Multi-Head Parallelism: Multiple attention heads capture diverse relationships
    3. Linear Projections: Learnable transformations for queries, keys, and values
    4. Attention Scaling: Prevents softmax saturation in high-dimensional spaces
    5. Dropout Regularization: Prevents overfitting in attention weights
    
    Mathematical Formulation:
    - Attention(Q,K,V) = softmax(QK^T/√d_k)V
    - MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
    - head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Key Advantages:
    - Global receptive field: Each token can attend to all other tokens
    - Parallel computation: Multiple heads process different aspects simultaneously
    - Learnable relationships: Attention weights adapt to task-specific patterns
    - Permutation invariance: Order-agnostic processing (with positional encoding)
    
    Tensor Contract:
    - Input: [B, T, D] where B=batch, T=sequence_length, D=embedding_dim
    - Output: [B, T, D] preserving sequence structure
    - Attention weights: [B, H, T, T] where H=num_heads (optional return)
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
        Initialize Multi-Head Attention Module
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads for even head distribution)
            num_heads: Number of parallel attention heads (typically 8 or 16 for vision tasks)
            dropout: Dropout probability for attention weights (regularization)
            bias: Whether to include bias terms in linear projections
            scale_factor: Custom attention scaling (defaults to 1/√d_k for stable gradients)
            
        Deep Learning Design Rationale:
        - embed_dim divisible by num_heads: Ensures even distribution of dimensions across heads
        - num_heads=8: Balances computational cost with representational capacity
        - dropout=0.1: Standard regularization to prevent attention overfitting
        - scale_factor=1/√d_k: Prevents softmax saturation in high-dimensional spaces
        - Xavier initialization: Maintains gradient variance across layers
        """
        super().__init__()
        
        # Validate architectural constraints
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per attention head
        
        # Attention scaling factor: prevents softmax saturation in high dimensions
        # Mathematical justification: maintains unit variance in attention scores
        self.scale = scale_factor if scale_factor is not None else (self.head_dim ** -0.5)
        
        # Linear projections for Query, Key, Value transformations
        # Each projection learns different aspects of input representations
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Query projection
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Key projection  
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Value projection
        
        # Output projection: combines multi-head outputs into final representation
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout for attention weights regularization
        # Applied to attention probabilities to prevent overfitting to specific patterns
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize Attention Weights Using Xavier Uniform Distribution
        
        Xavier initialization maintains gradient variance across layers, crucial for
        training stability in deep transformer networks. This prevents vanishing/
        exploding gradients that can occur with random initialization.
        
        Mathematical Rationale:
        - Xavier uniform: weights ~ U(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
        - Maintains unit variance of activations across layers
        - Zero bias initialization prevents systematic activation shifts
        """
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
        Forward Pass Through Multi-Head Attention Mechanism
        
        This method implements the complete scaled dot-product attention with multiple
        parallel heads, enabling the model to capture diverse types of relationships
        between spatial locations in the input.
        
        Processing Pipeline:
        1. Linear Projections: Transform input to Query, Key, Value representations
        2. Multi-Head Reshaping: Split embeddings across attention heads
        3. Attention Computation: Scaled dot-product attention per head
        4. Attention Masking: Optional masking for specific attention patterns
        5. Output Projection: Combine multi-head outputs into final representation
        
        Args:
            query: Query tensor [B, T, D] - what we want to find information about
            key: Key tensor [B, S, D] - what we search through (defaults to query for self-attention)
            value: Value tensor [B, S, D] - actual information to retrieve (defaults to key)
            attention_mask: Mask tensor [B, T, S] or [T, S] - controls attention patterns
            return_attention: Whether to return attention weights for visualization/analysis
            
        Returns:
            Tuple of (output, attention_weights)
            - output: [B, T, D] - attended representation
            - attention_weights: [B, H, T, S] - attention patterns (if requested)
            
        Mathematical Operations:
        1. Q, K, V = query·W_Q, key·W_K, value·W_V
        2. Attention_scores = (Q·K^T) / √d_k
        3. Attention_weights = softmax(Attention_scores)
        4. Output = Attention_weights·V
        """
        # Default to self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key
        
        B, T, D = query.shape
        S = key.shape[1]  # Source sequence length (may differ from target)
        
        # Validate input tensor dimensions
        assert_shape(query, f"B,T,{self.embed_dim}", "query")
        assert_shape(key, f"B,S,{self.embed_dim}", "key")
        assert_shape(value, f"B,S,{self.embed_dim}", "value")
        
        # Step 1: Linear projections to Query, Key, Value spaces
        # Each projection learns different aspects of the input representation
        Q = self.q_proj(query)  # [B, T, D] - what to search for
        K = self.k_proj(key)    # [B, S, D] - what to search in
        V = self.v_proj(value)  # [B, S, D] - what information to retrieve
        
        # Step 2: Reshape for multi-head attention processing
        # Split embedding dimension across multiple attention heads
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, head_dim]
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, head_dim]
        
        # Step 3: Compute scaled dot-product attention scores
        # Matrix multiplication: Q·K^T gives similarity between query and key vectors
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T, S]
        
        # Step 4: Apply attention mask if provided (for causal attention, padding, etc.)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Broadcast [T, S] to [B, H, T, S]
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                # Broadcast [B, T, S] to [B, H, T, S]
                attention_mask = attention_mask.unsqueeze(1)
            
            # Apply mask: set masked positions to -inf (becomes 0 after softmax)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Step 5: Convert attention scores to probabilities via softmax
        # Softmax ensures attention weights sum to 1 across source sequence
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, T, S]
        attn_weights = self.dropout(attn_weights)  # Regularization
        
        # Numerical stability check: detect NaN/Inf in attention weights
        check_nan_inf(attn_weights, "attention_weights")
        
        # Step 6: Apply attention weights to values
        # Weighted combination of value vectors based on attention probabilities
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T, head_dim]
        
        # Step 7: Reshape back to original format and combine heads
        # Concatenate all attention heads into single representation
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        
        # Step 8: Final output projection
        # Linear transformation to combine multi-head information
        output = self.out_proj(attn_output)
        
        # Validate output dimensions
        assert_shape(output, f"B,T,{self.embed_dim}", "output")
        
        if return_attention:
            return output, attn_weights
        else:
            return output, None


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network for Transformer Architecture
    
    The feed-forward network provides non-linear transformations and feature refinement
    after the attention mechanism. It processes each position independently, adding
    computational depth and non-linearity to the transformer block.
    
    Deep Learning Principles:
    1. Position-wise Processing: Same transformation applied to each sequence position
    2. Non-linear Activation: Introduces non-linearity between linear transformations
    3. Dimension Expansion: Hidden layer typically 4x larger than input (standard practice)
    4. Residual Connection: Enables gradient flow and feature preservation
    5. Dropout Regularization: Prevents overfitting in feed-forward layers
    
    Mathematical Formulation:
    - FFN(x) = max(0, xW₁ + b₁)W₂ + b₂  (for ReLU activation)
    - FFN(x) = GELU(xW₁ + b₁)W₂ + b₂     (for GELU activation)
    
    Architectural Design:
    - Two linear transformations with non-linear activation between them
    - Hidden dimension typically 4x input dimension (balances capacity vs efficiency)
    - GELU activation preferred over ReLU for smoother gradients
    - Dropout applied after both activation and final linear layer
    
    Tensor Contract:
    - Input: [B, T, D] where B=batch, T=sequence_length, D=embedding_dim
    - Output: [B, T, D] preserving sequence structure and dimensions
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
    2D Sinusoidal Positional Encoding for Spatial Vision Transformers
    
    Positional encoding is crucial for transformers to understand spatial relationships
    in 2D image data. Unlike 1D sequences, images have inherent 2D spatial structure
    that must be preserved when converting to token sequences.
    
    Deep Learning Rationale:
    1. Spatial Awareness: Transformers are permutation-invariant, need position info
    2. 2D Structure: Images have both height and width dimensions requiring encoding
    3. Sinusoidal Patterns: Enable learning of relative positions and spatial patterns
    4. Learnable vs Fixed: This implementation uses fixed sinusoidal (generalizes better)
    5. Temperature Scaling: Controls frequency of positional patterns
    
    Mathematical Foundation:
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))     for x-coordinates
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))   for x-coordinates
    - Similar formulation for y-coordinates
    - Final encoding concatenates x and y positional information
    
    Key Advantages:
    - Translation Invariance: Relative positions preserved under translation
    - Scalability: Works for any image size up to maximum dimensions
    - Interpretability: Sinusoidal patterns create interpretable spatial codes
    - Efficiency: Pre-computed encodings avoid runtime computation overhead
    
    Tensor Contract:
    - Input: [B, H*W, D] (flattened spatial tokens) or [B, C, H, W] (2D feature maps)
    - Output: [B, H*W, D] with positional information added to embeddings
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