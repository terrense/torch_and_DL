"""
Transformer Components for Paraformer ASR

Implements transformer layers from scratch including:
- Multi-head attention without torch.nn.MultiheadAttention
- Feed-forward networks with residual connections
- Layer normalization and attention masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..utils.tensor_utils import assert_shape, check_nan_inf


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention implementation from scratch.
    
    Tensor Contracts:
    - Input: [B, T, D] where B=batch, T=sequence_length, D=model_dim
    - Output: [B, T, D] same shape as input
    - Key padding mask: [B, T] boolean mask (True for valid positions)
    - Attention mask: [T, T] or [B, T, T] for causal/custom masking
    """
    
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert model_dim % num_heads == 0, f"model_dim {model_dim} must be divisible by num_heads {num_heads}"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: [B, T_q, D] query tensor
            key: [B, T_k, D] key tensor (defaults to query for self-attention)
            value: [B, T_v, D] value tensor (defaults to key)
            key_padding_mask: [B, T_k] boolean mask (True for valid positions)
            attention_mask: [T_q, T_k] or [B, T_q, T_k] attention mask
            need_weights: whether to return attention weights
            
        Returns:
            output: [B, T_q, D] attention output
            weights: [B, num_heads, T_q, T_k] attention weights (if need_weights)
        """
        # Input validation
        assert_shape(query, "B,T,D", "query")
        check_nan_inf(query, "query")
        
        B, T_q, D = query.shape
        
        # Default to self-attention
        if key is None:
            key = query
        if value is None:
            value = key
            
        assert_shape(key, f"B,T_k,{D}", "key")
        assert_shape(value, f"B,T_v,{D}", "value")
        
        T_k = key.shape[1]
        T_v = value.shape[1]
        assert T_k == T_v, f"Key length {T_k} must match value length {T_v}"
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [B, T_q, D]
        K = self.k_proj(key)    # [B, T_k, D]
        V = self.v_proj(value)  # [B, T_v, D]
        
        # Reshape for multi-head attention: [B, T, D] -> [B, num_heads, T, head_dim]
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_v, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: [B, num_heads, T_q, T_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [T_q, T_k] -> [1, 1, T_q, T_k]
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                # [B, T_q, T_k] -> [B, 1, T_q, T_k]
                attention_mask = attention_mask.unsqueeze(1)
            
            # Apply mask (True means keep, False means mask out)
            scores = scores.masked_fill(~attention_mask, float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            assert_shape(key_padding_mask, f"B,{T_k}", "key_padding_mask")
            # [B, T_k] -> [B, 1, 1, T_k]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # True means valid, False means padded - invert for masking
            scores = scores.masked_fill(~key_padding_mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, T_q, T_k]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, T_q, head_dim]
        
        # Reshape back: [B, num_heads, T_q, head_dim] -> [B, T_q, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, D)
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        # Output validation
        assert_shape(output, f"B,{T_q},{D}", "attention_output")
        check_nan_inf(output, "attention_output")
        
        if need_weights:
            # Average attention weights across heads for interpretability
            avg_weights = attn_weights.mean(dim=1)  # [B, T_q, T_k]
            return output, avg_weights
        else:
            return output, None


class FeedForward(nn.Module):
    """
    Feed-forward network with residual connections and layer normalization.
    
    Tensor Contracts:
    - Input: [B, T, D] where B=batch, T=sequence_length, D=model_dim
    - Output: [B, T, D] same shape as input
    """
    
    def __init__(self, model_dim: int, ff_dim: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: [B, T, D] input tensor
            
        Returns:
            output: [B, T, D] same shape as input
        """
        # Input validation
        assert_shape(x, "B,T,D", "ff_input")
        check_nan_inf(x, "ff_input")
        
        # Feed-forward transformation
        hidden = self.linear1(x)  # [B, T, ff_dim]
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)  # [B, T, D]
        output = self.dropout(output)
        
        # Output validation
        assert_shape(output, f"B,T,{self.model_dim}", "ff_output")
        check_nan_inf(output, "ff_output")
        
        return output


class TransformerLayer(nn.Module):
    """
    Complete transformer layer with self-attention and feed-forward.
    
    Tensor Contracts:
    - Input: [B, T, D] where B=batch, T=sequence_length, D=model_dim
    - Output: [B, T, D] same shape as input
    - Padding mask: [B, T] boolean mask (True for valid positions)
    """
    
    def __init__(
        self, 
        model_dim: int, 
        num_heads: int, 
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.model_dim = model_dim
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(model_dim, ff_dim, dropout, activation)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of transformer layer.
        
        Args:
            x: [B, T, D] input tensor
            padding_mask: [B, T] boolean mask (True for valid positions)
            attention_mask: [T, T] or [B, T, T] attention mask
            
        Returns:
            output: [B, T, D] same shape as input
        """
        # Input validation
        assert_shape(x, "B,T,D", "transformer_input")
        check_nan_inf(x, "transformer_input")
        
        # Self-attention with residual connection and layer norm (pre-norm)
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(
            x, x, x, 
            key_padding_mask=padding_mask,
            attention_mask=attention_mask
        )
        x = residual + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm (pre-norm)
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)
        
        # Output validation
        assert_shape(x, f"B,T,{self.model_dim}", "transformer_output")
        check_nan_inf(x, "transformer_output")
        
        return x


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower triangular) attention mask.
    
    Args:
        seq_len: sequence length
        device: torch device
        
    Returns:
        mask: [seq_len, seq_len] boolean mask (True for valid positions)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask from sequence lengths.
    
    Args:
        lengths: [B] tensor of sequence lengths
        max_len: maximum sequence length
        
    Returns:
        mask: [B, max_len] boolean mask (True for valid positions)
    """
    batch_size = lengths.shape[0]
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask