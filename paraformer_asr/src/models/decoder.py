"""
Paraformer Decoder/Refiner for Token Generation

The decoder takes encoder features and predictor signals to generate token sequences.
It uses attention over encoder outputs and integrates predictor information for
improved alignment and token generation.

Key Components:
- Cross-attention to encoder features
- Integration of predictor alignment signals
- Token sequence generation with proper masking
- Support for both training and inference modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from .transformer import MultiHeadAttention, FeedForward, create_causal_mask, create_padding_mask
from ..utils.tensor_utils import assert_shape, check_nan_inf


class DecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention and cross-attention.
    
    Tensor Contracts:
    - Input: [B, S, D] where S=target_sequence_length
    - Encoder features: [B, T, D] where T=source_sequence_length  
    - Output: [B, S, D] same shape as input
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
        
        # Self-attention for target sequence
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        
        # Cross-attention to encoder features
        self.cross_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(model_dim, ff_dim, dropout, activation)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_features: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            x: [B, S, D] target sequence embeddings
            encoder_features: [B, T, D] encoder output features
            target_mask: [B, S] padding mask for target sequence
            encoder_mask: [B, T] padding mask for encoder sequence
            causal_mask: [S, S] causal mask for self-attention
            
        Returns:
            output: [B, S, D] processed target sequence
        """
        # Input validation
        assert_shape(x, "B,S,D", "decoder_layer_input")
        assert_shape(encoder_features, "B,T,D", "decoder_encoder_features")
        check_nan_inf(x, "decoder_layer_input")
        check_nan_inf(encoder_features, "decoder_encoder_features")
        
        # 1. Self-attention with causal masking
        residual = x
        x = self.norm1(x)
        self_attn_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=target_mask,
            attention_mask=causal_mask
        )
        x = residual + self.dropout(self_attn_out)
        
        # 2. Cross-attention to encoder features
        residual = x
        x = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(
            query=x,
            key=encoder_features,
            value=encoder_features,
            key_padding_mask=encoder_mask
        )
        x = residual + self.dropout(cross_attn_out)
        
        # 3. Feed-forward
        residual = x
        x = self.norm3(x)
        ff_out = self.feed_forward(x)
        x = residual + self.dropout(ff_out)
        
        # Output validation
        assert_shape(x, f"B,S,{self.model_dim}", "decoder_layer_output")
        check_nan_inf(x, "decoder_layer_output")
        
        return x


class PredictorIntegration(nn.Module):
    """
    Integrates predictor signals into decoder processing.
    
    This module takes predictor outputs and combines them with encoder features
    to provide better alignment information to the decoder.
    """
    
    def __init__(
        self,
        model_dim: int,
        predictor_dim: int = 1,
        integration_type: str = 'concat',  # 'concat', 'add', 'gate'
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_dim = model_dim
        self.predictor_dim = predictor_dim
        self.integration_type = integration_type
        
        if integration_type == 'concat':
            # Concatenate predictor signals and project back to model_dim
            self.projection = nn.Linear(model_dim + predictor_dim, model_dim)
        elif integration_type == 'add':
            # Project predictor to model_dim and add
            self.projection = nn.Linear(predictor_dim, model_dim)
        elif integration_type == 'gate':
            # Use predictor as gating signal
            self.gate_projection = nn.Linear(predictor_dim, model_dim)
            self.sigmoid = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown integration_type: {integration_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(
        self,
        encoder_features: torch.Tensor,
        predictor_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate predictor signals with encoder features.
        
        Args:
            encoder_features: [B, T, D] encoder output
            predictor_output: [B, T, P] predictor signals
            
        Returns:
            integrated: [B, T, D] integrated features
        """
        assert_shape(encoder_features, "B,T,D", "integration_encoder_features")
        assert_shape(predictor_output, f"B,T,{self.predictor_dim}", "integration_predictor")
        
        if self.integration_type == 'concat':
            # Concatenate and project
            combined = torch.cat([encoder_features, predictor_output], dim=-1)
            integrated = self.projection(combined)
        elif self.integration_type == 'add':
            # Project predictor and add
            projected_predictor = self.projection(predictor_output)
            integrated = encoder_features + projected_predictor
        elif self.integration_type == 'gate':
            # Use predictor as gate
            gate = self.sigmoid(self.gate_projection(predictor_output))
            integrated = encoder_features * gate
        
        integrated = self.layer_norm(integrated)
        integrated = self.dropout(integrated)
        
        assert_shape(integrated, f"B,T,{self.model_dim}", "integrated_features")
        return integrated


class ParaformerDecoder(nn.Module):
    """
    Paraformer decoder that generates token sequences from encoder features.
    
    Tensor Contracts:
    - Encoder features: [B, T, D] where T=source_sequence_length
    - Target tokens: [B, S] where S=target_sequence_length (training)
    - Predictor output: [B, T, P] alignment predictions
    - Output logits: [B, S, vocab_size] token predictions
    
    Architecture:
    1. Token embedding and positional encoding
    2. Integration of predictor signals with encoder features
    3. Multi-layer decoder with self-attention and cross-attention
    4. Output projection to vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        max_target_length: int = 1000,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
        predictor_integration: str = 'concat',
        predictor_dim: int = 1,
        tie_embeddings: bool = False
    ):
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_target_length = max_target_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        
        # Positional encoding for target sequence
        self.pos_encoding = self._create_positional_encoding(max_target_length, model_dim)
        self.pos_dropout = nn.Dropout(dropout)
        
        # Predictor integration
        self.predictor_integration = PredictorIntegration(
            model_dim=model_dim,
            predictor_dim=predictor_dim,
            integration_type=predictor_integration,
            dropout=dropout
        )
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(model_dim, vocab_size)
        
        # Optionally tie input and output embeddings
        if tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self, max_len: int, model_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * 
                           (-math.log(10000.0) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, model_dim]
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.model_dim ** -0.5)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        encoder_features: torch.Tensor,
        predictor_output: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Paraformer decoder.
        
        Args:
            encoder_features: [B, T, D] encoder output features
            predictor_output: [B, T, P] predictor alignment signals
            target_tokens: [B, S] target token sequence (training mode)
            encoder_mask: [B, T] encoder padding mask
            target_lengths: [B] target sequence lengths
            
        Returns:
            logits: [B, S, vocab_size] output token logits
            target_mask: [B, S] target padding mask (if target_lengths provided)
        """
        # Input validation
        assert_shape(encoder_features, "B,T,D", "decoder_encoder_features")
        assert_shape(predictor_output, "B,T,P", "decoder_predictor_output")
        check_nan_inf(encoder_features, "decoder_encoder_features")
        check_nan_inf(predictor_output, "decoder_predictor_output")
        
        B, T, D = encoder_features.shape
        assert D == self.model_dim, f"Encoder dim {D} must match model_dim {self.model_dim}"
        
        # Integrate predictor signals with encoder features
        integrated_features = self.predictor_integration(encoder_features, predictor_output)
        
        if target_tokens is not None:
            # Training mode: use teacher forcing
            assert_shape(target_tokens, "B,S", "decoder_target_tokens")
            S = target_tokens.shape[1]
            
            # Create target embeddings
            target_embeds = self.token_embedding(target_tokens)  # [B, S, D]
            
            # Add positional encoding
            pos_enc = self.pos_encoding[:, :S, :].to(target_embeds.device)
            target_embeds = target_embeds + pos_enc
            target_embeds = self.pos_dropout(target_embeds)
            
            # Create causal mask for self-attention
            causal_mask = create_causal_mask(S, target_embeds.device)
            
            # Create target padding mask
            target_mask = None
            if target_lengths is not None:
                target_mask = create_padding_mask(target_lengths, S)
            
        else:
            # Inference mode: start with empty sequence or BOS token
            # This is handled by the inference/generation methods
            raise NotImplementedError("Inference mode should use generate() method")
        
        # Pass through decoder layers
        x = target_embeds
        for i, layer in enumerate(self.layers):
            x = layer(
                x=x,
                encoder_features=integrated_features,
                target_mask=target_mask,
                encoder_mask=encoder_mask,
                causal_mask=causal_mask
            )
            
            # Intermediate validation
            check_nan_inf(x, f"decoder_layer_{i}_output")
        
        # Final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # [B, S, vocab_size]
        
        # Output validation
        assert_shape(logits, f"B,S,{self.vocab_size}", "decoder_logits")
        check_nan_inf(logits, "decoder_logits")
        
        return logits, target_mask
    
    def generate(
        self,
        encoder_features: torch.Tensor,
        predictor_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate token sequences using greedy decoding or sampling.
        
        Args:
            encoder_features: [B, T, D] encoder output features
            predictor_output: [B, T, P] predictor alignment signals
            encoder_mask: [B, T] encoder padding mask
            max_length: maximum generation length
            bos_token_id: beginning of sequence token ID
            eos_token_id: end of sequence token ID
            pad_token_id: padding token ID
            temperature: sampling temperature (if do_sample=True)
            do_sample: whether to use sampling instead of greedy
            
        Returns:
            generated_tokens: [B, S] generated token sequences
            generated_lengths: [B] actual sequence lengths (before padding)
        """
        assert_shape(encoder_features, "B,T,D", "generate_encoder_features")
        
        B = encoder_features.shape[0]
        device = encoder_features.device
        
        # Integrate predictor signals
        integrated_features = self.predictor_integration(encoder_features, predictor_output)
        
        # Initialize with BOS tokens
        generated_tokens = torch.full(
            (B, max_length), pad_token_id, dtype=torch.long, device=device
        )
        generated_tokens[:, 0] = bos_token_id
        generated_lengths = torch.ones(B, dtype=torch.long, device=device)
        
        # Track which sequences are finished
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for step in range(1, max_length):
            # Current sequence length
            current_length = step
            
            # Get embeddings for current sequence
            current_tokens = generated_tokens[:, :current_length]
            target_embeds = self.token_embedding(current_tokens)
            
            # Add positional encoding
            pos_enc = self.pos_encoding[:, :current_length, :].to(device)
            target_embeds = target_embeds + pos_enc
            
            # Create causal mask
            causal_mask = create_causal_mask(current_length, device)
            
            # Pass through decoder layers
            x = target_embeds
            for layer in self.layers:
                x = layer(
                    x=x,
                    encoder_features=integrated_features,
                    encoder_mask=encoder_mask,
                    causal_mask=causal_mask
                )
            
            # Final normalization and projection
            x = self.final_norm(x)
            logits = self.output_projection(x)  # [B, current_length, vocab_size]
            
            # Get logits for next token prediction
            next_token_logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # Generate next tokens
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update sequences that are not finished
            generated_tokens[:, step] = torch.where(
                finished, pad_token_id, next_tokens
            )
            
            # Update finished status and lengths
            newly_finished = (next_tokens == eos_token_id) & ~finished
            finished = finished | newly_finished
            generated_lengths = torch.where(
                newly_finished, step + 1, generated_lengths
            )
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return generated_tokens, generated_lengths
    
    def get_config(self) -> dict:
        """Get decoder configuration for serialization."""
        return {
            'vocab_size': self.vocab_size,
            'model_dim': self.model_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.layers[0].feed_forward.ff_dim,
            'max_target_length': self.max_target_length,
            'predictor_integration': self.predictor_integration.integration_type,
            'predictor_dim': self.predictor_integration.predictor_dim
        }


def create_decoder_from_config(config: dict) -> ParaformerDecoder:
    """
    Create decoder from configuration dictionary.
    
    Args:
        config: Dictionary with decoder parameters
        
    Returns:
        decoder: Configured ParaformerDecoder instance
    """
    return ParaformerDecoder(
        vocab_size=config['vocab_size'],
        model_dim=config.get('model_dim', 512),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        ff_dim=config.get('ff_dim', 2048),
        max_target_length=config.get('max_target_length', 1000),
        dropout=config.get('dropout', 0.1),
        activation=config.get('activation', 'relu'),
        predictor_integration=config.get('predictor_integration', 'concat'),
        predictor_dim=config.get('predictor_dim', 1),
        tie_embeddings=config.get('tie_embeddings', False)
    )