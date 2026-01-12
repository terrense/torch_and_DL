"""
Complete Paraformer ASR Model

This module implements the complete Paraformer ASR model that integrates:
- Encoder for feature processing
- Predictor for alignment estimation  
- Decoder for token generation

The model demonstrates proper conditioning of decoder processing using predictor outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
import logging

from .encoder import ParaformerEncoder, create_encoder_from_config
from .predictor import AlignmentPredictor, create_predictor_from_config
from .decoder import ParaformerDecoder, create_decoder_from_config
from ..utils.tensor_utils import assert_shape, check_nan_inf

logger = logging.getLogger(__name__)


class ParaformerASR(nn.Module):
    """
    Complete Paraformer ASR model integrating encoder, predictor, and decoder.
    
    The model demonstrates how the predictor conditions decoder processing:
    1. Encoder processes audio features into contextual representations
    2. Predictor estimates token alignment boundaries in the feature sequence
    3. Decoder uses predictor output to condition token generation
    
    Tensor Contracts:
    - Audio features: [B, T, F] where F=feature_dim (e.g., mel-spectrogram)
    - Feature lengths: [B] sequence lengths for padding mask
    - Target tokens: [B, S] target token sequence (training)
    - Target lengths: [B] target sequence lengths
    - Output logits: [B, S, vocab_size] token predictions
    - Predictor output: [B, T, 1] alignment predictions
    """
    
    def __init__(
        self,
        # Required parameters
        input_dim: int,
        vocab_size: int,
        
        # Encoder configuration
        encoder_dim: int = 512,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        encoder_ff_dim: int = 2048,
        
        # Predictor configuration
        predictor_layers: int = 2,
        predictor_hidden_dim: Optional[int] = None,
        predictor_type: str = 'boundary',
        
        # Decoder configuration
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        max_target_length: int = 1000,
        
        # Shared configuration
        dropout: float = 0.1,
        activation: str = 'relu',
        
        # Integration configuration
        predictor_integration: str = 'concat',
        predictor_loss_weight: float = 0.1,
        
        # Special tokens
        pad_token_id: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3
    ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.predictor_type = predictor_type
        self.predictor_loss_weight = predictor_loss_weight
        
        # Special tokens
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        
        # Initialize components
        self.encoder = ParaformerEncoder(
            input_dim=input_dim,
            model_dim=encoder_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            ff_dim=encoder_ff_dim,
            dropout=dropout,
            activation=activation
        )
        
        self.predictor = AlignmentPredictor(
            input_dim=encoder_dim,
            hidden_dim=predictor_hidden_dim or encoder_dim // 2,
            num_layers=predictor_layers,
            dropout=dropout,
            activation=activation,
            predictor_type=predictor_type
        )
        
        self.decoder = ParaformerDecoder(
            vocab_size=vocab_size,
            model_dim=encoder_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ff_dim=decoder_ff_dim,
            max_target_length=max_target_length,
            dropout=dropout,
            activation=activation,
            predictor_integration=predictor_integration,
            predictor_dim=1  # Predictor outputs [B, T, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Components handle their own initialization
        pass
    
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        target_alignments: Optional[torch.Tensor] = None,
        return_predictor_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of complete Paraformer model.
        
        Args:
            features: [B, T, F] input audio features
            feature_lengths: [B] sequence lengths for features
            target_tokens: [B, S] target token sequence (training mode)
            target_lengths: [B] target sequence lengths
            target_alignments: [B, T] or [B, T, 1] alignment targets for predictor
            return_predictor_output: whether to return predictor outputs
            
        Returns:
            Dictionary containing:
            - logits: [B, S, vocab_size] decoder output logits
            - predictor_predictions: [B, T, 1] predictor raw predictions
            - predictor_probabilities: [B, T, 1] predictor probabilities
            - encoder_features: [B, T, D] encoder output (if requested)
            - padding_mask: [B, T] encoder padding mask
        """
        # Input validation
        assert_shape(features, "B,T,F", "input_features")
        check_nan_inf(features, "input_features")
        
        B, T, F = features.shape
        assert F == self.input_dim, f"Feature dim {F} must match input_dim {self.input_dim}"
        
        if feature_lengths is not None:
            assert_shape(feature_lengths, "B", "feature_lengths")
        
        if target_tokens is not None:
            assert_shape(target_tokens, "B,S", "target_tokens")
        
        # 1. Encode audio features
        encoder_features, padding_mask = self.encoder(features, feature_lengths)
        
        # Validate encoder output
        assert_shape(encoder_features, f"B,T,{self.encoder_dim}", "encoder_features")
        assert_shape(padding_mask, "B,T", "padding_mask")
        check_nan_inf(encoder_features, "encoder_features")
        
        # 2. Predict alignment boundaries
        predictor_predictions, predictor_probabilities = self.predictor(
            encoder_features, padding_mask
        )
        
        # Validate predictor output
        assert_shape(predictor_predictions, "B,T,1", "predictor_predictions")
        assert_shape(predictor_probabilities, "B,T,1", "predictor_probabilities")
        check_nan_inf(predictor_predictions, "predictor_predictions")
        check_nan_inf(predictor_probabilities, "predictor_probabilities")
        
        # 3. Generate tokens using decoder with predictor conditioning
        if target_tokens is not None:
            # Training mode: use teacher forcing
            decoder_logits, target_mask = self.decoder(
                encoder_features=encoder_features,
                predictor_output=predictor_probabilities,  # Use probabilities for conditioning
                target_tokens=target_tokens,
                encoder_mask=padding_mask,
                target_lengths=target_lengths
            )
            
            # Validate decoder output
            S = target_tokens.shape[1]
            assert_shape(decoder_logits, f"B,S,{self.vocab_size}", "decoder_logits")
            check_nan_inf(decoder_logits, "decoder_logits")
        else:
            # Inference mode: no target tokens provided
            decoder_logits = None
            target_mask = None
        
        # Prepare output dictionary
        outputs = {
            'predictor_predictions': predictor_predictions,
            'predictor_probabilities': predictor_probabilities,
            'padding_mask': padding_mask
        }
        
        if decoder_logits is not None:
            outputs['logits'] = decoder_logits
            if target_mask is not None:
                outputs['target_mask'] = target_mask
        
        if return_predictor_output:
            outputs['encoder_features'] = encoder_features
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_tokens: torch.Tensor,
        target_alignments: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for the model.
        
        Args:
            outputs: Model outputs from forward pass
            target_tokens: [B, S] target token sequence
            target_alignments: [B, T] alignment targets for predictor
            target_lengths: [B] target sequence lengths
            label_smoothing: label smoothing factor for cross-entropy
            
        Returns:
            Dictionary containing:
            - total_loss: combined loss
            - decoder_loss: cross-entropy loss for token prediction
            - predictor_loss: alignment prediction loss (if targets provided)
        """
        losses = {}
        
        # 1. Decoder loss (cross-entropy for token prediction)
        if 'logits' in outputs:
            logits = outputs['logits']  # [B, S, vocab_size]
            
            # Prepare targets
            if target_lengths is not None:
                # Create target mask for variable-length sequences
                S = target_tokens.shape[1]
                target_mask = torch.arange(S, device=target_tokens.device)[None, :] < target_lengths[:, None]
            else:
                target_mask = None
            
            # Compute cross-entropy loss
            if label_smoothing > 0:
                # Label smoothing
                log_probs = F.log_softmax(logits, dim=-1)
                nll_loss = F.nll_loss(
                    log_probs.view(-1, self.vocab_size),
                    target_tokens.view(-1),
                    ignore_index=self.pad_token_id,
                    reduction='none'
                )
                smooth_loss = -log_probs.mean(dim=-1).view(-1)
                
                loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss
                
                if target_mask is not None:
                    mask_flat = target_mask.view(-1)
                    loss = loss * mask_flat.float()
                    decoder_loss = loss.sum() / mask_flat.sum().clamp(min=1)
                else:
                    decoder_loss = loss.mean()
            else:
                # Standard cross-entropy
                decoder_loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    target_tokens.view(-1),
                    ignore_index=self.pad_token_id,
                    reduction='mean'
                )
            
            losses['decoder_loss'] = decoder_loss
        
        # 2. Predictor loss (alignment prediction)
        if target_alignments is not None and 'predictor_predictions' in outputs:
            predictor_predictions = outputs['predictor_predictions']
            padding_mask = outputs.get('padding_mask')
            
            predictor_loss = self.predictor.compute_alignment_loss(
                predictions=predictor_predictions,
                target_alignments=target_alignments,
                padding_mask=padding_mask,
                loss_type='bce' if self.predictor_type == 'boundary' else 'mse'
            )
            
            losses['predictor_loss'] = predictor_loss
        
        # 3. Combine losses
        total_loss = torch.tensor(0.0, device=target_tokens.device)
        
        if 'decoder_loss' in losses:
            total_loss = total_loss + losses['decoder_loss']
        
        if 'predictor_loss' in losses:
            total_loss = total_loss + self.predictor_loss_weight * losses['predictor_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def generate(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False,
        return_alignment: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate token sequences from audio features.
        
        Args:
            features: [B, T, F] input audio features
            feature_lengths: [B] sequence lengths for features
            max_length: maximum generation length
            temperature: sampling temperature
            do_sample: whether to use sampling instead of greedy
            return_alignment: whether to return predictor alignment
            
        Returns:
            generated_tokens: [B, S] generated token sequences
            alignment (optional): [B, T, 1] predictor alignment probabilities
        """
        self.eval()
        
        with torch.no_grad():
            # 1. Encode features
            encoder_features, padding_mask = self.encoder(features, feature_lengths)
            
            # 2. Predict alignment
            predictor_predictions, predictor_probabilities = self.predictor(
                encoder_features, padding_mask
            )
            
            # 3. Generate tokens using decoder
            generated_tokens, generated_lengths = self.decoder.generate(
                encoder_features=encoder_features,
                predictor_output=predictor_probabilities,
                encoder_mask=padding_mask,
                max_length=max_length,
                bos_token_id=self.sos_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                temperature=temperature,
                do_sample=do_sample
            )
        
        if return_alignment:
            return generated_tokens, predictor_probabilities
        else:
            return generated_tokens
    
    def extract_alignment(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        min_distance: int = 1
    ) -> Tuple[torch.Tensor, list]:
        """
        Extract token alignment positions from audio features.
        
        Args:
            features: [B, T, F] input audio features
            feature_lengths: [B] sequence lengths for features
            threshold: minimum probability for token boundary
            min_distance: minimum distance between boundaries
            
        Returns:
            probabilities: [B, T, 1] alignment probabilities
            positions: List of lists containing token positions for each batch item
        """
        self.eval()
        
        with torch.no_grad():
            # Encode features
            encoder_features, padding_mask = self.encoder(features, feature_lengths)
            
            # Predict alignment
            _, predictor_probabilities = self.predictor(encoder_features, padding_mask)
            
            # Extract positions
            positions = self.predictor.extract_token_positions(
                probabilities=predictor_probabilities,
                padding_mask=padding_mask,
                threshold=threshold,
                min_distance=min_distance
            )
        
        return predictor_probabilities, positions
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'encoder_dim': self.encoder_dim,
            'vocab_size': self.vocab_size,
            'predictor_type': self.predictor_type,
            'predictor_loss_weight': self.predictor_loss_weight,
            'pad_token_id': self.pad_token_id,
            'sos_token_id': self.sos_token_id,
            'eos_token_id': self.eos_token_id,
            'unk_token_id': self.unk_token_id,
            'encoder_config': self.encoder.get_config(),
            'predictor_config': self.predictor.get_config(),
            'decoder_config': self.decoder.get_config()
        }


def create_paraformer_from_config(config: Dict[str, Any]) -> ParaformerASR:
    """
    Create Paraformer model from configuration dictionary.
    
    Args:
        config: Dictionary with model parameters
        
    Returns:
        model: Configured ParaformerASR instance
    """
    # Extract main model parameters
    model_params = {
        'input_dim': config['input_dim'],
        'vocab_size': config['vocab_size'],
        'encoder_dim': config.get('encoder_dim', config.get('hidden_dim', 512)),
        'encoder_layers': config.get('encoder_layers', 6),
        'encoder_heads': config.get('encoder_heads', 8),
        'encoder_ff_dim': config.get('encoder_ff_dim', 2048),
        'predictor_layers': config.get('predictor_layers', 2),
        'predictor_hidden_dim': config.get('predictor_hidden_dim'),
        'predictor_type': config.get('predictor_type', 'boundary'),
        'decoder_layers': config.get('decoder_layers', 6),
        'decoder_heads': config.get('decoder_heads', 8),
        'decoder_ff_dim': config.get('decoder_ff_dim', 2048),
        'max_target_length': config.get('max_target_length', 1000),
        'dropout': config.get('dropout', 0.1),
        'activation': config.get('activation', 'relu'),
        'predictor_integration': config.get('predictor_integration', 'concat'),
        'predictor_loss_weight': config.get('predictor_loss_weight', 0.1),
        'pad_token_id': config.get('pad_token_id', 0),
        'sos_token_id': config.get('sos_token_id', 1),
        'eos_token_id': config.get('eos_token_id', 2),
        'unk_token_id': config.get('unk_token_id', 3)
    }
    
    return ParaformerASR(**model_params)