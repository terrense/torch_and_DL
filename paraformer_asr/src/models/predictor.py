"""
Alignment Predictor Module for Paraformer ASR

The predictor estimates token boundaries in feature sequences to help with alignment.
This is a key component of the Paraformer architecture that predicts where tokens
should be placed in the continuous feature sequence.

Key Concepts:
- Predicts alignment information from encoder features
- Outputs probability distributions over token positions
- Used to condition the decoder for better alignment
- Helps bridge the gap between continuous features and discrete tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from ..utils.tensor_utils import assert_shape, check_nan_inf


class AlignmentPredictor(nn.Module):
    """
    Predicts token alignment boundaries in feature sequences.
    
    The predictor takes encoder features and predicts where tokens should be aligned.
    This helps the decoder know which parts of the feature sequence correspond to
    which output tokens.
    
    Tensor Contracts:
    - Input: [B, T, D] encoder features where T=feature_sequence_length
    - Output: [B, T, 1] alignment predictions (probability of token boundary)
    - Padding mask: [B, T] boolean mask for valid positions
    
    Architecture:
    1. Feature processing with optional additional layers
    2. Alignment prediction head that outputs boundary probabilities
    3. Optional CTC-style alignment for training
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_layer_norm: bool = True,
        predictor_type: str = 'boundary'  # 'boundary' or 'duration'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.num_layers = num_layers
        self.predictor_type = predictor_type
        
        # Feature processing layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, self.hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(self.hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            
            layers.append(nn.Dropout(dropout))
            current_dim = self.hidden_dim
        
        self.feature_layers = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Prediction head
        if predictor_type == 'boundary':
            # Predicts probability of token boundary at each position
            self.prediction_head = nn.Linear(current_dim, 1)
        elif predictor_type == 'duration':
            # Predicts duration/count of tokens at each position
            self.prediction_head = nn.Linear(current_dim, 1)
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        encoder_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict alignment information from encoder features.
        
        Args:
            encoder_features: [B, T, D] encoder output features
            padding_mask: [B, T] boolean mask (True for valid positions)
            
        Returns:
            predictions: [B, T, 1] alignment predictions
            probabilities: [B, T, 1] sigmoid probabilities (for boundary type)
        """
        # Input validation
        assert_shape(encoder_features, "B,T,D", "predictor_input")
        check_nan_inf(encoder_features, "predictor_input")
        
        B, T, D = encoder_features.shape
        assert D == self.input_dim, f"Input dim {D} must match expected {self.input_dim}"
        
        if padding_mask is not None:
            assert_shape(padding_mask, "B,T", "predictor_padding_mask")
        
        # Process features through additional layers
        features = self.feature_layers(encoder_features)  # [B, T, hidden_dim]
        
        # Generate predictions
        predictions = self.prediction_head(features)  # [B, T, 1]
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # Set padded positions to very negative values
            mask_expanded = padding_mask.unsqueeze(-1)  # [B, T, 1]
            predictions = predictions.masked_fill(~mask_expanded, -1e9)
        
        # Convert to probabilities based on predictor type
        if self.predictor_type == 'boundary':
            # Sigmoid for boundary probabilities
            probabilities = torch.sigmoid(predictions)
        elif self.predictor_type == 'duration':
            # Softplus for duration (always positive)
            probabilities = F.softplus(predictions)
        
        # Output validation
        assert_shape(predictions, "B,T,1", "predictor_predictions")
        assert_shape(probabilities, "B,T,1", "predictor_probabilities")
        check_nan_inf(predictions, "predictor_predictions")
        check_nan_inf(probabilities, "predictor_probabilities")
        
        return predictions, probabilities
    
    def compute_alignment_loss(
        self,
        predictions: torch.Tensor,
        target_alignments: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        loss_type: str = 'bce'
    ) -> torch.Tensor:
        """
        Compute alignment prediction loss.
        
        Args:
            predictions: [B, T, 1] raw predictions from forward pass
            target_alignments: [B, T] or [B, T, 1] target alignment labels
            padding_mask: [B, T] boolean mask for valid positions
            loss_type: 'bce' for binary cross-entropy, 'mse' for mean squared error
            
        Returns:
            loss: scalar alignment loss
        """
        assert_shape(predictions, "B,T,1", "alignment_predictions")
        
        # Ensure target has correct shape
        if target_alignments.dim() == 2:
            target_alignments = target_alignments.unsqueeze(-1)  # [B, T, 1]
        assert_shape(target_alignments, "B,T,1", "target_alignments")
        
        # Compute loss
        if loss_type == 'bce':
            # Binary cross-entropy for boundary prediction
            loss = F.binary_cross_entropy_with_logits(
                predictions, target_alignments.float(), reduction='none'
            )
        elif loss_type == 'mse':
            # Mean squared error for duration prediction
            probabilities = F.softplus(predictions) if self.predictor_type == 'duration' else torch.sigmoid(predictions)
            loss = F.mse_loss(probabilities, target_alignments.float(), reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        # Apply padding mask
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(-1).float()  # [B, T, 1]
            loss = loss * mask_expanded
            # Average over valid positions
            loss = loss.sum() / mask_expanded.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss
    
    def extract_token_positions(
        self,
        probabilities: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        min_distance: int = 1
    ) -> list:
        """
        Extract predicted token positions from alignment probabilities.
        
        Args:
            probabilities: [B, T, 1] alignment probabilities
            padding_mask: [B, T] boolean mask for valid positions
            threshold: minimum probability for token boundary
            min_distance: minimum distance between token boundaries
            
        Returns:
            positions: List of lists, each containing token positions for one batch item
        """
        assert_shape(probabilities, "B,T,1", "alignment_probabilities")
        
        B, T, _ = probabilities.shape
        batch_positions = []
        
        for b in range(B):
            probs = probabilities[b, :, 0]  # [T]
            
            # Apply padding mask
            if padding_mask is not None:
                valid_length = padding_mask[b].sum().item()
                probs = probs[:valid_length]
            else:
                valid_length = T
            
            # Find peaks above threshold
            positions = []
            last_pos = -min_distance
            
            for t in range(valid_length):
                if probs[t] > threshold and t - last_pos >= min_distance:
                    positions.append(t)
                    last_pos = t
            
            batch_positions.append(positions)
        
        return batch_positions
    
    def get_config(self) -> dict:
        """Get predictor configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'predictor_type': self.predictor_type
        }
    
    def generate_alignment_targets(
        self,
        token_positions: list,
        sequence_length: int,
        method: str = 'boundary'
    ) -> torch.Tensor:
        """
        Generate alignment targets from token positions for training.
        
        This method shows how to create training targets for the predictor
        from known token alignments (e.g., from forced alignment).
        
        Args:
            token_positions: List of token boundary positions for each batch item
            sequence_length: Length of the feature sequence
            method: 'boundary' for boundary marking, 'gaussian' for soft boundaries
            
        Returns:
            targets: [B, T] alignment targets
        """
        B = len(token_positions)
        targets = torch.zeros(B, sequence_length)
        
        for b, positions in enumerate(token_positions):
            if method == 'boundary':
                # Mark exact boundary positions
                for pos in positions:
                    if 0 <= pos < sequence_length:
                        targets[b, pos] = 1.0
            elif method == 'gaussian':
                # Create soft boundaries with Gaussian distribution
                sigma = 2.0  # Standard deviation for Gaussian
                for pos in positions:
                    if 0 <= pos < sequence_length:
                        # Create Gaussian centered at position
                        indices = torch.arange(sequence_length, dtype=torch.float)
                        gaussian = torch.exp(-0.5 * ((indices - pos) / sigma) ** 2)
                        targets[b] = torch.maximum(targets[b], gaussian)
        
        return targets
    
    def visualize_alignment(
        self,
        probabilities: torch.Tensor,
        token_positions: Optional[list] = None,
        feature_lengths: Optional[torch.Tensor] = None,
        batch_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Create visualization data for alignment predictions.
        
        Args:
            probabilities: [B, T, 1] alignment probabilities
            token_positions: Optional ground truth positions
            feature_lengths: [B] sequence lengths
            batch_idx: which batch item to visualize
            
        Returns:
            Dictionary with visualization data
        """
        assert_shape(probabilities, "B,T,1", "alignment_probabilities")
        
        B, T, _ = probabilities.shape
        assert 0 <= batch_idx < B, f"batch_idx {batch_idx} out of range [0, {B})"
        
        # Extract probabilities for selected batch item
        probs = probabilities[batch_idx, :, 0].cpu().numpy()  # [T]
        
        # Get valid length
        if feature_lengths is not None:
            valid_length = feature_lengths[batch_idx].item()
            probs = probs[:valid_length]
        else:
            valid_length = T
        
        # Extract predicted positions
        predicted_positions = self.extract_token_positions(
            probabilities[batch_idx:batch_idx+1],
            padding_mask=None if feature_lengths is None else 
                        (torch.arange(T) < feature_lengths[batch_idx:batch_idx+1, None])
        )[0]
        
        viz_data = {
            'probabilities': probs,
            'predicted_positions': predicted_positions,
            'sequence_length': valid_length,
            'time_axis': list(range(valid_length))
        }
        
        if token_positions is not None and batch_idx < len(token_positions):
            viz_data['ground_truth_positions'] = token_positions[batch_idx]
        
        return viz_data


class CTCAlignmentPredictor(AlignmentPredictor):
    """
    CTC-style alignment predictor that predicts token labels at each position.
    
    This variant predicts actual token IDs at each position, similar to CTC,
    rather than just boundary probabilities. Can be used as an alternative
    or auxiliary predictor.
    
    Tensor Contracts:
    - Input: [B, T, D] encoder features
    - Output: [B, T, vocab_size] token predictions at each position
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_layer_norm: bool = True
    ):
        # Initialize base class with boundary type
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            use_layer_norm=use_layer_norm,
            predictor_type='boundary'
        )
        
        self.vocab_size = vocab_size
        
        # Replace prediction head with CTC head
        current_dim = self.hidden_dim if self.hidden_dim else input_dim
        self.prediction_head = nn.Linear(current_dim, vocab_size)
    
    def forward(
        self,
        encoder_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict token labels at each position (CTC-style).
        
        Args:
            encoder_features: [B, T, D] encoder output features
            padding_mask: [B, T] boolean mask (True for valid positions)
            
        Returns:
            logits: [B, T, vocab_size] token logits at each position
            probabilities: [B, T, vocab_size] softmax probabilities
        """
        # Input validation
        assert_shape(encoder_features, "B,T,D", "ctc_predictor_input")
        check_nan_inf(encoder_features, "ctc_predictor_input")
        
        B, T, D = encoder_features.shape
        
        # Process features
        features = self.feature_layers(encoder_features)  # [B, T, hidden_dim]
        
        # Generate token predictions
        logits = self.prediction_head(features)  # [B, T, vocab_size]
        
        # Apply padding mask
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(-1)  # [B, T, 1]
            logits = logits.masked_fill(~mask_expanded, -1e9)
        
        # Convert to probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Output validation
        assert_shape(logits, f"B,T,{self.vocab_size}", "ctc_logits")
        assert_shape(probabilities, f"B,T,{self.vocab_size}", "ctc_probabilities")
        check_nan_inf(logits, "ctc_logits")
        
        return logits, probabilities


def create_predictor_from_config(config: dict) -> AlignmentPredictor:
    """
    Create predictor from configuration dictionary.
    
    Args:
        config: Dictionary with predictor parameters
        
    Returns:
        predictor: Configured AlignmentPredictor instance
    """
    predictor_type = config.get('type', 'boundary')
    
    if predictor_type == 'ctc':
        return CTCAlignmentPredictor(
            input_dim=config['input_dim'],
            vocab_size=config['vocab_size'],
            hidden_dim=config.get('hidden_dim'),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'relu'),
            use_layer_norm=config.get('use_layer_norm', True)
        )
    else:
        return AlignmentPredictor(
            input_dim=config['input_dim'],
            hidden_dim=config.get('hidden_dim'),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'relu'),
            use_layer_norm=config.get('use_layer_norm', True),
            predictor_type=config.get('predictor_type', 'boundary')
        )