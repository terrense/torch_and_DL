"""
Sequence loss functions for ASR training.

Implements masked cross-entropy for variable-length sequences,
auxiliary losses for predictor training, and proper handling
of padding and sequence length variations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
import logging

from ..utils.tensor_utils import assert_shape, check_nan_inf

logger = logging.getLogger(__name__)


class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with masking for variable-length sequences.
    
    Handles padding tokens and sequence length variations properly,
    with optional label smoothing for improved training stability.
    """
    
    def __init__(
        self,
        ignore_index: int = 0,  # Usually pad_token_id
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Initialize masked cross-entropy loss.
        
        Args:
            ignore_index: Token ID to ignore (usually padding token)
            label_smoothing: Label smoothing factor [0, 1]
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        if label_smoothing < 0.0 or label_smoothing > 1.0:
            raise ValueError(f"Label smoothing must be in [0, 1], got {label_smoothing}")
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute masked cross-entropy loss.
        
        Args:
            logits: [B, S, vocab_size] model predictions
            targets: [B, S] target token sequences
            lengths: [B] actual sequence lengths (optional)
            
        Returns:
            loss: Scalar loss value
        """
        # Input validation
        assert_shape(logits, "B,S,V", "logits")
        assert_shape(targets, "B,S", "targets")
        check_nan_inf(logits, "logits")
        
        B, S, V = logits.shape
        
        if lengths is not None:
            assert_shape(lengths, "B", "lengths")
            # Create mask from lengths
            mask = torch.arange(S, device=targets.device)[None, :] < lengths[:, None]
        else:
            # Create mask from ignore_index
            mask = (targets != self.ignore_index)
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, V)  # [B*S, V]
        targets_flat = targets.view(-1)   # [B*S]
        mask_flat = mask.view(-1)         # [B*S]
        
        if self.label_smoothing > 0.0:
            # Label smoothing implementation
            log_probs = F.log_softmax(logits_flat, dim=-1)
            
            # Standard cross-entropy component
            nll_loss = F.nll_loss(
                log_probs, 
                targets_flat, 
                ignore_index=self.ignore_index,
                reduction='none'
            )
            
            # Smoothing component (uniform distribution over vocabulary)
            smooth_loss = -log_probs.mean(dim=-1)
            
            # Combine losses
            loss = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
            
            # Apply mask
            if lengths is not None:
                loss = loss * mask_flat.float()
            else:
                # Mask is already applied by ignore_index in nll_loss
                pass
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.ignore_index,
                reduction='none'
            )
            
            # Apply mask if using lengths
            if lengths is not None:
                loss = loss * mask_flat.float()
        
        # Apply reduction
        if self.reduction == 'none':
            return loss.view(B, S)
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            if lengths is not None:
                # Average over valid tokens only
                return loss.sum() / mask_flat.sum().clamp(min=1)
            else:
                return loss.mean()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class PredictorLoss(nn.Module):
    """
    Loss function for predictor training.
    
    Supports different predictor types (boundary detection, duration prediction)
    with appropriate loss functions and proper masking.
    """
    
    def __init__(
        self,
        predictor_type: str = 'boundary',
        loss_type: str = 'auto',
        pos_weight: Optional[float] = None,
        reduction: str = 'mean'
    ):
        """
        Initialize predictor loss.
        
        Args:
            predictor_type: Type of predictor ('boundary', 'duration')
            loss_type: Loss function type ('bce', 'mse', 'auto')
            pos_weight: Positive class weight for BCE (boundary detection)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.predictor_type = predictor_type
        self.reduction = reduction
        
        # Auto-select loss type based on predictor type
        if loss_type == 'auto':
            if predictor_type == 'boundary':
                loss_type = 'bce'
            elif predictor_type == 'duration':
                loss_type = 'mse'
            else:
                raise ValueError(f"Unknown predictor type: {predictor_type}")
        
        self.loss_type = loss_type
        
        # Initialize loss function
        if loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight) if pos_weight else None,
                reduction='none'
            )
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute predictor loss.
        
        Args:
            predictions: [B, T, 1] predictor predictions
            targets: [B, T] or [B, T, 1] target values
            mask: [B, T] padding mask (True for valid positions)
            
        Returns:
            loss: Scalar loss value
        """
        # Input validation
        assert_shape(predictions, "B,T,1", "predictions")
        check_nan_inf(predictions, "predictions")
        
        B, T, _ = predictions.shape
        
        # Handle target shape
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)  # [B, T, 1]
        assert_shape(targets, "B,T,1", "targets")
        
        if mask is not None:
            assert_shape(mask, "B,T", "mask")
        
        # Compute loss
        if self.loss_type == 'bce':
            # Binary cross-entropy for boundary detection
            loss = self.loss_fn(predictions, targets.float())
        elif self.loss_type == 'mse':
            # Mean squared error for duration prediction
            loss = self.loss_fn(predictions, targets.float())
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
            loss = loss * mask_expanded.float()
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            if mask is not None:
                return loss.sum() / mask.sum().clamp(min=1)
            else:
                return loss.mean()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class CombinedASRLoss(nn.Module):
    """
    Combined loss function for ASR training.
    
    Combines decoder cross-entropy loss with optional predictor auxiliary loss,
    with configurable weighting and proper handling of variable-length sequences.
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        label_smoothing: float = 0.0,
        predictor_loss_weight: float = 0.1,
        predictor_type: str = 'boundary',
        predictor_pos_weight: Optional[float] = None
    ):
        """
        Initialize combined ASR loss.
        
        Args:
            vocab_size: Vocabulary size for decoder
            pad_token_id: Padding token ID
            label_smoothing: Label smoothing for decoder loss
            predictor_loss_weight: Weight for predictor loss
            predictor_type: Type of predictor ('boundary', 'duration')
            predictor_pos_weight: Positive class weight for predictor BCE
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.predictor_loss_weight = predictor_loss_weight
        
        # Initialize loss functions
        self.decoder_loss = MaskedCrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        
        self.predictor_loss = PredictorLoss(
            predictor_type=predictor_type,
            pos_weight=predictor_pos_weight,
            reduction='mean'
        )
    
    def forward(
        self,
        decoder_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        predictor_predictions: Optional[torch.Tensor] = None,
        predictor_targets: Optional[torch.Tensor] = None,
        feature_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined ASR loss.
        
        Args:
            decoder_logits: [B, S, vocab_size] decoder predictions
            target_tokens: [B, S] target token sequences
            target_lengths: [B] target sequence lengths
            predictor_predictions: [B, T, 1] predictor predictions (optional)
            predictor_targets: [B, T] predictor targets (optional)
            feature_mask: [B, T] feature padding mask (optional)
            
        Returns:
            Dictionary containing:
            - total_loss: Combined loss
            - decoder_loss: Cross-entropy loss
            - predictor_loss: Predictor loss (if applicable)
            - token_accuracy: Token-level accuracy
        """
        losses = {}
        
        # 1. Decoder loss (always computed)
        decoder_loss_val = self.decoder_loss(
            decoder_logits, 
            target_tokens, 
            target_lengths
        )
        losses['decoder_loss'] = decoder_loss_val
        
        # 2. Predictor loss (optional)
        if predictor_predictions is not None and predictor_targets is not None:
            predictor_loss_val = self.predictor_loss(
                predictor_predictions,
                predictor_targets,
                feature_mask
            )
            losses['predictor_loss'] = predictor_loss_val
        
        # 3. Combine losses
        total_loss = decoder_loss_val
        if 'predictor_loss' in losses:
            total_loss = total_loss + self.predictor_loss_weight * losses['predictor_loss']
        
        losses['total_loss'] = total_loss
        
        # 4. Compute token accuracy
        token_accuracy = compute_token_accuracy(
            decoder_logits,
            target_tokens,
            target_lengths,
            ignore_index=self.decoder_loss.ignore_index
        )
        losses['token_accuracy'] = token_accuracy
        
        return losses


def compute_token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    ignore_index: int = 0
) -> torch.Tensor:
    """
    Compute token-level accuracy for sequence prediction.
    
    Args:
        logits: [B, S, vocab_size] model predictions
        targets: [B, S] target token sequences
        lengths: [B] actual sequence lengths (optional)
        ignore_index: Token ID to ignore in accuracy calculation
        
    Returns:
        accuracy: Scalar accuracy value [0, 1]
    """
    # Input validation
    assert_shape(logits, "B,S,V", "logits")
    assert_shape(targets, "B,S", "targets")
    
    B, S, V = logits.shape
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)  # [B, S]
    
    # Create mask
    if lengths is not None:
        assert_shape(lengths, "B", "lengths")
        mask = torch.arange(S, device=targets.device)[None, :] < lengths[:, None]
    else:
        mask = (targets != ignore_index)
    
    # Compute accuracy
    correct = (predictions == targets) & mask
    total = mask.sum().clamp(min=1)
    
    accuracy = correct.sum().float() / total.float()
    
    return accuracy


def create_sequence_loss(
    vocab_size: int,
    pad_token_id: int = 0,
    label_smoothing: float = 0.0,
    predictor_loss_weight: float = 0.1,
    predictor_type: str = 'boundary'
) -> CombinedASRLoss:
    """
    Create a combined ASR loss function with default settings.
    
    Args:
        vocab_size: Vocabulary size
        pad_token_id: Padding token ID
        label_smoothing: Label smoothing factor
        predictor_loss_weight: Weight for predictor loss
        predictor_type: Type of predictor loss
        
    Returns:
        loss_fn: Configured CombinedASRLoss instance
    """
    return CombinedASRLoss(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        label_smoothing=label_smoothing,
        predictor_loss_weight=predictor_loss_weight,
        predictor_type=predictor_type
    )


if __name__ == "__main__":
    # Test the loss functions
    print("Testing sequence loss functions...")
    
    # Test parameters
    B, S, T, V = 2, 10, 20, 100
    pad_token_id = 0
    
    # Create test data
    logits = torch.randn(B, S, V)
    targets = torch.randint(1, V, (B, S))
    targets[:, -2:] = pad_token_id  # Add padding
    lengths = torch.tensor([S-2, S-1])
    
    predictor_preds = torch.randn(B, T, 1)
    predictor_targets = torch.randint(0, 2, (T,)).float().unsqueeze(0).expand(B, T)
    feature_mask = torch.ones(B, T, dtype=torch.bool)
    feature_mask[0, -3:] = False  # Add padding
    
    # Test MaskedCrossEntropyLoss
    print("\n1. Testing MaskedCrossEntropyLoss:")
    ce_loss = MaskedCrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
    loss_val = ce_loss(logits, targets, lengths)
    print(f"   Loss value: {loss_val.item():.4f}")
    
    # Test PredictorLoss
    print("\n2. Testing PredictorLoss:")
    pred_loss = PredictorLoss(predictor_type='boundary')
    pred_loss_val = pred_loss(predictor_preds, predictor_targets, feature_mask)
    print(f"   Predictor loss: {pred_loss_val.item():.4f}")
    
    # Test CombinedASRLoss
    print("\n3. Testing CombinedASRLoss:")
    combined_loss = CombinedASRLoss(
        vocab_size=V,
        pad_token_id=pad_token_id,
        label_smoothing=0.1,
        predictor_loss_weight=0.1
    )
    
    loss_dict = combined_loss(
        decoder_logits=logits,
        target_tokens=targets,
        target_lengths=lengths,
        predictor_predictions=predictor_preds,
        predictor_targets=predictor_targets,
        feature_mask=feature_mask
    )
    
    print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"   Decoder loss: {loss_dict['decoder_loss'].item():.4f}")
    print(f"   Predictor loss: {loss_dict['predictor_loss'].item():.4f}")
    print(f"   Token accuracy: {loss_dict['token_accuracy'].item():.4f}")
    
    # Test token accuracy
    print("\n4. Testing token accuracy:")
    accuracy = compute_token_accuracy(logits, targets, lengths, ignore_index=pad_token_id)
    print(f"   Accuracy: {accuracy.item():.4f}")
    
    print("\nAll loss function tests passed!")