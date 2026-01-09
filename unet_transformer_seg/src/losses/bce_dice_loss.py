"""Combined Binary Cross Entropy and Dice loss implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .dice_loss import DiceLoss


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice loss with configurable weighting.
    
    This loss combines the benefits of BCE (good for pixel-level accuracy) 
    and Dice (good for handling class imbalance and shape preservation).
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None,
        pos_weight: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize combined BCE + Dice loss.
        
        Args:
            bce_weight: Weight for BCE loss component
            dice_weight: Weight for Dice loss component  
            smooth: Smoothing factor for Dice loss
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Class index to ignore in loss calculation
            pos_weight: Positive class weights for BCE (for class imbalance)
            class_weights: Per-class weights for handling class imbalance
        """
        super().__init__()
        
        # Validate weights
        if abs(bce_weight + dice_weight - 1.0) > 1e-6:
            raise ValueError(f"BCE and Dice weights should sum to 1.0, got {bce_weight + dice_weight}")
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Initialize loss components
        self.dice_loss = DiceLoss(
            smooth=smooth,
            reduction=reduction,
            ignore_index=ignore_index,
            per_class=False
        )
        
        # BCE loss setup
        self.pos_weight = pos_weight
        self.class_weights = class_weights
    
    def _calculate_bce_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate BCE loss component."""
        
        # Handle multi-class case
        if predictions.dim() == 4 and predictions.size(1) > 1:
            # Multi-class: use CrossEntropyLoss
            if targets.dim() == 4:  # One-hot targets
                targets = targets.argmax(dim=1)  # Convert to class indices
            
            # Create CrossEntropyLoss with appropriate parameters
            ce_loss = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=self.ignore_index if self.ignore_index is not None else -100,
                reduction=self.reduction
            )
            return ce_loss(predictions, targets)
        
        else:
            # Binary case: use BCEWithLogitsLoss
            if predictions.dim() == 4 and predictions.size(1) == 1:
                predictions = predictions.squeeze(1)  # Remove channel dimension
            
            if targets.dim() == 4:
                if targets.size(1) == 1:
                    targets = targets.squeeze(1)
                else:
                    # Multi-class one-hot to binary (assume class 1 is positive)
                    targets = targets[:, 1] if targets.size(1) > 1 else targets.squeeze(1)
            
            # Handle ignore_index for binary case
            if self.ignore_index is not None:
                mask = (targets != self.ignore_index).float()
                targets = targets * mask
                predictions = predictions * mask
            
            bce_loss = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight,
                reduction='none'
            )
            loss = bce_loss(predictions, targets.float())
            
            # Apply ignore mask if needed
            if self.ignore_index is not None:
                loss = loss * mask
                if self.reduction == 'mean':
                    return loss.sum() / mask.sum().clamp(min=1)
                elif self.reduction == 'sum':
                    return loss.sum()
                else:
                    return loss
            
            # Apply reduction
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined BCE + Dice loss.
        
        Args:
            predictions: Model predictions [B, C, H, W] (logits)
            targets: Ground truth targets [B, H, W] (class indices) or [B, C, H, W] (one-hot)
            
        Returns:
            Combined loss value
        """
        # Calculate individual loss components
        bce_loss = self._calculate_bce_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        
        # Combine with weights
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return combined_loss
    
    def get_component_losses(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> dict:
        """
        Get individual loss components for logging/monitoring.
        
        Returns:
            Dictionary with 'bce', 'dice', and 'combined' losses
        """
        bce_loss = self._calculate_bce_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return {
            'bce': bce_loss.item() if torch.is_tensor(bce_loss) else bce_loss,
            'dice': dice_loss.item() if torch.is_tensor(dice_loss) else dice_loss,
            'combined': combined_loss.item() if torch.is_tensor(combined_loss) else combined_loss
        }


def calculate_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    method: str = 'inverse_freq'
) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        targets: Target tensor [B, H, W] with class indices
        num_classes: Number of classes
        method: Weighting method ('inverse_freq', 'balanced')
        
    Returns:
        Class weights tensor [num_classes]
    """
    # Count class frequencies
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        class_counts[i] = (targets == i).sum().float()
    
    # Avoid division by zero
    class_counts = class_counts.clamp(min=1)
    
    if method == 'inverse_freq':
        # Inverse frequency weighting
        total_samples = targets.numel()
        weights = total_samples / (num_classes * class_counts)
    elif method == 'balanced':
        # Balanced weighting (sklearn style)
        total_samples = targets.numel()
        weights = total_samples / (num_classes * class_counts)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights