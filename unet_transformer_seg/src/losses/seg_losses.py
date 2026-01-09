"""Segmentation loss function registry and utilities."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .dice_loss import DiceLoss
from .bce_dice_loss import BCEDiceLoss, calculate_class_weights


def get_loss_function(
    loss_type: str,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.
    
    Args:
        loss_type: Type of loss function ('dice', 'bce', 'dice_bce', 'ce')
        num_classes: Number of classes in the segmentation task
        **kwargs: Additional arguments for loss function initialization
        
    Returns:
        Initialized loss function
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    
    elif loss_type == 'bce':
        if num_classes == 2:
            # Binary segmentation
            return nn.BCEWithLogitsLoss(**kwargs)
        else:
            raise ValueError("BCE loss only supports binary segmentation (num_classes=2)")
    
    elif loss_type in ['dice_bce', 'bce_dice']:
        return BCEDiceLoss(**kwargs)
    
    elif loss_type in ['ce', 'crossentropy']:
        return nn.CrossEntropyLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (typically 0.25)
            gamma: Focusing parameter (typically 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Class index to ignore
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss.
        
        Args:
            predictions: Model predictions [B, C, H, W] (logits)
            targets: Ground truth targets [B, H, W] (class indices)
            
        Returns:
            Focal loss value
        """
        # Calculate cross entropy
        ce_loss = nn.functional.cross_entropy(
            predictions, targets, 
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
            reduction='none'
        )
        
        # Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Handle ignore_index
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            focal_loss = focal_loss * mask
            
            if self.reduction == 'mean':
                return focal_loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with configurable precision/recall balance.
    
    When alpha=beta=0.5, it becomes Dice loss.
    When alpha=0, beta=1, it becomes recall-focused.
    When alpha=1, beta=0, it becomes precision-focused.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-6,
        reduction: str = 'mean'
    ):
        """
        Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Tversky Loss."""
        # Convert to probabilities
        if predictions.dim() == 4 and predictions.size(1) > 1:
            predictions = torch.softmax(predictions, dim=1)
        else:
            predictions = torch.sigmoid(predictions)
        
        # Convert targets to one-hot if needed
        if targets.dim() == 3:
            num_classes = predictions.size(1)
            targets = nn.functional.one_hot(targets, num_classes=num_classes)
            targets = targets.permute(0, 3, 1, 2).float()
        
        # Flatten
        predictions_flat = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        # Calculate Tversky components
        true_pos = (predictions_flat * targets_flat).sum(dim=2)
        false_neg = (targets_flat * (1 - predictions_flat)).sum(dim=2)
        false_pos = ((1 - targets_flat) * predictions_flat).sum(dim=2)
        
        # Tversky index
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        # Convert to loss
        tversky_loss = 1 - tversky
        
        # Apply reduction
        tversky_loss = tversky_loss.mean(dim=1)  # Average over classes
        
        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()
        else:
            return tversky_loss


def create_loss_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create loss function from configuration dictionary.
    
    Args:
        config: Configuration dictionary with loss parameters
        
    Returns:
        Initialized loss function
    """
    loss_type = config.get('loss_type', 'dice_bce')
    num_classes = config.get('num_classes', 3)
    
    # Extract loss-specific parameters
    loss_params = {}
    
    if loss_type in ['dice_bce', 'bce_dice']:
        loss_params.update({
            'bce_weight': config.get('bce_weight', 0.5),
            'dice_weight': config.get('dice_weight', 0.5),
            'smooth': config.get('smooth', 1e-6)
        })
    elif loss_type == 'dice':
        loss_params.update({
            'smooth': config.get('smooth', 1e-6)
        })
    elif loss_type == 'focal':
        loss_params.update({
            'alpha': config.get('focal_alpha', 1.0),
            'gamma': config.get('focal_gamma', 2.0)
        })
    elif loss_type == 'tversky':
        loss_params.update({
            'alpha': config.get('tversky_alpha', 0.5),
            'beta': config.get('tversky_beta', 0.5),
            'smooth': config.get('smooth', 1e-6)
        })
    
    # Common parameters
    common_params = {
        'reduction': config.get('reduction', 'mean'),
        'ignore_index': config.get('ignore_index', None)
    }
    loss_params.update(common_params)
    
    return get_loss_function(loss_type, num_classes, **loss_params)