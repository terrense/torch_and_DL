"""
Dice Loss Implementation for Segmentation Tasks

This module implements the Dice loss function, a critical component for training
segmentation models. Dice loss addresses class imbalance issues common in medical
imaging and provides better gradient flow for small objects.

Key Deep Learning Concepts:
1. Dice Coefficient: Measures overlap between predicted and ground truth regions
2. Soft Dice: Differentiable version using probabilities instead of hard labels
3. Class Imbalance: Addresses unequal class distribution in segmentation
4. Gradient Flow: Provides meaningful gradients even for small objects
5. Multi-class Support: Handles both binary and multi-class segmentation

Mathematical Foundation:
- Dice Coefficient: DSC = 2|X∩Y| / (|X| + |Y|)
- Soft Dice: Uses predicted probabilities instead of binary predictions
- Dice Loss: Loss = 1 - Dice Coefficient (minimization objective)
- Smoothing: Prevents division by zero in edge cases

Advantages over Cross-Entropy:
- Better handling of class imbalance (especially for small objects)
- Direct optimization of segmentation metric (Dice coefficient)
- More stable gradients for sparse segmentation masks
- Robust to class distribution variations

References:
- "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
- Milletari et al. introduced Dice loss for 3D segmentation
- Widely adopted in medical imaging and semantic segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for Semantic Segmentation with Advanced Features
    
    This implementation provides a robust Dice loss function specifically designed
    for segmentation tasks. It addresses key challenges in medical imaging and
    computer vision applications.
    
    Deep Learning Advantages:
    1. Class Imbalance Handling: Effective for datasets with unequal class distribution
    2. Small Object Sensitivity: Better gradients for small segmentation regions
    3. Direct Metric Optimization: Optimizes the actual evaluation metric (Dice)
    4. Numerical Stability: Smoothing prevents division by zero edge cases
    5. Multi-class Support: Handles both binary and multi-class segmentation
    
    Mathematical Formulation:
    - Dice Coefficient: DSC = 2×|P∩T| / (|P| + |T|)
    - Soft Dice: Uses predicted probabilities P and target probabilities T
    - Dice Loss: L = 1 - DSC (converts similarity to loss)
    - Smoothing: Adds small constant to numerator and denominator
    
    Key Features:
    - Ignore Index: Skip specific classes (e.g., background, padding)
    - Per-class Loss: Return individual class losses for analysis
    - Flexible Reduction: Mean, sum, or no reduction options
    - Robust Implementation: Handles edge cases and numerical instability
    
    Use Cases:
    - Medical image segmentation (organs, tumors, lesions)
    - Natural image segmentation with class imbalance
    - Small object detection and segmentation
    - Multi-class dense prediction tasks
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None,
        per_class: bool = False
    ):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Class index to ignore in loss calculation
            per_class: If True, return per-class losses instead of averaged
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.per_class = per_class
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            predictions: Model predictions [B, C, H, W] (logits or probabilities)
            targets: Ground truth targets [B, H, W] (class indices) or [B, C, H, W] (one-hot)
            
        Returns:
            Dice loss value
        """
        # Convert predictions to probabilities if they are logits
        if predictions.dim() == 4 and predictions.size(1) > 1:
            predictions = F.softmax(predictions, dim=1)
        elif predictions.dim() == 4 and predictions.size(1) == 1:
            predictions = torch.sigmoid(predictions)
        
        # Handle target format conversion
        if targets.dim() == 3:  # Class indices [B, H, W]
            num_classes = predictions.size(1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        else:  # Already one-hot [B, C, H, W]
            targets_one_hot = targets.float()
        
        # Handle ignore_index by masking
        if self.ignore_index is not None:
            if targets.dim() == 3:  # Class indices format
                mask = (targets != self.ignore_index).float()
                mask = mask.unsqueeze(1).expand_as(predictions)
            else:  # One-hot format
                mask = torch.ones_like(predictions)
                if self.ignore_index < predictions.size(1):
                    mask[:, self.ignore_index] = 0
            
            predictions = predictions * mask
            targets_one_hot = targets_one_hot * mask
        
        # Calculate Dice coefficient for each class
        batch_size, num_classes = predictions.shape[:2]
        
        # Flatten spatial dimensions
        predictions_flat = predictions.view(batch_size, num_classes, -1)
        targets_flat = targets_one_hot.view(batch_size, num_classes, -1)
        
        # Calculate intersection and union
        intersection = (predictions_flat * targets_flat).sum(dim=2)  # [B, C]
        predictions_sum = predictions_flat.sum(dim=2)  # [B, C]
        targets_sum = targets_flat.sum(dim=2)  # [B, C]
        
        # Dice coefficient calculation
        dice_coeff = (2.0 * intersection + self.smooth) / (
            predictions_sum + targets_sum + self.smooth
        )
        
        # Convert to loss (1 - dice)
        dice_loss = 1.0 - dice_coeff
        
        # Handle reduction
        if self.per_class:
            if self.reduction == 'mean':
                return dice_loss.mean(dim=0)  # Average over batch, keep classes
            elif self.reduction == 'sum':
                return dice_loss.sum(dim=0)
            else:  # 'none'
                return dice_loss
        else:
            # Average over classes first, then apply reduction
            dice_loss = dice_loss.mean(dim=1)  # [B]
            
            if self.reduction == 'mean':
                return dice_loss.mean()
            elif self.reduction == 'sum':
                return dice_loss.sum()
            else:  # 'none'
                return dice_loss


def soft_dice_coefficient(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate soft Dice coefficient.
    
    Args:
        predictions: Predicted probabilities [B, C, H, W]
        targets: Target one-hot encoded [B, C, H, W]
        smooth: Smoothing factor
        
    Returns:
        Dice coefficient per class [B, C]
    """
    # Flatten spatial dimensions
    predictions_flat = predictions.view(predictions.size(0), predictions.size(1), -1)
    targets_flat = targets.view(targets.size(0), targets.size(1), -1)
    
    # Calculate intersection and sums
    intersection = (predictions_flat * targets_flat).sum(dim=2)
    predictions_sum = predictions_flat.sum(dim=2)
    targets_sum = targets_flat.sum(dim=2)
    
    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (predictions_sum + targets_sum + smooth)
    
    return dice