"""Segmentation metrics implementation from scratch."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import numpy as np


def calculate_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    per_class: bool = True
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) from scratch.
    
    Args:
        predictions: Predicted class indices [B, H, W] or probabilities [B, C, H, W]
        targets: Ground truth class indices [B, H, W]
        num_classes: Number of classes
        ignore_index: Class index to ignore in calculation
        per_class: If True, return per-class IoU, else return mean IoU
        
    Returns:
        IoU scores [num_classes] if per_class=True, else scalar mean IoU
    """
    # Convert predictions to class indices if needed
    if predictions.dim() == 4:  # Probabilities [B, C, H, W]
        predictions = predictions.argmax(dim=1)  # [B, H, W]
    
    # Initialize IoU storage
    iou_scores = torch.zeros(num_classes, dtype=torch.float32, device=predictions.device)
    
    for class_idx in range(num_classes):
        if ignore_index is not None and class_idx == ignore_index:
            iou_scores[class_idx] = float('nan')
            continue
        
        # Create binary masks for current class
        pred_mask = (predictions == class_idx)
        target_mask = (targets == class_idx)
        
        # Handle ignore_index by excluding those pixels
        if ignore_index is not None:
            valid_mask = (targets != ignore_index)
            pred_mask = pred_mask & valid_mask
            target_mask = target_mask & valid_mask
        
        # Calculate intersection and union
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        # Calculate IoU (handle division by zero)
        if union > 0:
            iou_scores[class_idx] = intersection / union
        else:
            # If no pixels of this class exist, set IoU to NaN
            iou_scores[class_idx] = float('nan')
    
    if per_class:
        return iou_scores
    else:
        # Return mean IoU (excluding NaN values)
        valid_ious = iou_scores[~torch.isnan(iou_scores)]
        return valid_ious.mean() if len(valid_ious) > 0 else torch.tensor(0.0)


def calculate_dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    per_class: bool = True,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Calculate Dice score from scratch.
    
    Args:
        predictions: Predicted class indices [B, H, W] or probabilities [B, C, H, W]
        targets: Ground truth class indices [B, H, W]
        num_classes: Number of classes
        ignore_index: Class index to ignore in calculation
        per_class: If True, return per-class Dice, else return mean Dice
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice scores [num_classes] if per_class=True, else scalar mean Dice
    """
    # Convert predictions to class indices if needed
    if predictions.dim() == 4:  # Probabilities [B, C, H, W]
        predictions = predictions.argmax(dim=1)  # [B, H, W]
    
    # Initialize Dice storage
    dice_scores = torch.zeros(num_classes, dtype=torch.float32, device=predictions.device)
    
    for class_idx in range(num_classes):
        if ignore_index is not None and class_idx == ignore_index:
            dice_scores[class_idx] = float('nan')
            continue
        
        # Create binary masks for current class
        pred_mask = (predictions == class_idx).float()
        target_mask = (targets == class_idx).float()
        
        # Handle ignore_index by excluding those pixels
        if ignore_index is not None:
            valid_mask = (targets != ignore_index).float()
            pred_mask = pred_mask * valid_mask
            target_mask = target_mask * valid_mask
        
        # Calculate Dice coefficient
        intersection = (pred_mask * target_mask).sum()
        pred_sum = pred_mask.sum()
        target_sum = target_mask.sum()
        
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice_scores[class_idx] = dice
    
    if per_class:
        return dice_scores
    else:
        # Return mean Dice (excluding NaN values)
        valid_dice = dice_scores[~torch.isnan(dice_scores)]
        return valid_dice.mean() if len(valid_dice) > 0 else torch.tensor(0.0)


def calculate_pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: Optional[int] = None
) -> torch.Tensor:
    """
    Calculate pixel-wise accuracy.
    
    Args:
        predictions: Predicted class indices [B, H, W] or probabilities [B, C, H, W]
        targets: Ground truth class indices [B, H, W]
        ignore_index: Class index to ignore in calculation
        
    Returns:
        Pixel accuracy as scalar tensor
    """
    # Convert predictions to class indices if needed
    if predictions.dim() == 4:  # Probabilities [B, C, H, W]
        predictions = predictions.argmax(dim=1)  # [B, H, W]
    
    # Create mask for valid pixels
    if ignore_index is not None:
        valid_mask = (targets != ignore_index)
        correct = (predictions == targets) & valid_mask
        total = valid_mask.sum().float()
    else:
        correct = (predictions == targets)
        total = torch.tensor(targets.numel(), dtype=torch.float32, device=targets.device)
    
    if total > 0:
        return correct.sum().float() / total
    else:
        return torch.tensor(0.0, device=targets.device)


class IoUMetric:
    """IoU metric calculator with per-class support."""
    
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        self.total_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metric with new batch."""
        # Convert predictions to class indices if needed
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)
        
        batch_size = predictions.size(0)
        self.total_samples += batch_size
        
        for class_idx in range(self.num_classes):
            if self.ignore_index is not None and class_idx == self.ignore_index:
                continue
            
            # Create binary masks
            pred_mask = (predictions == class_idx)
            target_mask = (targets == class_idx)
            
            # Handle ignore_index
            if self.ignore_index is not None:
                valid_mask = (targets != self.ignore_index)
                pred_mask = pred_mask & valid_mask
                target_mask = t