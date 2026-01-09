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
                target_mask = target_mask & valid_mask
            
            # Update intersection and union
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            self.intersection[class_idx] += intersection
            self.union[class_idx] += union
    
    def compute(self, per_class: bool = True) -> torch.Tensor:
        """Compute IoU scores."""
        iou_scores = torch.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            if self.ignore_index is not None and class_idx == self.ignore_index:
                iou_scores[class_idx] = float('nan')
                continue
            
            if self.union[class_idx] > 0:
                iou_scores[class_idx] = self.intersection[class_idx] / self.union[class_idx]
            else:
                iou_scores[class_idx] = float('nan')
        
        if per_class:
            return iou_scores
        else:
            valid_ious = iou_scores[~torch.isnan(iou_scores)]
            return valid_ious.mean() if len(valid_ious) > 0 else torch.tensor(0.0)


class DiceMetric:
    """Dice metric calculator with per-class support."""
    
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.intersection = torch.zeros(self.num_classes)
        self.pred_sum = torch.zeros(self.num_classes)
        self.target_sum = torch.zeros(self.num_classes)
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
            pred_mask = (predictions == class_idx).float()
            target_mask = (targets == class_idx).float()
            
            # Handle ignore_index
            if self.ignore_index is not None:
                valid_mask = (targets != self.ignore_index).float()
                pred_mask = pred_mask * valid_mask
                target_mask = target_mask * valid_mask
            
            # Update sums
            intersection = (pred_mask * target_mask).sum()
            pred_sum = pred_mask.sum()
            target_sum = target_mask.sum()
            
            self.intersection[class_idx] += intersection
            self.pred_sum[class_idx] += pred_sum
            self.target_sum[class_idx] += target_sum
    
    def compute(self, per_class: bool = True) -> torch.Tensor:
        """Compute Dice scores."""
        dice_scores = torch.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            if self.ignore_index is not None and class_idx == self.ignore_index:
                dice_scores[class_idx] = float('nan')
                continue
            
            dice = (2.0 * self.intersection[class_idx] + self.smooth) / (
                self.pred_sum[class_idx] + self.target_sum[class_idx] + self.smooth
            )
            dice_scores[class_idx] = dice
        
        if per_class:
            return dice_scores
        else:
            valid_dice = dice_scores[~torch.isnan(dice_scores)]
            return valid_dice.mean() if len(valid_dice) > 0 else torch.tensor(0.0)


class PixelAccuracyMetric:
    """Pixel accuracy metric calculator."""
    
    def __init__(self, ignore_index: Optional[int] = None):
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.correct_pixels = 0
        self.total_pixels = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metric with new batch."""
        # Convert predictions to class indices if needed
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)
        
        # Create mask for valid pixels
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            correct = (predictions == targets) & valid_mask
            total = valid_mask.sum()
        else:
            correct = (predictions == targets)
            total = targets.numel()
        
        self.correct_pixels += correct.sum().item()
        self.total_pixels += total
    
    def compute(self) -> float:
        """Compute pixel accuracy."""
        if self.total_pixels > 0:
            return self.correct_pixels / self.total_pixels
        else:
            return 0.0


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics aggregator.
    
    Combines IoU, Dice, and pixel accuracy metrics with aggregation and reporting utilities.
    """
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: Optional[int] = None,
        smooth: float = 1e-6
    ):
        """
        Initialize segmentation metrics.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names for reporting
            ignore_index: Class index to ignore in calculations
            smooth: Smoothing factor for Dice calculation
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        
        # Initialize individual metrics
        self.iou_metric = IoUMetric(num_classes, ignore_index)
        self.dice_metric = DiceMetric(num_classes, ignore_index, smooth)
        self.pixel_acc_metric = PixelAccuracyMetric(ignore_index)
    
    def reset(self):
        """Reset all metrics."""
        self.iou_metric.reset()
        self.dice_metric.reset()
        self.pixel_acc_metric.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update all metrics with new batch."""
        self.iou_metric.update(predictions, targets)
        self.dice_metric.update(predictions, targets)
        self.pixel_acc_metric.update(predictions, targets)
    
    def compute(self) -> Dict[str, Union[float, torch.Tensor, Dict[str, float]]]:
        """
        Compute all metrics and return comprehensive results.
        
        Returns:
            Dictionary containing all computed metrics
        """
        # Compute individual metrics
        iou_per_class = self.iou_metric.compute(per_class=True)
        dice_per_class = self.dice_metric.compute(per_class=True)
        pixel_accuracy = self.pixel_acc_metric.compute()
        
        # Calculate mean metrics (excluding NaN values)
        valid_iou = iou_per_class[~torch.isnan(iou_per_class)]
        valid_dice = dice_per_class[~torch.isnan(dice_per_class)]
        
        mean_iou = valid_iou.mean().item() if len(valid_iou) > 0 else 0.0
        mean_dice = valid_dice.mean().item() if len(valid_dice) > 0 else 0.0
        
        # Create per-class dictionaries
        iou_dict = {}
        dice_dict = {}
        
        for i, class_name in enumerate(self.class_names):
            if self.ignore_index is None or i != self.ignore_index:
                iou_val = iou_per_class[i].item()
                dice_val = dice_per_class[i].item()
                
                iou_dict[class_name] = iou_val if not torch.isnan(torch.tensor(iou_val)) else 0.0
                dice_dict[class_name] = dice_val if not torch.isnan(torch.tensor(dice_val)) else 0.0
        
        return {
            'pixel_accuracy': pixel_accuracy,
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'iou_per_class': iou_dict,
            'dice_per_class': dice_dict,
            'iou_scores': iou_per_class,
            'dice_scores': dice_per_class
        }
    
    def get_summary_string(self) -> str:
        """Get formatted summary string of all metrics."""
        results = self.compute()
        
        summary = f"Segmentation Metrics Summary:\n"
        summary += f"  Pixel Accuracy: {results['pixel_accuracy']:.4f}\n"
        summary += f"  Mean IoU: {results['mean_iou']:.4f}\n"
        summary += f"  Mean Dice: {results['mean_dice']:.4f}\n"
        summary += f"\nPer-class IoU:\n"
        
        for class_name, iou in results['iou_per_class'].items():
            summary += f"  {class_name}: {iou:.4f}\n"
        
        summary += f"\nPer-class Dice:\n"
        for class_name, dice in results['dice_per_class'].items():
            summary += f"  {class_name}: {dice:.4f}\n"
        
        return summary