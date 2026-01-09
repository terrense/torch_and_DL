"""
Segmentation Metrics Implementation from Scratch

This module provides comprehensive implementations of standard segmentation metrics
used in computer vision and medical imaging. All metrics are implemented from
first principles to ensure transparency and educational value.

Key Deep Learning Metrics:
1. Intersection over Union (IoU/Jaccard): Measures spatial overlap accuracy
2. Dice Coefficient (F1-Score): Emphasizes true positive predictions
3. Pixel Accuracy: Overall classification correctness
4. Sensitivity (Recall): True positive rate for each class
5. Specificity: True negative rate for each class

Mathematical Foundations:
- IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
- Dice = 2 × |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
- Pixel Accuracy = Correct Pixels / Total Pixels
- Sensitivity = True Positives / (True Positives + False Negatives)
- Specificity = True Negatives / (True Negatives + False Positives)

Implementation Features:
- Multi-class Support: Handles binary and multi-class segmentation
- Ignore Index: Excludes specific classes (background, padding) from evaluation
- Numerical Stability: Smoothing factors prevent division by zero
- Batch Processing: Efficient computation across mini-batches
- Memory Efficiency: Optimized for large-scale evaluation

Clinical and Research Applications:
- Medical Imaging: Organ, tumor, lesion segmentation assessment
- Autonomous Driving: Road scene understanding evaluation
- Satellite Imagery: Land cover classification assessment
- Industrial Inspection: Quality control and defect detection

Advantages of From-Scratch Implementation:
- Educational Transparency: Clear understanding of metric computation
- Customization Flexibility: Easy modification for specific requirements
- Numerical Control: Direct handling of edge cases and stability
- Performance Optimization: Tailored for specific use cases
- Debugging Capability: Full visibility into computation steps

References:
- "Metrics for evaluating 3D medical image segmentation" - Taha & Hanbury
- "A survey on evaluation metrics for image segmentation" - Csurka et al.
- "The PASCAL Visual Object Classes Challenge" - Everingham et al.
"""

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
    Calculate Intersection over Union (IoU) from Scratch
    
    IoU is a fundamental metric for evaluating segmentation quality, measuring
    the overlap between predicted and ground truth regions. It provides a
    normalized measure of spatial accuracy that is robust to class imbalance.
    
    Mathematical Definition:
    IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
    
    Deep Learning Interpretation:
    - Intersection: Pixels correctly predicted as positive (true positives)
    - Union: All pixels predicted or actually positive (TP + FP + FN)
    - Range: [0, 1] where 1 indicates perfect overlap
    - Threshold: Typically 0.5 for binary classification, varies for multi-class
    
    Key Properties:
    - Scale Invariant: Normalized by union, handles different object sizes
    - Class Imbalance Robust: Focuses on positive predictions rather than negatives
    - Symmetric: IoU(A,B) = IoU(B,A) for prediction-target pairs
    - Differentiable: Can be used as loss function (1 - IoU)
    
    Args:
        predictions: Predicted class indices [B, H, W] or probabilities [B, C, H, W]
        targets: Ground truth class indices [B, H, W]
        num_classes: Number of classes in segmentation task
        ignore_index: Class index to exclude from evaluation (e.g., background)
        per_class: Return individual class IoU scores vs. mean IoU
        
    Returns:
        IoU scores [num_classes] if per_class=True, else scalar mean IoU
        
    Implementation Notes:
    - Handles both hard predictions (class indices) and soft predictions (probabilities)
    - Supports ignore_index for excluding irrelevant classes
    - Uses efficient tensor operations for GPU acceleration
    - Handles edge cases (empty predictions/targets) with NaN values
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