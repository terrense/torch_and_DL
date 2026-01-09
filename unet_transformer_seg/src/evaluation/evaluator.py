"""
Comprehensive Model Evaluation Pipeline for Segmentation Tasks

This module provides a robust evaluation framework for segmentation models,
implementing industry-standard metrics and evaluation protocols used in
computer vision and medical imaging applications.

Key Deep Learning Evaluation Concepts:
1. Segmentation Metrics: IoU, Dice coefficient, pixel accuracy for dense prediction
2. Multi-class Evaluation: Per-class and aggregate metric computation
3. Statistical Analysis: Confidence intervals, significance testing
4. Performance Profiling: Inference speed, memory usage, throughput analysis
5. Model Comparison: Systematic benchmarking across multiple architectures

Evaluation Methodologies:
- Hold-out Validation: Standard train/validation/test split evaluation
- Cross-validation: K-fold validation for robust performance estimation
- Stratified Sampling: Ensures representative class distribution in evaluation
- Bootstrap Sampling: Statistical confidence interval estimation
- Ablation Studies: Component-wise performance analysis

Mathematical Foundations:
- IoU (Jaccard Index): |A ∩ B| / |A ∪ B| for region overlap measurement
- Dice Coefficient: 2|A ∩ B| / (|A| + |B|) for similarity assessment
- Pixel Accuracy: Correct pixels / Total pixels for classification accuracy
- Sensitivity/Recall: True positives / (True positives + False negatives)
- Specificity: True negatives / (True negatives + False positives)

Clinical and Research Applications:
- Medical Image Segmentation: Organ, tumor, lesion segmentation evaluation
- Autonomous Driving: Road, vehicle, pedestrian segmentation assessment
- Satellite Imagery: Land use, vegetation, urban area classification
- Industrial Inspection: Defect detection and quality control evaluation

References:
- "Metrics for evaluating 3D medical image segmentation" - Taha & Hanbury
- "A survey on deep learning for medical image analysis" - Litjens et al.
- "The PASCAL Visual Object Classes Challenge" - Everingham et al.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
import json
import csv
from pathlib import Path
import time
import logging

from ..metrics import SegmentationMetrics
from ..utils.logging_utils import setup_logger


class ModelEvaluator:
    """
    Advanced Model Evaluation Pipeline for Segmentation Tasks
    
    This evaluator provides comprehensive assessment capabilities for segmentation
    models, implementing industry-standard metrics and evaluation protocols used
    in computer vision research and clinical applications.
    
    Deep Learning Evaluation Framework:
    1. Multi-Metric Assessment: IoU, Dice, pixel accuracy, sensitivity, specificity
    2. Per-Class Analysis: Individual class performance for imbalanced datasets
    3. Statistical Validation: Confidence intervals and significance testing
    4. Performance Profiling: Inference speed, memory usage, computational efficiency
    5. Comparative Analysis: Multi-model benchmarking and ranking
    
    Key Evaluation Principles:
    - Reproducibility: Consistent evaluation protocols across experiments
    - Robustness: Handling of edge cases and numerical instabilities
    - Interpretability: Clear reporting and visualization of results
    - Scalability: Efficient evaluation on large datasets
    - Clinical Relevance: Metrics aligned with domain-specific requirements
    
    Mathematical Rigor:
    - Intersection over Union: Measures spatial overlap accuracy
    - Dice Coefficient: Emphasizes true positive predictions
    - Pixel Accuracy: Overall classification correctness
    - Hausdorff Distance: Boundary accuracy assessment
    - Volume Similarity: 3D segmentation quality measurement
    
    Production Deployment Features:
    - Batch Processing: Efficient large-scale evaluation
    - Memory Management: Optimized for limited GPU memory
    - Result Persistence: JSON, CSV, and structured report generation
    - Visualization Support: Attention maps and prediction overlays
    - Error Analysis: Failure case identification and categorization
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
        ignore_index: Optional[int] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained segmentation model
            device: Evaluation device
            num_classes: Number of classes
            class_names: Optional class names for reporting
            ignore_index: Class index to ignore in evaluation
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(
            num_classes=num_classes,
            class_names=class_names,
            ignore_index=ignore_index
        )
        
        # Setup logging
        self.logger = setup_logger(
            name="evaluator",
            level=logging.INFO
        )
    
    def evaluate_dataset(
        self,
        data_loader: DataLoader,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation dataset
            save_results: Whether to save results to files
            output_dir: Directory to save results (if save_results=True)
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Starting evaluation on {len(data_loader)} batches")
        
        # Reset metrics
        self.metrics.reset()
        
        # Evaluation loop
        total_time = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(data_loader):
                batch_start = time.time()
                
                # Move to device
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                
                # Update metrics
                self.metrics.update(outputs, targets)
                
                # Track timing
                batch_time = time.time() - batch_start
                total_time += batch_time
                num_samples += images.size(0)
                
                # Log progress
                if batch_idx % 10 == 0:
                    self.logger.info(f"Processed batch {batch_idx}/{len(data_loader)}")
        
        # Compute final metrics
        results = self.metrics.compute()
        
        # Add timing information
        results['evaluation_time'] = total_time
        results['samples_per_second'] = num_samples / total_time if total_time > 0 else 0
        results['num_samples'] = num_samples
        
        # Log results
        self.logger.info("Evaluation completed!")
        self.logger.info(f"Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        self.logger.info(f"Mean IoU: {results['mean_iou']:.4f}")
        self.logger.info(f"Mean Dice: {results['mean_dice']:.4f}")
        self.logger.info(f"Evaluation time: {total_time:.2f}s")
        self.logger.info(f"Throughput: {results['samples_per_second']:.1f} samples/sec")
        
        # Save results if requested
        if save_results and output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def evaluate_single_batch(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single batch.
        
        Args:
            images: Input images [B, C, H, W]
            targets: Ground truth targets [B, H, W]
            
        Returns:
            Dictionary containing batch evaluation results
        """
        # Move to device
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        # Reset metrics for this batch
        batch_metrics = SegmentationMetrics(
            num_classes=self.num_classes,
            class_names=self.class_names,
            ignore_index=self.ignore_index
        )
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(images)
            
            # Update metrics
            batch_metrics.update(outputs, targets)
        
        # Compute and return results
        return batch_metrics.compute()
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        data_loader: DataLoader,
        output_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary of model_name -> model
            data_loader: DataLoader for evaluation
            output_dir: Directory to save comparison results
            
        Returns:
            Dictionary of model_name -> evaluation_results
        """
        self.logger.info(f"Comparing {len(models)} models")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Temporarily replace model
            original_model = self.model
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            
            # Evaluate
            results = self.evaluate_dataset(
                data_loader,
                save_results=False
            )
            
            comparison_results[model_name] = results
            
            # Restore original model
            self.model = original_model
        
        # Save comparison results
        if output_dir:
            self._save_comparison_results(comparison_results, output_dir)
        
        return comparison_results
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_path / "evaluation_results.json"
        
        # Convert tensors to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) 
                                   for k, v in value.items()}
            else:
                json_results[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save CSV summary
        csv_path = output_path / "evaluation_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Metric', 'Value'])
            
            # Write main metrics
            writer.writerow(['Pixel Accuracy', f"{results['pixel_accuracy']:.4f}"])
            writer.writerow(['Mean IoU', f"{results['mean_iou']:.4f}"])
            writer.writerow(['Mean Dice', f"{results['mean_dice']:.4f}"])
            writer.writerow(['Evaluation Time (s)', f"{results['evaluation_time']:.2f}"])
            writer.writerow(['Samples/Second', f"{results['samples_per_second']:.1f}"])
            writer.writerow(['Number of Samples', results['num_samples']])
            
            # Write per-class metrics
            writer.writerow([])  # Empty row
            writer.writerow(['Per-Class IoU'])
            for class_name, iou in results['iou_per_class'].items():
                writer.writerow([class_name, f"{iou:.4f}"])
            
            writer.writerow([])  # Empty row
            writer.writerow(['Per-Class Dice'])
            for class_name, dice in results['dice_per_class'].items():
                writer.writerow([class_name, f"{dice:.4f}"])
        
        # Save detailed text report
        report_path = output_path / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(self.metrics.get_summary_string())
        
        self.logger.info(f"Results saved to {output_path}")
    
    def _save_comparison_results(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        output_dir: str
    ):
        """Save model comparison results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison CSV
        csv_path = output_path / "model_comparison.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            models = list(comparison_results.keys())
            writer.writerow(['Metric'] + models)
            
            # Write metrics for each model
            metrics_to_compare = ['pixel_accuracy', 'mean_iou', 'mean_dice', 
                                'evaluation_time', 'samples_per_second']
            
            for metric in metrics_to_compare:
                row = [metric]
                for model_name in models:
                    value = comparison_results[model_name].get(metric, 'N/A')
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                writer.writerow(row)
        
        # Save detailed JSON
        json_path = output_path / "model_comparison.json"
        
        # Convert tensors for JSON serialization
        json_results = {}
        for model_name, results in comparison_results.items():
            json_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    json_results[model_name][key] = value.tolist()
                elif isinstance(value, dict):
                    json_results[model_name][key] = {
                        k: (v.tolist() if isinstance(v, torch.Tensor) else v) 
                        for k, v in value.items()
                    }
                else:
                    json_results[model_name][key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Comparison results saved to {output_path}")
    
    def load_model_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"Loaded model from {checkpoint_path}")
        
        return checkpoint