"""Visualization utilities for segmentation results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import torch


class SegmentationVisualizer:
    """
    Comprehensive visualization utilities for segmentation results.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
        class_colors: Optional[List[Tuple[int, int, int]]] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            num_classes: Number of classes
            class_names: Optional class names
            class_colors: Optional RGB colors for each class
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
        # Set up colors
        if class_colors:
            self.class_colors = class_colors
        else:
            self.class_colors = self._generate_colors()
        
        # Create colormap
        self.colormap = ListedColormap([np.array(color) / 255.0 for color in self.class_colors])
    
    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class."""
        colors = [
            (0, 0, 0),        # Background - black
            (255, 0, 0),      # Class 1 - red
            (0, 255, 0),      # Class 2 - green
            (0, 0, 255),      # Class 3 - blue
            (255, 255, 0),    # Class 4 - yellow
            (255, 0, 255),    # Class 5 - magenta
            (0, 255, 255),    # Class 6 - cyan
            (128, 0, 0),      # Class 7 - maroon
            (0, 128, 0),      # Class 8 - dark green
            (0, 0, 128),      # Class 9 - navy
        ]
        
        # Generate additional colors if needed
        while len(colors) < self.num_classes:
            colors.append(tuple(np.random.randint(0, 256, 3)))
        
        return colors[:self.num_classes]
    
    def plot_prediction_comparison(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot side-by-side comparison of ground truth and prediction.
        
        Args:
            image: Original image [H, W, 3]
            ground_truth: Ground truth mask [H, W]
            prediction: Predicted mask [H, W]
            save_path: Optional path to save figure
            title: Optional figure title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        im1 = axes[1].imshow(ground_truth, cmap=self.colormap, vmin=0, vmax=self.num_classes-1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        im2 = axes[2].imshow(prediction, cmap=self.colormap, vmin=0, vmax=self.num_classes-1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Difference (errors)
        difference = (ground_truth != prediction).astype(np.uint8)
        axes[3].imshow(image)
        axes[3].imshow(difference, cmap='Reds', alpha=0.5)
        axes[3].set_title('Errors (Red)')
        axes[3].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im1, ax=axes, orientation='horizontal', 
                           fraction=0.05, pad=0.1, aspect=50)
        cbar.set_ticks(range(self.num_classes))
        cbar.set_ticklabels(self.class_names)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_probability_maps(
        self,
        probabilities: np.ndarray,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot class probability maps.
        
        Args:
            probabilities: Class probabilities [C, H, W]
            save_path: Optional path to save figure
            title: Optional figure title
            
        Returns:
            Matplotlib figure
        """
        num_cols = min(4, self.num_classes)
        num_rows = (self.num_classes + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(self.num_classes):
            row = i // num_cols
            col = i % num_cols
            
            im = axes[row, col].imshow(probabilities[i], cmap='hot', vmin=0, vmax=1)
            axes[row, col].set_title(f'{self.class_names[i]} Probability')
            axes[row, col].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(self.num_classes, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            axes[row, col].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_summary(
        self,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot metrics summary with bar charts.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            save_path: Optional path to save figure
            title: Optional figure title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall metrics
        overall_metrics = ['pixel_accuracy', 'mean_iou', 'mean_dice']
        overall_values = [metrics.get(metric, 0) for metric in overall_metrics]
        overall_labels = ['Pixel Accuracy', 'Mean IoU', 'Mean Dice']
        
        bars1 = axes[0, 0].bar(overall_labels, overall_values, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Overall Metrics')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, overall_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Per-class IoU
        if 'iou_per_class' in metrics:
            iou_values = list(metrics['iou_per_class'].values())
            iou_labels = list(metrics['iou_per_class'].keys())
            
            bars2 = axes[0, 1].bar(iou_labels, iou_values, color='lightgreen')
            axes[0, 1].set_title('Per-Class IoU')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars2, iou_values):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Per-class Dice
        if 'dice_per_class' in metrics:
            dice_values = list(metrics['dice_per_class'].values())
            dice_labels = list(metrics['dice_per_class'].keys())
            
            bars3 = axes[1, 0].bar(dice_labels, dice_values, color='lightcoral')
            axes[1, 0].set_title('Per-Class Dice')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars3, dice_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Confusion matrix (if available)
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            im = axes[1, 1].imshow(cm, cmap='Blues')
            axes[1, 1].set_title('Confusion Matrix')
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[1, 1].text(j, i, f'{cm[i, j]:.0f}',
                                   ha='center', va='center')
            
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('True')
            plt.colorbar(im, ax=axes[1, 1])
        else:
            axes[1, 1].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(
        self,
        history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training history curves.
        
        Args:
            history: List of epoch results from training
            save_path: Optional path to save figure
            title: Optional figure title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train']['loss'] for h in history]
        train_iou = [h['train']['mean_iou'] for h in history]
        train_dice = [h['train']['mean_dice'] for h in history]
        learning_rates = [h['learning_rate'] for h in history]
        
        # Check if validation data exists
        has_val = 'val' in history[0] and history[0]['val']
        if has_val:
            val_loss = [h['val']['loss'] for h in history if h['val']]
            val_iou = [h['val']['mean_iou'] for h in history if h['val']]
            val_dice = [h['val']['mean_dice'] for h in history if h['val']]
            val_epochs = [h['epoch'] for h in history if h['val']]
        
        # Loss curves
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss')
        if has_val:
            axes[0, 0].plot(val_epochs, val_loss, 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoU curves
        axes[0, 1].plot(epochs, train_iou, 'b-', label='Train IoU')
        if has_val:
            axes[0, 1].plot(val_epochs, val_iou, 'r-', label='Val IoU')
        axes[0, 1].set_title('Mean IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dice curves
        axes[1, 0].plot(epochs, train_dice, 'b-', label='Train Dice')
        if has_val:
            axes[1, 0].plot(val_epochs, val_dice, 'r-', label='Val Dice')
        axes[1, 0].set_title('Mean Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(epochs, learning_rates, 'g-')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_overlay_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Create overlay of segmentation mask on original image.
        
        Args:
            image: Original image [H, W, 3]
            mask: Segmentation mask [H, W]
            alpha: Transparency factor
            
        Returns:
            Overlay image [H, W, 3]
        """
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_idx in range(self.num_classes):
            mask_indices = (mask == class_idx)
            colored_mask[mask_indices] = self.class_colors[class_idx]
        
        # Create overlay
        overlay = (1 - alpha) * image + alpha * colored_mask
        
        return overlay.astype(np.uint8)
    
    def save_legend(self, save_path: str):
        """
        Save class legend as separate image.
        
        Args:
            save_path: Path to save legend image
        """
        fig, ax = plt.subplots(figsize=(2, self.num_classes * 0.5))
        
        # Create legend patches
        patches_list = []
        for i, (name, color) in enumerate(zip(self.class_names, self.class_colors)):
            patch = patches.Rectangle((0, 0), 1, 1, 
                                    facecolor=np.array(color) / 255.0,
                                    label=name)
            patches_list.append(patch)
        
        # Add legend
        ax.legend(handles=patches_list, loc='center', frameon=False)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)