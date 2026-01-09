"""
Data utilities for segmentation tasks.

Includes collation functions, data loading helpers, and batch processing utilities.
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np


def collate_variable_size(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for variable-sized inputs with padding masks.
    
    Pads all samples to the maximum size in the batch and returns padding masks
    to indicate valid regions.
    
    Args:
        batch: List of (image, mask) tuples with potentially different sizes
        
    Returns:
        Tuple of (padded_images, padded_masks, padding_masks) where:
            - padded_images: [B, C, H_max, W_max] padded images
            - padded_masks: [B, H_max, W_max] padded segmentation masks  
            - padding_masks: [B, H_max, W_max] boolean masks (True = valid pixel)
    """
    images, masks = zip(*batch)
    
    # Find maximum dimensions in the batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    batch_size = len(images)
    num_channels = images[0].shape[0]
    
    # Initialize padded tensors
    padded_images = torch.zeros(batch_size, num_channels, max_h, max_w, dtype=images[0].dtype)
    padded_masks = torch.zeros(batch_size, max_h, max_w, dtype=masks[0].dtype)
    padding_masks = torch.zeros(batch_size, max_h, max_w, dtype=torch.bool)
    
    # Fill padded tensors
    for i, (image, mask) in enumerate(zip(images, masks)):
        c, h, w = image.shape
        
        # Copy original data
        padded_images[i, :, :h, :w] = image
        padded_masks[i, :h, :w] = mask
        padding_masks[i, :h, :w] = True  # Mark valid regions
    
    return padded_images, padded_masks, padding_masks


def collate_fixed_size(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard collate function for fixed-size inputs.
    
    Args:
        batch: List of (image, mask) tuples with same sizes
        
    Returns:
        Tuple of (batched_images, batched_masks)
    """
    images, masks = zip(*batch)
    
    # Stack into batches
    batched_images = torch.stack(images, dim=0)
    batched_masks = torch.stack(masks, dim=0)
    
    return batched_images, batched_masks


def create_dataloader(dataset,
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True,
                     variable_size: bool = False,
                     **kwargs) -> DataLoader:
    """
    Create a DataLoader with appropriate collation function.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        variable_size: Whether to use variable-size collation
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Configured DataLoader
    """
    if variable_size:
        collate_fn = collate_variable_size
    else:
        collate_fn = collate_fixed_size
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        **kwargs
    )


def compute_dataset_stats(dataset, 
                         num_samples: Optional[int] = None,
                         batch_size: int = 32) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.
    
    Args:
        dataset: Dataset to analyze
        num_samples: Number of samples to use (None = all)
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with dataset statistics
    """
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    # Create dataloader
    loader = create_dataloader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # Initialize accumulators
    pixel_sum = torch.zeros(3)  # RGB channels
    pixel_sum_sq = torch.zeros(3)
    pixel_count = 0
    
    class_counts = {}
    image_sizes = []
    
    samples_processed = 0
    
    print(f"Computing statistics for {num_samples} samples...")
    
    for batch_idx, (images, masks) in enumerate(loader):
        if samples_processed >= num_samples:
            break
            
        batch_size_actual = images.shape[0]
        samples_processed += batch_size_actual
        
        # Image statistics
        # Reshape to [B*H*W, C] for easier computation
        images_flat = images.permute(0, 2, 3, 1).reshape(-1, 3)
        
        pixel_sum += images_flat.sum(dim=0)
        pixel_sum_sq += (images_flat ** 2).sum(dim=0)
        pixel_count += images_flat.shape[0]
        
        # Mask statistics
        for mask in masks:
            unique_classes, counts = torch.unique(mask, return_counts=True)
            for cls, count in zip(unique_classes.tolist(), counts.tolist()):
                if cls not in class_counts:
                    class_counts[cls] = 0
                class_counts[cls] += count
        
        # Image size statistics
        for i in range(batch_size_actual):
            h, w = images[i].shape[1], images[i].shape[2]
            image_sizes.append((h, w))
        
        if batch_idx % 10 == 0:
            print(f"Processed {samples_processed}/{num_samples} samples")
    
    # Compute final statistics
    mean = pixel_sum / pixel_count
    var = (pixel_sum_sq / pixel_count) - (mean ** 2)
    std = torch.sqrt(var)
    
    # Image size statistics
    heights, widths = zip(*image_sizes)
    size_stats = {
        'mean_height': np.mean(heights),
        'mean_width': np.mean(widths),
        'min_height': min(heights),
        'max_height': max(heights),
        'min_width': min(widths),
        'max_width': max(widths)
    }
    
    return {
        'num_samples': samples_processed,
        'image_stats': {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'size_stats': size_stats
        },
        'class_distribution': class_counts,
        'num_classes': len(class_counts)
    }


def visualize_batch(images: torch.Tensor, 
                   masks: torch.Tensor,
                   class_names: Optional[List[str]] = None,
                   denormalize: bool = True,
                   mean: List[float] = [0.485, 0.456, 0.406],
                   std: List[float] = [0.229, 0.224, 0.225]) -> Dict[str, Any]:
    """
    Prepare batch data for visualization.
    
    Args:
        images: Batch of images [B, C, H, W]
        masks: Batch of masks [B, H, W]
        class_names: List of class names for legend
        denormalize: Whether to denormalize images
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Dictionary with visualization data
    """
    batch_size = images.shape[0]
    
    # Denormalize images if needed
    if denormalize:
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1)
        images = images * std_tensor + mean_tensor
        images = torch.clamp(images, 0, 1)
    
    # Convert to numpy for visualization
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
    masks_np = masks.cpu().numpy()  # [B, H, W]
    
    # Prepare visualization data
    viz_data = {
        'images': images_np,
        'masks': masks_np,
        'batch_size': batch_size,
        'class_names': class_names,
        'unique_classes': np.unique(masks_np).tolist()
    }
    
    return viz_data


def create_class_color_map(num_classes: int) -> np.ndarray:
    """
    Create a color map for visualization of segmentation masks.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        Color map array [num_classes, 3] with RGB values in [0, 1]
    """
    # Use a predefined color palette for common cases
    if num_classes <= 10:
        colors = [
            [0, 0, 0],        # Black (background)
            [1, 0, 0],        # Red
            [0, 1, 0],        # Green  
            [0, 0, 1],        # Blue
            [1, 1, 0],        # Yellow
            [1, 0, 1],        # Magenta
            [0, 1, 1],        # Cyan
            [1, 0.5, 0],      # Orange
            [0.5, 0, 1],      # Purple
            [0, 0.5, 0]       # Dark green
        ]
        return np.array(colors[:num_classes])
    else:
        # Generate colors using HSV space for many classes
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            # Convert HSV to RGB (simplified)
            if i == 0:
                colors.append([0, 0, 0])  # Black for background
            else:
                # Simple HSV to RGB conversion
                h = hue * 6
                c = 1  # Chroma
                x = c * (1 - abs((h % 2) - 1))
                
                if h < 1:
                    rgb = [c, x, 0]
                elif h < 2:
                    rgb = [x, c, 0]
                elif h < 3:
                    rgb = [0, c, x]
                elif h < 4:
                    rgb = [0, x, c]
                elif h < 5:
                    rgb = [x, 0, c]
                else:
                    rgb = [c, 0, x]
                
                colors.append(rgb)
        
        return np.array(colors)