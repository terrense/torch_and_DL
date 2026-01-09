"""
Toy shapes dataset generator for segmentation tasks.

Creates synthetic segmentation data with multiple shape types, configurable noise,
blur, occlusion, and class imbalance options.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ShapeConfig:
    """Configuration for shape generation parameters."""
    image_size: int = 256
    num_classes: int = 4  # background + 3 shape types
    shape_types: List[str] = None
    min_shapes_per_image: int = 1
    max_shapes_per_image: int = 5
    min_shape_size: int = 20
    max_shape_size: int = 80
    noise_std: float = 0.05
    blur_prob: float = 0.2
    blur_kernel_size: int = 3
    occlusion_prob: float = 0.1
    class_weights: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.shape_types is None:
            self.shape_types = ['circle', 'square', 'triangle']
        if self.class_weights is None:
            # Equal weights for all classes (background gets weight 0.1)
            self.class_weights = [0.1] + [0.9 / len(self.shape_types)] * len(self.shape_types)


class ToyShapesDataset(Dataset):
    """
    Synthetic segmentation dataset with configurable shape generation.
    
    Generates images with multiple geometric shapes and corresponding segmentation masks.
    Supports various augmentation options including noise, blur, and occlusion.
    """
    
    def __init__(self, 
                 size: int = 1000,
                 config: Optional[ShapeConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize toy shapes dataset.
        
        Args:
            size: Number of samples in the dataset
            config: Shape generation configuration
            seed: Random seed for reproducible generation
        """
        self.size = size
        self.config = config or ShapeConfig()
        
        # Set up random number generator
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            torch.manual_seed(seed)
        else:
            self.rng = np.random.RandomState()
            
        # Create class to shape type mapping
        self.class_to_shape = {0: 'background'}
        for i, shape_type in enumerate(self.config.shape_types):
            self.class_to_shape[i + 1] = shape_type
            
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        assert self.config.image_size > 0, "Image size must be positive"
        assert self.config.num_classes == len(self.config.shape_types) + 1, \
            "num_classes must equal len(shape_types) + 1 (for background)"
        assert self.config.min_shape_size < self.config.max_shape_size, \
            "min_shape_size must be less than max_shape_size"
        assert len(self.config.class_weights) == self.config.num_classes, \
            "class_weights length must match num_classes"
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single sample.
        
        Args:
            idx: Sample index (used for reproducible generation)
            
        Returns:
            Tuple of (image, mask) where:
                - image: [3, H, W] RGB image tensor in range [0, 1]
                - mask: [H, W] segmentation mask with class indices
        """
        # Use index for reproducible generation
        sample_seed = hash((idx, self.size)) % (2**32)
        sample_rng = np.random.RandomState(sample_seed)
        
        # Generate image and mask
        image, mask = self._generate_sample(sample_rng)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def _generate_sample(self, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single image and mask pair.
        
        Args:
            rng: Random number generator for this sample
            
        Returns:
            Tuple of (image, mask) as numpy arrays
        """
        size = self.config.image_size
        
        # Initialize image and mask
        image = np.zeros((3, size, size), dtype=np.float32)
        mask = np.zeros((size, size), dtype=np.int64)
        
        # Determine number of shapes to generate
        num_shapes = rng.randint(
            self.config.min_shapes_per_image,
            self.config.max_shapes_per_image + 1
        )
        
        # Generate shapes
        for _ in range(num_shapes):
            shape_type = rng.choice(self.config.shape_types)
            class_id = self.config.shape_types.index(shape_type) + 1
            
            # Generate shape parameters
            center_x = rng.randint(size // 4, 3 * size // 4)
            center_y = rng.randint(size // 4, 3 * size // 4)
            shape_size = rng.randint(self.config.min_shape_size, self.config.max_shape_size)
            
            # Generate random color for this shape
            color = rng.rand(3) * 0.8 + 0.2  # Avoid very dark colors
            
            # Create shape mask and apply to image
            shape_mask = self._create_shape(
                shape_type, center_x, center_y, shape_size, size, rng
            )
            
            # Apply shape to image (with potential occlusion)
            for c in range(3):
                image[c] = np.where(shape_mask, color[c], image[c])
            
            # Update segmentation mask (later shapes can occlude earlier ones)
            mask = np.where(shape_mask, class_id, mask)
        
        # Apply post-processing effects
        image = self._apply_noise(image, rng)
        image = self._apply_blur(image, rng)
        
        # Apply occlusion
        if rng.rand() < self.config.occlusion_prob:
            image, mask = self._apply_occlusion(image, mask, rng)
        
        # Ensure image is in [0, 1] range
        image = np.clip(image, 0.0, 1.0)
        
        return image, mask
    
    def _create_shape(self, 
                     shape_type: str, 
                     center_x: int, 
                     center_y: int, 
                     size: int, 
                     image_size: int,
                     rng: np.random.RandomState) -> np.ndarray:
        """
        Create a binary mask for a specific shape.
        
        Args:
            shape_type: Type of shape ('circle', 'square', 'triangle')
            center_x, center_y: Shape center coordinates
            size: Shape size parameter
            image_size: Size of the image canvas
            rng: Random number generator
            
        Returns:
            Binary mask for the shape
        """
        mask = np.zeros((image_size, image_size), dtype=bool)
        
        if shape_type == 'circle':
            y, x = np.ogrid[:image_size, :image_size]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            radius = size // 2
            mask = dist_from_center <= radius
            
        elif shape_type == 'square':
            half_size = size // 2
            x1, x2 = max(0, center_x - half_size), min(image_size, center_x + half_size)
            y1, y2 = max(0, center_y - half_size), min(image_size, center_y + half_size)
            mask[y1:y2, x1:x2] = True
            
        elif shape_type == 'triangle':
            # Create equilateral triangle
            height = int(size * 0.866)  # sqrt(3)/2 * size
            half_base = size // 2
            
            y, x = np.ogrid[:image_size, :image_size]
            
            # Triangle vertices (pointing up)
            top_y = center_y - height // 2
            bottom_y = center_y + height // 2
            left_x = center_x - half_base
            right_x = center_x + half_base
            
            # Create triangle using three line equations
            # Top edge (horizontal)
            cond1 = y >= top_y
            
            # Left edge
            if height > 0:
                left_slope = height / half_base
                cond2 = (y - top_y) <= left_slope * (x - center_x + half_base)
            else:
                cond2 = True
                
            # Right edge  
            if height > 0:
                right_slope = height / half_base
                cond3 = (y - top_y) <= right_slope * (center_x + half_base - x)
            else:
                cond3 = True
            
            # Bottom boundary
            cond4 = y <= bottom_y
            
            mask = cond1 & cond2 & cond3 & cond4
            
        return mask
    
    def _apply_noise(self, image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Apply Gaussian noise to the image."""
        if self.config.noise_std > 0:
            noise = rng.normal(0, self.config.noise_std, image.shape)
            image = image + noise.astype(np.float32)
        return image
    
    def _apply_blur(self, image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Apply Gaussian blur to the image."""
        if rng.rand() < self.config.blur_prob:
            # Convert to tensor for blur operation
            image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add batch dim
            
            # Create Gaussian kernel
            kernel_size = self.config.blur_kernel_size
            sigma = kernel_size / 6.0  # Standard sigma for given kernel size
            
            # Apply Gaussian blur
            image_tensor = F.gaussian_blur(image_tensor, kernel_size, sigma)
            
            # Convert back to numpy
            image = image_tensor.squeeze(0).numpy()
            
        return image
    
    def _apply_occlusion(self, 
                        image: np.ndarray, 
                        mask: np.ndarray, 
                        rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random rectangular occlusion."""
        size = self.config.image_size
        
        # Random occlusion rectangle
        occ_width = rng.randint(size // 8, size // 4)
        occ_height = rng.randint(size // 8, size // 4)
        occ_x = rng.randint(0, size - occ_width)
        occ_y = rng.randint(0, size - occ_height)
        
        # Apply occlusion (set to black/background)
        image[:, occ_y:occ_y+occ_height, occ_x:occ_x+occ_width] = 0.0
        mask[occ_y:occ_y+occ_height, occ_x:occ_x+occ_width] = 0  # Background class
        
        return image, mask
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss computation."""
        return torch.tensor(self.config.class_weights, dtype=torch.float32)
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return [self.class_to_shape[i] for i in range(self.config.num_classes)]
    
    def visualize_sample(self, idx: int) -> Dict[str, Any]:
        """
        Generate and return visualization data for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with image, mask, and metadata
        """
        image, mask = self[idx]
        
        return {
            'image': image,
            'mask': mask,
            'class_names': self.get_class_names(),
            'config': self.config,
            'sample_idx': idx
        }


def create_toy_dataset(config: Optional[ShapeConfig] = None, 
                      train_size: int = 1000,
                      val_size: int = 200,
                      test_size: int = 200,
                      seed: Optional[int] = None) -> Dict[str, ToyShapesDataset]:
    """
    Create train/val/test splits of the toy shapes dataset.
    
    Args:
        config: Shape generation configuration
        train_size: Number of training samples
        val_size: Number of validation samples  
        test_size: Number of test samples
        seed: Random seed for reproducible splits
        
    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    if seed is not None:
        base_seed = seed
    else:
        base_seed = 42
    
    datasets = {
        'train': ToyShapesDataset(train_size, config, base_seed),
        'val': ToyShapesDataset(val_size, config, base_seed + 1),
        'test': ToyShapesDataset(test_size, config, base_seed + 2)
    }
    
    return datasets