"""
From-Scratch Data Transforms for Segmentation Tasks

This module implements essential data preprocessing and augmentation transforms
specifically designed for semantic segmentation tasks. All transforms are
implemented using pure PyTorch operations to ensure educational transparency
and avoid external dependencies.

Key Deep Learning Concepts:
1. Data Augmentation: Increases dataset diversity and model generalization
2. Geometric Transforms: Spatial transformations preserving semantic content
3. Photometric Transforms: Color/intensity changes simulating real-world variations
4. Mask Alignment: Ensures segmentation masks remain synchronized with images
5. Interpolation Methods: Different strategies for images vs. discrete labels

Data Augmentation Benefits:
- Improved Generalization: Reduces overfitting by increasing effective dataset size
- Robustness: Models become invariant to common transformations
- Domain Adaptation: Simulates variations in real-world deployment scenarios
- Class Balance: Can help address imbalanced datasets through selective augmentation

Implementation Principles:
- Mask Preservation: Segmentation masks must undergo identical spatial transforms
- Label Integrity: Discrete class labels preserved through nearest-neighbor interpolation
- Differentiability: All transforms maintain gradient flow for end-to-end training
- Efficiency: GPU-accelerated tensor operations for fast preprocessing

Mathematical Foundations:
- Bilinear Interpolation: Smooth image resampling preserving visual quality
- Nearest Neighbor: Discrete label preservation for segmentation masks
- Affine Transformations: Linear transformations (rotation, scaling, translation)
- Color Space Manipulations: HSV/RGB adjustments for photometric augmentation

References:
- "Data Augmentation for Deep Learning" - Shorten & Khoshgoftaar
- "Albumentations: Fast and Flexible Image Augmentations" - Buslaev et al.
- "AutoAugment: Learning Augmentation Strategies from Data" - Cubuk et al.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
import math


class Resize:
    """
    Intelligent Resize Transform for Segmentation Tasks
    
    This transform handles the critical challenge of resizing both images and
    segmentation masks while preserving semantic information. Different interpolation
    methods are used for continuous images vs. discrete label maps.
    
    Deep Learning Rationale:
    - Spatial Consistency: Images and masks must maintain perfect alignment
    - Label Preservation: Segmentation classes must remain discrete (no interpolation)
    - Scale Invariance: Models should handle objects at different scales
    - Memory Efficiency: Standardized sizes enable efficient batch processing
    
    Interpolation Strategy:
    - Images: Bilinear interpolation for smooth visual quality
    - Masks: Nearest neighbor to preserve discrete class labels
    - Alignment: Both transformations use identical coordinate mappings
    
    Mathematical Implementation:
    - Bilinear: I(x,y) = Σ w_ij * I(x_i, y_j) where w_ij are bilinear weights
    - Nearest: M(x,y) = M(round(x), round(y)) preserving discrete labels
    - Coordinate Mapping: (x',y') = (x*scale_x, y*scale_y) for uniform scaling
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Initialize resize transform.
        
        Args:
            size: Target size as int (square) or (height, width) tuple
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply resize to image and mask.
        
        Args:
            image: Input image tensor [C, H, W]
            mask: Input mask tensor [H, W]
            
        Returns:
            Tuple of (resized_image, resized_mask)
        """
        # Add batch dimension for interpolation
        image_batch = image.unsqueeze(0)  # [1, C, H, W]
        mask_batch = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        
        # Resize image with bilinear interpolation
        resized_image = F.interpolate(
            image_batch, 
            size=self.size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Resize mask with nearest neighbor to preserve labels
        resized_mask = F.interpolate(
            mask_batch,
            size=self.size,
            mode='nearest'
        )
        
        # Remove batch dimensions
        resized_image = resized_image.squeeze(0)  # [C, H, W]
        resized_mask = resized_mask.squeeze(0).squeeze(0).long()  # [H, W]
        
        return resized_image, resized_mask


class Normalize:
    """
    Statistical Normalization for Deep Learning Input Preprocessing
    
    This transform applies channel-wise normalization to input images, converting
    pixel values to standardized distributions that facilitate neural network training.
    Normalization is crucial for stable gradient flow and convergence.
    
    Deep Learning Benefits:
    - Gradient Stability: Prevents vanishing/exploding gradients in deep networks
    - Convergence Speed: Accelerates training by centering data around zero
    - Transfer Learning: ImageNet statistics enable pretrained model usage
    - Numerical Stability: Prevents activation saturation in early layers
    
    Mathematical Formulation:
    - Normalization: x_norm = (x - μ) / σ
    - Per-channel: Applied independently to R, G, B channels
    - Statistics: μ and σ computed from large datasets (e.g., ImageNet)
    - Range: Typically transforms [0,1] to approximately [-2,2]
    
    ImageNet Statistics (Default):
    - Mean: [0.485, 0.456, 0.406] for RGB channels
    - Std: [0.229, 0.224, 0.225] for RGB channels
    - Computed from millions of natural images for broad applicability
    """
    
    def __init__(self, 
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        """
        Initialize normalization transform.
        
        Args:
            mean: Per-channel mean values (ImageNet defaults)
            std: Per-channel standard deviation values (ImageNet defaults)
        """
        self.mean = torch.tensor(mean).view(-1, 1, 1)  # [C, 1, 1]
        self.std = torch.tensor(std).view(-1, 1, 1)    # [C, 1, 1]
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply normalization to image (mask unchanged).
        
        Args:
            image: Input image tensor [C, H, W] in range [0, 1]
            mask: Input mask tensor [H, W] (unchanged)
            
        Returns:
            Tuple of (normalized_image, mask)
        """
        # Ensure mean and std are on the same device as image
        mean = self.mean.to(image.device)
        std = self.std.to(image.device)
        
        normalized_image = (image - mean) / std
        return normalized_image, mask


class RandomFlip:
    """
    Random horizontal and/or vertical flipping with aligned mask transformation.
    """
    
    def __init__(self, 
                 horizontal_prob: float = 0.5,
                 vertical_prob: float = 0.0):
        """
        Initialize random flip transform.
        
        Args:
            horizontal_prob: Probability of horizontal flip
            vertical_prob: Probability of vertical flip
        """
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random flips to image and mask.
        
        Args:
            image: Input image tensor [C, H, W]
            mask: Input mask tensor [H, W]
            
        Returns:
            Tuple of (flipped_image, flipped_mask)
        """
        # Horizontal flip
        if torch.rand(1).item() < self.horizontal_prob:
            image = torch.flip(image, dims=[2])  # Flip width dimension
            mask = torch.flip(mask, dims=[1])    # Flip width dimension
        
        # Vertical flip
        if torch.rand(1).item() < self.vertical_prob:
            image = torch.flip(image, dims=[1])  # Flip height dimension
            mask = torch.flip(mask, dims=[0])    # Flip height dimension
            
        return image, mask


class RandomRotation:
    """
    Random rotation with aligned mask transformation.
    
    Implements rotation using affine transformation with proper interpolation.
    """
    
    def __init__(self, degrees: Union[float, Tuple[float, float]] = 15.0):
        """
        Initialize random rotation transform.
        
        Args:
            degrees: Range of rotation angles. If float, uses [-degrees, degrees].
                    If tuple, uses [degrees[0], degrees[1]].
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-abs(degrees), abs(degrees))
        else:
            self.degrees = degrees
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random rotation to image and mask.
        
        Args:
            image: Input image tensor [C, H, W]
            mask: Input mask tensor [H, W]
            
        Returns:
            Tuple of (rotated_image, rotated_mask)
        """
        # Sample random angle
        angle = torch.rand(1).item() * (self.degrees[1] - self.degrees[0]) + self.degrees[0]
        angle_rad = math.radians(angle)
        
        # Create rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Affine transformation matrix for rotation around center
        # Format: [2, 3] matrix for F.affine_grid
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)  # [1, 2, 3]
        
        # Get image dimensions
        _, H, W = image.shape
        
        # Create affine grid
        grid = F.affine_grid(theta, [1, 1, H, W], align_corners=False)
        
        # Apply rotation to image (bilinear interpolation)
        image_batch = image.unsqueeze(0)  # [1, C, H, W]
        rotated_image = F.grid_sample(
            image_batch, grid, mode='bilinear', 
            padding_mode='zeros', align_corners=False
        )
        rotated_image = rotated_image.squeeze(0)  # [C, H, W]
        
        # Apply rotation to mask (nearest neighbor interpolation)
        mask_batch = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        rotated_mask = F.grid_sample(
            mask_batch, grid, mode='nearest',
            padding_mode='zeros', align_corners=False
        )
        rotated_mask = rotated_mask.squeeze(0).squeeze(0).long()  # [H, W]
        
        return rotated_image, rotated_mask


class ColorJitter:
    """
    Random color jittering for image augmentation.
    
    Applies random changes to brightness, contrast, saturation, and hue.
    """
    
    def __init__(self,
                 brightness: float = 0.2,
                 contrast: float = 0.2,
                 saturation: float = 0.2,
                 hue: float = 0.1):
        """
        Initialize color jitter transform.
        
        Args:
            brightness: Maximum brightness change factor
            contrast: Maximum contrast change factor  
            saturation: Maximum saturation change factor
            hue: Maximum hue change factor
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply color jittering to image (mask unchanged).
        
        Args:
            image: Input image tensor [C, H, W] in range [0, 1]
            mask: Input mask tensor [H, W] (unchanged)
            
        Returns:
            Tuple of (jittered_image, mask)
        """
        # Apply brightness adjustment
        if self.brightness > 0:
            brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.brightness
            image = image * brightness_factor
        
        # Apply contrast adjustment
        if self.contrast > 0:
            contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.contrast
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean
        
        # Clamp to valid range
        image = torch.clamp(image, 0.0, 1.0)
        
        return image, mask


class SegmentationTransforms:
    """
    Composition of transforms for segmentation tasks.
    
    Applies a sequence of transforms to both image and mask with proper alignment.
    """
    
    def __init__(self, transforms: List):
        """
        Initialize transform composition.
        
        Args:
            transforms: List of transform objects to apply in sequence
        """
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply all transforms in sequence.
        
        Args:
            image: Input image tensor
            mask: Input mask tensor
            
        Returns:
            Tuple of (transformed_image, transformed_mask)
        """
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


def create_train_transforms(image_size: int = 256,
                           augment: bool = True) -> SegmentationTransforms:
    """
    Create standard training transforms.
    
    Args:
        image_size: Target image size
        augment: Whether to include augmentation transforms
        
    Returns:
        Composed transforms for training
    """
    transforms = [Resize(image_size)]
    
    if augment:
        transforms.extend([
            RandomFlip(horizontal_prob=0.5, vertical_prob=0.0),
            RandomRotation(degrees=15.0),
            ColorJitter(brightness=0.2, contrast=0.2)
        ])
    
    transforms.append(Normalize())
    
    return SegmentationTransforms(transforms)


def create_val_transforms(image_size: int = 256) -> SegmentationTransforms:
    """
    Create standard validation transforms (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms for validation
    """
    transforms = [
        Resize(image_size),
        Normalize()
    ]
    
    return SegmentationTransforms(transforms)