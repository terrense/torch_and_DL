"""
Folder-based dataset loader for real image/mask pairs.

Loads images and corresponding segmentation masks from folder structure,
with support for various image formats and proper mask alignment.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Callable, Union
import os
from PIL import Image


class FolderSegmentationDataset(Dataset):
    """
    Dataset for loading image/mask pairs from folder structure.
    
    Expected folder structure:
    root/
    ├── images/
    │   ├── img1.jpg
    │   ├── img2.png
    │   └── ...
    └── masks/
        ├── img1.png
        ├── img2.png
        └── ...
    
    Or flat structure:
    root/
    ├── img1.jpg
    ├── img1_mask.png
    ├── img2.jpg
    ├── img2_mask.png
    └── ...
    """
    
    def __init__(self,
                 root: Union[str, Path],
                 transform: Optional[Callable] = None,
                 image_folder: str = "images",
                 mask_folder: str = "masks",
                 mask_suffix: str = "_mask",
                 image_extensions: List[str] = None,
                 mask_extensions: List[str] = None,
                 num_classes: Optional[int] = None):
        """
        Initialize folder segmentation dataset.
        
        Args:
            root: Root directory path
            transform: Transform function to apply to (image, mask) pairs
            image_folder: Name of images subfolder (for nested structure)
            mask_folder: Name of masks subfolder (for nested structure)
            mask_suffix: Suffix for mask files (for flat structure)
            image_extensions: Valid image file extensions
            mask_extensions: Valid mask file extensions
            num_classes: Number of classes (for validation)
        """
        self.root = Path(root)
        self.transform = transform
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.mask_suffix = mask_suffix
        self.num_classes = num_classes
        
        if image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        else:
            self.image_extensions = image_extensions
            
        if mask_extensions is None:
            self.mask_extensions = ['.png', '.bmp', '.tiff']
        else:
            self.mask_extensions = mask_extensions
        
        # Find all image/mask pairs
        self.samples = self._find_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image/mask pairs found in {root}")
    
    def _find_samples(self) -> List[Tuple[Path, Path]]:
        """
        Find all valid image/mask pairs.
        
        Returns:
            List of (image_path, mask_path) tuples
        """
        samples = []
        
        # Check for nested structure (images/ and masks/ folders)
        images_dir = self.root / self.image_folder
        masks_dir = self.root / self.mask_folder
        
        if images_dir.exists() and masks_dir.exists():
            samples = self._find_nested_samples(images_dir, masks_dir)
        else:
            # Try flat structure
            samples = self._find_flat_samples()
        
        return samples
    
    def _find_nested_samples(self, images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
        """Find samples in nested folder structure."""
        samples = []
        
        # Get all image files
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        for image_path in sorted(image_files):
            # Find corresponding mask
            image_stem = image_path.stem
            
            # Try different mask extensions
            mask_path = None
            for ext in self.mask_extensions:
                potential_mask = masks_dir / f"{image_stem}{ext}"
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            if mask_path is not None:
                samples.append((image_path, mask_path))
        
        return samples
    
    def _find_flat_samples(self) -> List[Tuple[Path, Path]]:
        """Find samples in flat folder structure."""
        samples = []
        
        # Get all image files (excluding mask files)
        image_files = []
        for ext in self.image_extensions:
            candidates = list(self.root.glob(f"*{ext}"))
            candidates.extend(self.root.glob(f"*{ext.upper()}"))
            
            # Filter out mask files
            for candidate in candidates:
                if self.mask_suffix not in candidate.stem:
                    image_files.append(candidate)
        
        for image_path in sorted(image_files):
            # Find corresponding mask
            image_stem = image_path.stem
            
            # Try different mask extensions
            mask_path = None
            for ext in self.mask_extensions:
                potential_mask = self.root / f"{image_stem}{self.mask_suffix}{ext}"
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            
            if mask_path is not None:
                samples.append((image_path, mask_path))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return image/mask pair.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, mask) tensors
        """
        image_path, mask_path = self.samples[idx]
        
        # Load image
        image = self._load_image(image_path)
        
        # Load mask
        mask = self._load_mask(mask_path)
        
        # Apply transforms if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        return image, mask
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Load image from file.
        
        Args:
            path: Path to image file
            
        Returns:
            Image tensor [C, H, W] in range [0, 1]
        """
        try:
            # Load with PIL
            pil_image = Image.open(path).convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image, dtype=np.float32) / 255.0
            
            # Convert to tensor and rearrange dimensions
            image = torch.from_numpy(image_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            
            return image
            
        except Exception as e:
            raise RuntimeError(f"Error loading image {path}: {e}")
    
    def _load_mask(self, path: Path) -> torch.Tensor:
        """
        Load segmentation mask from file.
        
        Args:
            path: Path to mask file
            
        Returns:
            Mask tensor [H, W] with class indices
        """
        try:
            # Load with PIL in grayscale mode
            pil_mask = Image.open(path).convert('L')
            
            # Convert to numpy array
            mask_array = np.array(pil_mask, dtype=np.int64)
            
            # Convert to tensor
            mask = torch.from_numpy(mask_array)
            
            # Validate class indices if num_classes is specified
            if self.num_classes is not None:
                max_class = mask.max().item()
                if max_class >= self.num_classes:
                    raise ValueError(
                        f"Mask contains class {max_class} but num_classes is {self.num_classes}"
                    )
            
            return mask
            
        except Exception as e:
            raise RuntimeError(f"Error loading mask {path}: {e}")
    
    def get_sample_paths(self, idx: int) -> Tuple[Path, Path]:
        """
        Get file paths for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_path, mask_path)
        """
        return self.samples[idx]
    
    def get_class_distribution(self) -> torch.Tensor:
        """
        Compute class distribution across the dataset.
        
        Returns:
            Tensor with pixel counts per class
        """
        if self.num_classes is None:
            raise ValueError("num_classes must be specified to compute class distribution")
        
        class_counts = torch.zeros(self.num_classes, dtype=torch.long)
        
        print("Computing class distribution...")
        for i in range(len(self)):
            if i % 100 == 0:
                print(f"Processed {i}/{len(self)} samples")
            
            _, mask = self[i]
            
            # Count pixels for each class
            for class_id in range(self.num_classes):
                class_counts[class_id] += (mask == class_id).sum()
        
        return class_counts
    
    def compute_class_weights(self, method: str = 'inverse') -> torch.Tensor:
        """
        Compute class weights for balanced training.
        
        Args:
            method: Weighting method ('inverse' or 'sqrt_inverse')
            
        Returns:
            Class weights tensor
        """
        class_counts = self.get_class_distribution()
        
        if method == 'inverse':
            # Inverse frequency weighting
            weights = 1.0 / (class_counts.float() + 1e-8)
        elif method == 'sqrt_inverse':
            # Square root inverse frequency weighting
            weights = 1.0 / torch.sqrt(class_counts.float() + 1e-8)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        return weights


def collate_segmentation_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for segmentation batches with variable-sized inputs.
    
    Pads images and masks to the maximum size in the batch.
    
    Args:
        batch: List of (image, mask) tuples
        
    Returns:
        Tuple of (batched_images, batched_masks)
    """
    images, masks = zip(*batch)
    
    # Find maximum dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    # Pad all images and masks to max size
    padded_images = []
    padded_masks = []
    
    for image, mask in zip(images, masks):
        c, h, w = image.shape
        
        # Calculate padding
        pad_h = max_h - h
        pad_w = max_w - w
        
        # Pad image (pad with zeros)
        if pad_h > 0 or pad_w > 0:
            padded_image = torch.nn.functional.pad(
                image, (0, pad_w, 0, pad_h), mode='constant', value=0
            )
            padded_mask = torch.nn.functional.pad(
                mask, (0, pad_w, 0, pad_h), mode='constant', value=0
            )
        else:
            padded_image = image
            padded_mask = mask
        
        padded_images.append(padded_image)
        padded_masks.append(padded_mask)
    
    # Stack into batches
    batched_images = torch.stack(padded_images, dim=0)
    batched_masks = torch.stack(padded_masks, dim=0)
    
    return batched_images, batched_masks


def create_folder_dataloaders(root: Union[str, Path],
                             batch_size: int = 8,
                             num_workers: int = 4,
                             train_transform: Optional[Callable] = None,
                             val_transform: Optional[Callable] = None,
                             train_split: float = 0.8,
                             val_split: float = 0.2,
                             shuffle: bool = True,
                             **dataset_kwargs) -> dict:
    """
    Create train/val dataloaders from folder dataset.
    
    Args:
        root: Root directory path
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_transform: Transform for training data
        val_transform: Transform for validation data
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        shuffle: Whether to shuffle training data
        **dataset_kwargs: Additional arguments for FolderSegmentationDataset
        
    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    full_dataset = FolderSegmentationDataset(root, **dataset_kwargs)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    if train_transform is not None:
        train_dataset.dataset.transform = train_transform
    if val_transform is not None:
        val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_segmentation_batch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_segmentation_batch,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }