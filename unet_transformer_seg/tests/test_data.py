"""
Unit tests for data pipeline components in U-Net Transformer Segmentation.

Tests dataset generation, transforms, and data loading functionality.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.toy_shapes import ToyShapesDataset, ShapeConfig, create_toy_dataset
from src.data.transforms import SegmentationTransforms
from src.data.folder_dataset import FolderDataset
from src.utils.tensor_utils import assert_shape, check_nan_inf


class TestToyShapesDataset:
    """Test toy shapes dataset generation."""
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        config = ShapeConfig(image_size=128, num_classes=4)
        dataset = ToyShapesDataset(size=10, config=config, seed=42)
        
        assert len(dataset) == 10
        assert dataset.config.image_size == 128
        assert dataset.config.num_classes == 4
    
    def test_sample_generation(self):
        """Test individual sample generation."""
        config = ShapeConfig(image_size=64, num_classes=3)
        dataset = ToyShapesDataset(size=5, config=config, seed=42)
        
        image, mask = dataset[0]
        
        # Check shapes
        assert_shape(image, "3,64,64", "image")
        assert_shape(mask, "64,64", "mask")
        
        # Check data types
        assert image.dtype == torch.float32
        assert mask.dtype == torch.long
        
        # Check value ranges
        assert image.min() >= 0.0
        assert image.max() <= 1.0
        assert mask.min() >= 0
        assert mask.max() < config.num_classes
        
        # Check for NaN/Inf
        check_nan_inf(image, "image")
        check_nan_inf(mask.float(), "mask")
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        config = ShapeConfig(image_size=32, num_classes=3)
        
        dataset1 = ToyShapesDataset(size=3, config=config, seed=123)
        dataset2 = ToyShapesDataset(size=3, config=config, seed=123)
        
        for i in range(3):
            img1, mask1 = dataset1[i]
            img2, mask2 = dataset2[i]
            
            assert torch.allclose(img1, img2, atol=1e-6)
            assert torch.equal(mask1, mask2)
    
    def test_shape_types(self):
        """Test different shape type configurations."""
        # Test with different shape types
        config = ShapeConfig(
            image_size=64,
            shape_types=['circle', 'square'],
            num_classes=3  # background + 2 shapes
        )
        dataset = ToyShapesDataset(size=10, config=config, seed=42)
        
        # Generate several samples and check mask values
        mask_values = set()
        for i in range(10):
            _, mask = dataset[i]
            mask_values.update(mask.unique().tolist())
        
        # Should only have background (0) and the two shape classes (1, 2)
        assert mask_values.issubset({0, 1, 2})
    
    def test_class_weights(self):
        """Test class weight generation."""
        config = ShapeConfig(num_classes=4)
        dataset = ToyShapesDataset(size=1, config=config)
        
        weights = dataset.get_class_weights()
        assert len(weights) == 4
        assert weights.dtype == torch.float32
        assert torch.all(weights >= 0)
    
    def test_dataset_splits(self):
        """Test dataset split creation."""
        config = ShapeConfig(image_size=32, num_classes=3)
        datasets = create_toy_dataset(
            config=config,
            train_size=10,
            val_size=5,
            test_size=3,
            seed=42
        )
        
        assert len(datasets['train']) == 10
        assert len(datasets['val']) == 5
        assert len(datasets['test']) == 3
        
        # Test that splits are different
        train_img, _ = datasets['train'][0]
        val_img, _ = datasets['val'][0]
        assert not torch.allclose(train_img, val_img)


class TestSegmentationTransforms:
    """Test segmentation transforms."""
    
    def test_resize_transform(self):
        """Test resize with mask alignment."""
        transforms = SegmentationTransforms(
            image_size=64,
            normalize=False,
            augment=False
        )
        
        # Create test data
        image = torch.randn(3, 128, 128)
        mask = torch.randint(0, 4, (128, 128))
        
        # Apply transforms
        transformed = transforms(image, mask)
        t_image, t_mask = transformed['image'], transformed['mask']
        
        # Check output shapes
        assert_shape(t_image, "3,64,64", "transformed_image")
        assert_shape(t_mask, "64,64", "transformed_mask")
        
        # Check data types
        assert t_image.dtype == torch.float32
        assert t_mask.dtype == torch.long
    
    def test_normalization(self):
        """Test image normalization."""
        transforms = SegmentationTransforms(
            image_size=32,
            normalize=True,
            mean=[0.5, 0.5, 0.5],
            std=[0.2, 0.2, 0.2]
        )
        
        # Create test image with known values
        image = torch.ones(3, 32, 32) * 0.7  # All pixels = 0.7
        mask = torch.zeros(32, 32)
        
        transformed = transforms(image, mask)
        t_image = transformed['image']
        
        # Check normalization: (0.7 - 0.5) / 0.2 = 1.0
        expected = torch.ones_like(t_image) * 1.0
        assert torch.allclose(t_image, expected, atol=1e-6)
    
    def test_augmentation_consistency(self):
        """Test that augmentations maintain image-mask alignment."""
        transforms = SegmentationTransforms(
            image_size=64,
            augment=True,
            flip_prob=1.0  # Always flip for testing
        )
        
        # Create test data with specific pattern
        image = torch.zeros(3, 64, 64)
        image[:, :32, :] = 1.0  # Top half bright
        
        mask = torch.zeros(64, 64)
        mask[:32, :] = 1  # Top half class 1
        
        transformed = transforms(image, mask)
        t_image, t_mask = transformed['image']
        
        # After horizontal flip, bright region should be on bottom
        # (assuming horizontal flip was applied)
        # This is a probabilistic test, so we just check consistency
        bright_pixels = (t_image[0] > 0.5).float()
        mask_pixels = (t_mask == 1).float()
        
        # Image and mask should have same spatial pattern
        correlation = torch.corrcoef(torch.stack([
            bright_pixels.flatten(),
            mask_pixels.flatten()
        ]))[0, 1]
        
        assert correlation > 0.8  # Strong positive correlation


class TestDataLoading:
    """Test data loading and batching."""
    
    def test_dataloader_batching(self):
        """Test DataLoader with toy dataset."""
        config = ShapeConfig(image_size=32, num_classes=3)
        dataset = ToyShapesDataset(size=8, config=config, seed=42)
        
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        images, masks = batch
        
        # Check batch shapes
        assert_shape(images, "4,3,32,32", "batch_images")
        assert_shape(masks, "4,32,32", "batch_masks")
        
        # Check data types
        assert images.dtype == torch.float32
        assert masks.dtype == torch.long
        
        # Check value ranges
        assert images.min() >= 0.0
        assert images.max() <= 1.0
        assert masks.min() >= 0
        assert masks.max() < config.num_classes
    
    def test_variable_batch_sizes(self):
        """Test handling of different batch sizes."""
        config = ShapeConfig(image_size=32, num_classes=3)
        dataset = ToyShapesDataset(size=7, config=config, seed=42)
        
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        batches = list(dataloader)
        
        # Should have 3 batches: [3, 3, 1]
        assert len(batches) == 3
        assert batches[0][0].shape[0] == 3  # First batch size
        assert batches[1][0].shape[0] == 3  # Second batch size  
        assert batches[2][0].shape[0] == 1  # Last batch size
    
    def test_transforms_in_dataloader(self):
        """Test transforms integration with DataLoader."""
        config = ShapeConfig(image_size=64, num_classes=3)
        dataset = ToyShapesDataset(size=4, config=config, seed=42)
        
        transforms = SegmentationTransforms(
            image_size=32,  # Resize from 64 to 32
            normalize=True
        )
        
        # Apply transforms to dataset
        class TransformedDataset:
            def __init__(self, dataset, transforms):
                self.dataset = dataset
                self.transforms = transforms
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                image, mask = self.dataset[idx]
                transformed = self.transforms(image, mask)
                return transformed['image'], transformed['mask']
        
        transformed_dataset = TransformedDataset(dataset, transforms)
        dataloader = DataLoader(transformed_dataset, batch_size=2, num_workers=0)
        
        batch = next(iter(dataloader))
        images, masks = batch
        
        # Check that transforms were applied
        assert_shape(images, "2,3,32,32", "transformed_batch_images")
        assert_shape(masks, "2,32,32", "transformed_batch_masks")


class TestTensorContracts:
    """Test tensor contract validation."""
    
    def test_shape_assertions(self):
        """Test shape assertion utilities."""
        tensor = torch.randn(2, 3, 64, 64)
        
        # Should pass
        assert_shape(tensor, "2,3,64,64", "test_tensor")
        assert_shape(tensor, "B,3,64,64", "test_tensor")  # B can be any size
        
        # Should fail
        with pytest.raises(AssertionError):
            assert_shape(tensor, "2,3,32,32", "test_tensor")
    
    def test_nan_inf_detection(self):
        """Test NaN/Inf detection."""
        # Normal tensor should pass
        normal_tensor = torch.randn(10, 10)
        check_nan_inf(normal_tensor, "normal_tensor")  # Should not raise
        
        # NaN tensor should fail
        nan_tensor = torch.randn(10, 10)
        nan_tensor[0, 0] = float('nan')
        
        with pytest.raises(ValueError, match="NaN"):
            check_nan_inf(nan_tensor, "nan_tensor")
        
        # Inf tensor should fail
        inf_tensor = torch.randn(10, 10)
        inf_tensor[0, 0] = float('inf')
        
        with pytest.raises(ValueError, match="Inf"):
            check_nan_inf(inf_tensor, "inf_tensor")


if __name__ == "__main__":
    # Run basic tests
    print("Running data pipeline tests...")
    
    # Test dataset creation
    test_dataset = TestToyShapesDataset()
    test_dataset.test_dataset_creation()
    test_dataset.test_sample_generation()
    test_dataset.test_reproducibility()
    print("✓ Dataset tests passed")
    
    # Test transforms
    test_transforms = TestSegmentationTransforms()
    test_transforms.test_resize_transform()
    test_transforms.test_normalization()
    print("✓ Transform tests passed")
    
    # Test data loading
    test_loading = TestDataLoading()
    test_loading.test_dataloader_batching()
    test_loading.test_variable_batch_sizes()
    print("✓ DataLoader tests passed")
    
    # Test tensor contracts
    test_contracts = TestTensorContracts()
    test_contracts.test_shape_assertions()
    test_contracts.test_nan_inf_detection()
    print("✓ Tensor contract tests passed")
    
    print("All data pipeline tests completed successfully!")