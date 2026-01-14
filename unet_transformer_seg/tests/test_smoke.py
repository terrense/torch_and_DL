"""
Smoke tests for U-Net Transformer Segmentation end-to-end training.

Tests that run 30-100 training steps on toy data to verify:
- Loss decrease during training
- Proper convergence behavior  
- Checkpoint save/load functionality
- Reproducibility
"""
import pytest
import torch
import tempfile
import os
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.toy_shapes import ToyShapesDataset, ShapeConfig
from src.models.unet import UNet
from src.losses.seg_losses import get_loss_function
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import CheckpointManager


@pytest.fixture
def toy_dataset():
    """Create a small toy dataset for smoke testing."""
    from src.data.toy_shapes import ShapeConfig
    
    config = ShapeConfig(
        image_size=64,
        num_classes=3,
        shape_types=['circle', 'square'],
        blur_prob=0.0  # Disable blur to avoid compatibility issues
    )
    
    return ToyShapesDataset(
        size=100,  # Increased for more batches
        config=config,
        seed=42
    )


@pytest.fixture
def model_config():
    """Create a minimal model config for smoke tests."""
    from src.config import ModelConfig
    
    return ModelConfig(
        name='unet_test',
        input_channels=3,
        num_classes=3,
        encoder_channels=[32, 64, 128],
        decoder_channels=[64, 32, 16],
        hidden_dim=32,
        num_layers=3,
        dropout=0.1
    )


def test_training_loss_decreases(toy_dataset, model_config):
    """Test that loss decreases over 30-50 training steps."""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model directly
    model = UNet(model_config)
    model.to(device)
    
    loss_fn = get_loss_function('dice_bce', num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=8,
        shuffle=True
    )
    
    # Train for limited steps
    model.train()
    initial_losses = []
    final_losses = []
    
    step = 0
    max_steps = 50
    
    for epoch in range(10):  # Increase epochs to ensure we reach max_steps
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record losses
            if step < 5:
                initial_losses.append(loss.item())
            elif step >= max_steps - 5:
                final_losses.append(loss.item())
            
            step += 1
            
            if step >= max_steps:
                break
        
        if step >= max_steps:
            break
    
    # Verify loss decreased
    avg_initial = sum(initial_losses) / len(initial_losses)
    avg_final = sum(final_losses) / len(final_losses)
    
    assert avg_final < avg_initial, \
        f"Loss did not decrease: {avg_initial:.4f} -> {avg_final:.4f}"
    
    print(f"Loss decreased from {avg_initial:.4f} to {avg_final:.4f}")


def test_checkpoint_save_load(toy_dataset, model_config):
    """Test that checkpoints can be saved and loaded correctly."""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and optimizer
    model = UNet(model_config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = get_loss_function('dice_bce', num_classes=3)
    
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=8,
        shuffle=False
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(tmpdir),
            max_checkpoints=3
        )
        
        # Train a few steps
        model.train()
        for i, (images, masks) in enumerate(train_loader):
            if i >= 5:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Save checkpoint using the correct API
        metrics = {'val_loss': 0.5, 'val_dice': 0.75}
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics=metrics
        )
        
        # Get saved model weights
        original_weights = {
            name: param.clone() 
            for name, param in model.state_dict().items()
        }
        
        # Create new model and load checkpoint
        model2 = UNet(model_config)
        model2.to(device)
        
        # Load checkpoint using CheckpointManager
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        loaded_data = checkpoint_manager.load_checkpoint(
            model=model2,
            optimizer=optimizer2,
            load_best=True,
            map_location=device
        )
        
        # Verify checkpoint metadata
        assert loaded_data['epoch'] == 1
        assert 'val_loss' in loaded_data['metrics']
        
        # Verify models have same weights
        for name, param in model2.state_dict().items():
            assert torch.allclose(param, original_weights[name], atol=1e-6), \
                f"Checkpoint weights don't match for {name}"
        
        print("Checkpoint save/load successful")


def test_reproducibility(toy_dataset, model_config):
    """Test that training is reproducible with same seed."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_n_steps(seed, n_steps=15):
        set_seed(seed)
        
        model = UNet(model_config)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = get_loss_function('dice_bce', num_classes=3)
        
        train_loader = torch.utils.data.DataLoader(
            toy_dataset,
            batch_size=8,
            shuffle=True
        )
        
        model.train()
        losses = []
        
        for i, (images, masks) in enumerate(train_loader):
            if i >= n_steps:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        return losses
    
    # Train twice with same seed
    losses1 = train_n_steps(seed=42, n_steps=15)
    losses2 = train_n_steps(seed=42, n_steps=15)
    
    # Verify losses are identical (or very close due to GPU non-determinism)
    for i, (l1, l2) in enumerate(zip(losses1, losses2)):
        assert abs(l1 - l2) < 1e-4, \
            f"Training not reproducible at step {i}: {l1} != {l2}"
    
    print("Reproducibility test passed")


def test_overfitting_on_single_batch(toy_dataset, model_config):
    """Test that model can overfit on a single batch (sanity check)."""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(model_config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = get_loss_function('dice_bce', num_classes=3)
    
    # Get single batch
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=4,
        shuffle=False
    )
    
    images, masks = next(iter(train_loader))
    images = images.to(device)
    masks = masks.to(device)
    
    # Train on same batch repeatedly
    model.train()
    losses = []
    
    for i in range(100):
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # Loss should decrease significantly
    initial_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10
    
    assert final_loss < initial_loss * 0.5, \
        f"Model failed to overfit: {initial_loss:.4f} -> {final_loss:.4f}"
    
    print(f"Overfitting test passed: {initial_loss:.4f} -> {final_loss:.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
