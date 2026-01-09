"""Simplified training loop interface."""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path

from .trainer import SegmentationTrainer
from ..models.registry import get_model
from ..data.toy_shapes import ToyShapesDataset
from ..data.transforms import get_transforms
from ..config import Config, load_config


def train_model(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    output_dir: str = "runs",
    resume_from: Optional[str] = None
) -> SegmentationTrainer:
    """
    Simplified interface to train a segmentation model.
    
    Args:
        config_path: Path to YAML configuration file
        config_dict: Configuration dictionary (alternative to config_path)
        output_dir: Output directory for logs and checkpoints
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Trained SegmentationTrainer instance
    """
    # Load configuration
    if config_path:
        config = load_config(config_path)
    elif config_dict:
        config = Config(**config_dict)
    else:
        config = Config()  # Use defaults
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = get_model(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        **config.model.__dict__
    )
    
    # Setup data loaders
    train_loader, val_loader = setup_data_loaders(config)
    
    # Create trainer
    trainer_config = {
        'num_epochs': config.training.num_epochs,
        'num_classes': config.model.num_classes,
        'optimizer': {
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay
        },
        'scheduler': {
            'type': config.training.scheduler
        },
        'loss': {
            'type': config.training.loss_type,
            'params': {
                'bce_weight': config.training.bce_weight,
                'dice_weight': config.training.dice_weight
            }
        },
        'gradient_clip': config.training.gradient_clip,
        'mixed_precision': config.training.mixed_precision,
        'accumulate_grad_batches': config.training.accumulate_grad_batches,
        'log_every': config.experiment.log_every,
        'save_every': config.experiment.save_every
    }
    
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
        output_dir=output_dir
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    # Start training
    training_history = trainer.train(config.training.num_epochs)
    
    return trainer


def setup_data_loaders(config: Config) -> tuple:
    """
    Setup training and validation data loaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get transforms
    train_transform, val_transform = get_transforms(
        image_size=config.data.image_size,
        augment=True  # Enable augmentation for training
    )
    
    # Create datasets
    train_dataset = ToyShapesDataset(
        num_samples=1000,  # Generate 1000 training samples
        image_size=config.data.image_size,
        num_shapes=config.data.num_shapes,
        noise_level=config.data.noise_level,
        blur_sigma=config.data.blur_sigma,
        occlusion_prob=config.data.occlusion_prob,
        class_imbalance=config.data.class_imbalance,
        transform=train_transform,
        seed=config.experiment.seed
    )
    
    val_dataset = ToyShapesDataset(
        num_samples=200,  # Generate 200 validation samples
        image_size=config.data.image_size,
        num_shapes=config.data.num_shapes,
        noise_level=config.data.noise_level * 0.5,  # Less noise for validation
        blur_sigma=config.data.blur_sigma * 0.5,
        occlusion_prob=config.data.occlusion_prob * 0.5,
        class_imbalance=config.data.class_imbalance,
        transform=val_transform,
        seed=config.experiment.seed + 1000  # Different seed for validation
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print(f"Created train loader with {len(train_dataset)} samples")
    print(f"Created val loader with {len(val_dataset)} samples")
    
    return train_loader, val_loader


def create_training_script(
    config_path: str = "config.yaml",
    output_dir: str = "runs/experiment_1"
):
    """
    Create a standalone training script.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
    """
    script_content = f'''#!/usr/bin/env python3
"""
Standalone training script for segmentation model.
Generated automatically by the training system.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from training.train_loop import train_model

if __name__ == "__main__":
    # Train the model
    trainer = train_model(
        config_path="{config_path}",
        output_dir="{output_dir}"
    )
    
    print("Training completed successfully!")
    print(f"Best validation metric: {{trainer.best_val_metric:.4f}}")
    print(f"Results saved to: {output_dir}")
'''
    
    # Save script
    script_path = Path("train.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created training script: {script_path}")
    return script_path


if __name__ == "__main__":
    # Example usage
    trainer = train_model(
        output_dir="runs/default_experiment"
    )
    print(f"Training completed! Best metric: {trainer.best_val_metric:.4f}")