#!/usr/bin/env python3
"""
Training script for U-Net Transformer Segmentation.

Provides command-line interface for training segmentation models with
configuration-based setup and comprehensive logging.
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import load_config
from data.toy_shapes import ToyShapesDataset
from data.transforms import get_transforms
from models.registry import create_model_from_config
from training.trainer import SegmentationTrainer
from utils.logging_utils import setup_logger
from utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def create_dataloaders(config):
    """Create training and validation data loaders."""
    data_config = config.get('data', {})
    
    # Get transforms
    train_transform, val_transform = get_transforms(
        image_size=tuple(data_config.get('image_size', [256, 256])),
        augment=data_config.get('augment', True)
    )
    
    # Create datasets
    train_dataset = ToyShapesDataset(
        num_samples=data_config.get('train_samples', 1000),
        image_size=tuple(data_config.get('image_size', [256, 256])),
        num_classes=config.get('num_classes', 3),
        shapes=data_config.get('shapes', ['circle', 'square', 'triangle']),
        min_shapes=data_config.get('min_shapes', 1),
        max_shapes=data_config.get('max_shapes', 5),
        noise_level=data_config.get('noise_level', 0.1),
        blur_prob=data_config.get('blur_prob', 0.3),
        occlusion_prob=data_config.get('occlusion_prob', 0.2),
        transform=train_transform
    )
    
    val_dataset = ToyShapesDataset(
        num_samples=data_config.get('val_samples', 200),
        image_size=tuple(data_config.get('image_size', [256, 256])),
        num_classes=config.get('num_classes', 3),
        shapes=data_config.get('shapes', ['circle', 'square', 'triangle']),
        min_shapes=data_config.get('min_shapes', 1),
        max_shapes=data_config.get('max_shapes', 5),
        noise_level=data_config.get('noise_level', 0.1),
        blur_prob=data_config.get('blur_prob', 0.0),
        occlusion_prob=data_config.get('occlusion_prob', 0.0),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 16),
        shuffle=True,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 16),
        shuffle=False,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train U-Net Transformer Segmentation model')
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Training device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(args.output_dir / 'train.log', level=log_level)
    
    logger.info("Starting U-Net Transformer Segmentation training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Set random seed
    seed = args.seed or config.__dict__.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(config.__dict__)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Create model
        logger.info("Creating model...")
        model = create_model_from_config(config.__dict__)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {num_params:,} parameters")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = SegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.__dict__,
            device=device,
            output_dir=args.output_dir
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Run training
        num_epochs = config.__dict__.get('num_epochs', 100)
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        history = trainer.train(num_epochs=num_epochs)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation metric: {trainer.best_val_metric:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
