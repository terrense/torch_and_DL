#!/usr/bin/env python3
"""
Evaluation script for U-Net Transformer Segmentation.

Provides command-line interface for evaluating trained segmentation models
with comprehensive metrics including IoU, Dice, and pixel accuracy.
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import load_config
from data.toy_shapes import ToyShapesDataset
from data.transforms import get_transforms
from models.registry import create_model_from_config
from evaluation.evaluator import ModelEvaluator
from utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def create_test_dataloader(config):
    """Create test data loader."""
    data_config = config.get('data', {})
    
    # Get validation transforms (no augmentation)
    _, val_transform = get_transforms(
        image_size=tuple(data_config.get('image_size', [256, 256])),
        augment=False
    )
    
    # Create test dataset
    test_dataset = ToyShapesDataset(
        num_samples=data_config.get('test_samples', 500),
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
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.get('batch_size', 16),
        shuffle=False,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )
    
    return test_loader


def main():
    parser = argparse.ArgumentParser(description='Evaluate U-Net Transformer Segmentation model')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=Path, default=None,
                        help='Path to configuration file (optional)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Evaluation device (cuda/cpu)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(args.output_dir / 'evaluate.log', level=log_level)
    
    logger.info("Starting U-Net Transformer Segmentation evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    try:
        # Load checkpoint
        logger.info("Loading model from checkpoint...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Get config from checkpoint or load from file
        if args.config is not None:
            config = load_config(args.config).__dict__
            logger.info("Configuration loaded from file")
        elif 'config' in checkpoint:
            config = checkpoint['config']
            logger.info("Configuration loaded from checkpoint")
        else:
            # Use default configuration
            config = {
                'model': {'name': 'unet', 'in_channels': 3, 'num_classes': 3},
                'num_classes': 3,
                'data': {'batch_size': 16, 'test_samples': 500, 'image_size': [256, 256]}
            }
            logger.info("Using default configuration")
        
        # Create model
        logger.info("Creating model...")
        model = create_model_from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")
        
        # Create test data loader
        logger.info("Creating test data loader...")
        test_loader = create_test_dataloader(config)
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Create evaluator
        num_classes = config.get('num_classes', 3)
        class_names = config.get('class_names', None)
        
        evaluator = ModelEvaluator(
            model=model,
            device=device,
            num_classes=num_classes,
            class_names=class_names
        )
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.evaluate_dataset(
            data_loader=test_loader,
            save_results=True,
            output_dir=str(args.output_dir)
        )
        
        # Print summary
        logger.info("Evaluation completed!")
        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Samples evaluated: {results['num_samples']}")
        logger.info(f"Pixel accuracy: {results['pixel_accuracy']:.4f}")
        logger.info(f"Mean IoU: {results['mean_iou']:.4f}")
        logger.info(f"Mean Dice: {results['mean_dice']:.4f}")
        logger.info(f"Evaluation time: {results['evaluation_time']:.1f}s")
        logger.info(f"Throughput: {results['samples_per_second']:.1f} samples/sec")
        logger.info("=" * 50)
        
        # Print per-class metrics
        logger.info("Per-class IoU:")
        for class_name, iou in results['iou_per_class'].items():
            logger.info(f"  {class_name}: {iou:.4f}")
        
        logger.info("Per-class Dice:")
        for class_name, dice in results['dice_per_class'].items():
            logger.info(f"  {class_name}: {dice:.4f}")
        
        logger.info("=" * 50)
        logger.info(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
