#!/usr/bin/env python3
"""
Inference script for U-Net Transformer Segmentation.

Provides command-line interface for running inference on images
with visualization and result saving capabilities.
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import load_config
from models.registry import create_model_from_config
from evaluation.inference import SegmentationInference
from utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def generate_sample_image(image_size=(256, 256), num_shapes=3):
    """Generate a sample image with random shapes for testing."""
    from data.toy_shapes import ToyShapesDataset
    
    # Create temporary dataset to generate one sample
    dataset = ToyShapesDataset(
        num_samples=1,
        image_size=image_size,
        num_classes=num_shapes + 1,
        shapes=['circle', 'square', 'triangle'][:num_shapes],
        min_shapes=1,
        max_shapes=3
    )
    
    image, mask = dataset[0]
    
    # Convert tensor to PIL Image
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
    else:
        pil_image = image
    
    return pil_image


def main():
    parser = argparse.ArgumentParser(description='Run U-Net Transformer Segmentation inference')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=Path, default=None,
                        help='Path to input image')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output path for results')
    parser.add_argument('--device', type=str, default=None,
                        help='Inference device (cuda/cpu)')
    parser.add_argument('--config', type=Path, default=None,
                        help='Path to configuration file (optional)')
    
    # Visualization options
    parser.add_argument('--save-overlay', action='store_true',
                        help='Save prediction overlay visualization')
    parser.add_argument('--save-mask', action='store_true',
                        help='Save raw prediction mask')
    parser.add_argument('--save-probabilities', action='store_true',
                        help='Save class probability maps')
    parser.add_argument('--show-probabilities', action='store_true',
                        help='Show class probabilities in visualization')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Transparency for overlay (0-1)')
    
    # Testing options
    parser.add_argument('--generate-sample', action='store_true',
                        help='Generate sample image for testing')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
                        help='Image size for sample generation')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.output:
        log_file = args.output.parent / 'inference.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = None
    
    setup_logger(log_file, level=log_level)
    
    logger.info("Starting U-Net Transformer Segmentation inference")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
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
        
        # Get config
        if args.config is not None:
            config = load_config(args.config).__dict__
        elif 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = {
                'model': {'name': 'unet', 'in_channels': 3, 'num_classes': 3},
                'num_classes': 3,
                'data': {'image_size': [256, 256]}
            }
        
        # Create model
        model = create_model_from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")
        
        # Create inference pipeline
        num_classes = config.get('num_classes', 3)
        class_names = config.get('class_names', None)
        image_size = tuple(config.get('data', {}).get('image_size', [256, 256]))
        
        inference = SegmentationInference(
            model=model,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            image_size=image_size
        )
        
        # Load or generate image
        if args.generate_sample:
            logger.info(f"Generating sample image {tuple(args.image_size)}")
            image = generate_sample_image(
                image_size=tuple(args.image_size),
                num_shapes=num_classes - 1
            )
        elif args.image is not None:
            logger.info(f"Loading image from {args.image}")
            image = Image.open(args.image).convert('RGB')
        else:
            logger.error("Either --image or --generate-sample must be specified")
            return 1
        
        # Run inference
        logger.info("Running inference...")
        
        if args.show_probabilities or args.save_probabilities:
            predicted_mask, probabilities = inference.predict_image(
                image, return_probabilities=True
            )
        else:
            predicted_mask = inference.predict_image(image)
            probabilities = None
        
        # Print results
        logger.info("Inference completed!")
        logger.info("=" * 50)
        logger.info("INFERENCE RESULTS")
        logger.info("=" * 50)
        logger.info(f"Predicted mask shape: {predicted_mask.shape}")
        logger.info(f"Unique classes: {np.unique(predicted_mask)}")
        
        # Count pixels per class
        for class_idx in range(num_classes):
            count = np.sum(predicted_mask == class_idx)
            percentage = (count / predicted_mask.size) * 100
            class_name = class_names[class_idx] if class_names else f"class_{class_idx}"
            logger.info(f"{class_name}: {count} pixels ({percentage:.2f}%)")
        
        logger.info("=" * 50)
        
        # Save results if output specified
        if args.output is not None:
            logger.info(f"Saving results to {args.output}")
            
            inference.save_prediction(
                image=image,
                output_path=str(args.output),
                save_overlay=args.save_overlay or True,  # Default to True
                save_mask=args.save_mask or True,  # Default to True
                save_probabilities=args.save_probabilities
            )
            
            logger.info("Results saved successfully")
        
        # Show visualization if requested
        if args.show_probabilities:
            import matplotlib.pyplot as plt
            fig = inference.visualize_prediction(
                image=image,
                show_probabilities=True,
                alpha=args.alpha
            )
            plt.show()
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
