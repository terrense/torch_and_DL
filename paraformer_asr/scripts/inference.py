#!/usr/bin/env python3
"""
Inference script for Paraformer ASR.

Provides command-line interface for running inference on audio features
with text output and optional FastAPI service mode.
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.inference import create_inference_from_checkpoint, run_fastapi_service
from utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def load_features_from_file(file_path: Path):
    """Load features from various file formats."""
    if file_path.suffix == '.npy':
        return np.load(file_path)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
            return np.array(data['features'])
    elif file_path.suffix == '.txt':
        # Assume space-separated values
        return np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def generate_sample_features(seq_length: int = 100, feature_dim: int = 80):
    """Generate sample features for testing."""
    # Create synthetic mel-spectrogram-like features
    features = np.random.randn(seq_length, feature_dim) * 0.5
    
    # Add some structure to make it more realistic
    for i in range(feature_dim):
        # Add some frequency-dependent patterns
        freq_pattern = np.sin(np.linspace(0, 4 * np.pi, seq_length)) * (i / feature_dim)
        features[:, i] += freq_pattern * 0.3
    
    # Add temporal smoothing
    for i in range(1, seq_length):
        features[i] = 0.7 * features[i] + 0.3 * features[i-1]
    
    return features


def main():
    parser = argparse.ArgumentParser(description='Run Paraformer ASR inference')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--features', type=Path, default=None,
                        help='Path to features file (.npy, .json, .txt)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output file for results')
    parser.add_argument('--device', type=str, default=None,
                        help='Inference device (cuda/cpu)')
    parser.add_argument('--max-length', type=int, default=200,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--do-sample', action='store_true',
                        help='Use sampling instead of greedy decoding')
    parser.add_argument('--return-alignment', action='store_true',
                        help='Return predictor alignment')
    parser.add_argument('--return-confidence', action='store_true',
                        help='Return confidence scores')
    parser.add_argument('--return-tokens', action='store_true',
                        help='Return token sequences')
    
    # Service mode options
    parser.add_argument('--service', action='store_true',
                        help='Run as FastAPI service')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Service host (service mode)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Service port (service mode)')
    
    # Testing options
    parser.add_argument('--generate-sample', action='store_true',
                        help='Generate sample features for testing')
    parser.add_argument('--sample-length', type=int, default=100,
                        help='Sample feature sequence length')
    parser.add_argument('--feature-dim', type=int, default=80,
                        help='Feature dimension')
    
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
    
    logger.info("Starting Paraformer ASR inference")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Service mode
    if args.service:
        logger.info(f"Starting FastAPI service on {args.host}:{args.port}")
        try:
            run_fastapi_service(
                checkpoint_path=args.checkpoint,
                host=args.host,
                port=args.port,
                device=device
            )
        except ImportError:
            logger.error("FastAPI is required for service mode. Install with: pip install fastapi uvicorn")
            return 1
        except Exception as e:
            logger.error(f"Service failed: {e}")
            return 1
        return 0
    
    try:
        # Create inference instance
        logger.info("Loading model from checkpoint...")
        inference = create_inference_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=device,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=args.do_sample
        )
        
        logger.info("Model loaded successfully")
        
        # Load or generate features
        if args.generate_sample:
            logger.info(f"Generating sample features ({args.sample_length}, {args.feature_dim})")
            features = generate_sample_features(args.sample_length, args.feature_dim)
            feature_length = args.sample_length
        elif args.features is not None:
            logger.info(f"Loading features from {args.features}")
            features = load_features_from_file(args.features)
            feature_length = features.shape[0]
            logger.info(f"Loaded features shape: {features.shape}")
        else:
            logger.error("Either --features or --generate-sample must be specified")
            return 1
        
        # Run inference
        logger.info("Running inference...")
        
        result = inference.infer_features(
            features=features,
            feature_length=feature_length,
            return_alignment=args.return_alignment,
            return_confidence=args.return_confidence,
            return_tokens=args.return_tokens
        )
        
        # Print results
        logger.info("Inference completed!")
        logger.info("=" * 50)
        logger.info("INFERENCE RESULTS")
        logger.info("=" * 50)
        logger.info(f"Predicted text: '{result['text']}'")
        
        if 'tokens' in result:
            logger.info(f"Token sequence: {result['tokens']}")
            logger.info(f"Token length: {result['token_length']}")
        
        if 'avg_confidence' in result:
            logger.info(f"Average confidence: {result['avg_confidence']:.4f}")
        
        if 'alignment' in result:
            logger.info(f"Alignment shape: {result['alignment'].shape}")
        
        logger.info("=" * 50)
        
        # Save results if output file specified
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            output_result = result.copy()
            if 'alignment' in output_result:
                output_result['alignment'] = output_result['alignment'].tolist()
            if 'confidence' in output_result:
                output_result['confidence'] = output_result['confidence'].tolist()
            
            with open(args.output, 'w') as f:
                json.dump(output_result, f, indent=2)
            
            logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())