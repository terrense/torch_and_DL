#!/usr/bin/env python3
"""
Evaluation script for Paraformer ASR.

Provides command-line interface for evaluating trained ASR models
with comprehensive metrics including WER, CER, and token accuracy.
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
from data.toy_seq2seq import ToySeq2SeqDataset
from data.tokenizer import create_default_tokenizer, CharTokenizer
from data.utils import collate_seq2seq_batch
from evaluation.evaluator import evaluate_model
from evaluation.inference import create_inference_from_checkpoint
from losses.seq_loss import create_sequence_loss
from utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def create_test_dataloader(config, tokenizer):
    """Create test data loader."""
    data_config = config.get('data', {})
    
    # Create test dataset
    test_dataset = ToySeq2SeqDataset(
        vocab_size=len(tokenizer),
        num_samples=data_config.get('test_samples', 500),
        min_seq_len=data_config.get('min_seq_len', 10),
        max_seq_len=data_config.get('max_seq_len', 100),
        feature_dim=config.get('model', {}).get('input_dim', 80),
        feature_noise=data_config.get('feature_noise', 0.1),
        correlation_strength=data_config.get('correlation_strength', 0.8),
        difficulty_level=data_config.get('difficulty_level', 0.5),
        tokenizer=tokenizer
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.get('batch_size', 16),
        shuffle=False,
        num_workers=data_config.get('num_workers', 0),
        collate_fn=collate_seq2seq_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    return test_loader


def main():
    parser = argparse.ArgumentParser(description='Evaluate Paraformer ASR model')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=Path, default=None,
                        help='Path to configuration file (optional)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Evaluation device (cuda/cpu)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to evaluate')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save individual predictions to file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(args.output_dir / 'evaluate.log', level=log_level)
    
    logger.info("Starting Paraformer ASR evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    try:
        # Create inference instance from checkpoint
        logger.info("Loading model from checkpoint...")
        inference = create_inference_from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=device
        )
        
        model = inference.model
        tokenizer = inference.tokenizer
        
        logger.info(f"Model loaded with vocab_size={len(tokenizer)}")
        
        # Load configuration if provided
        config = {}
        if args.config is not None:
            config = load_config(args.config).__dict__
            logger.info("Configuration loaded")
        else:
            # Use default configuration
            config = {
                'model': {'input_dim': 80, 'vocab_size': len(tokenizer)},
                'data': {'batch_size': 16, 'test_samples': 500}
            }
            logger.info("Using default configuration")
        
        # Create test data loader
        logger.info("Creating test data loader...")
        test_loader = create_test_dataloader(config, tokenizer)
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Create loss function for evaluation
        loss_fn = create_sequence_loss(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            label_smoothing=0.0,
            predictor_loss_weight=0.1
        )
        
        # Run evaluation
        logger.info("Starting evaluation...")
        
        output_file = None
        if args.save_predictions:
            output_file = args.output_dir / 'predictions.json'
        
        results = evaluate_model(
            model=model,
            dataloader=test_loader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            device=device,
            max_batches=args.max_batches,
            output_file=output_file
        )
        
        # Save results
        results_file = args.output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        logger.info("Evaluation completed!")
        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        
        if 'num_samples' in results:
            logger.info(f"Samples evaluated: {results['num_samples']}")
        
        if 'avg_loss' in results and results['avg_loss'] is not None:
            logger.info(f"Average loss: {results['avg_loss']:.4f}")
        
        if 'avg_token_accuracy' in results and results['avg_token_accuracy'] is not None:
            logger.info(f"Token accuracy: {results['avg_token_accuracy']:.4f}")
        
        if 'sequence_accuracy' in results:
            logger.info(f"Sequence accuracy: {results['sequence_accuracy']:.4f}")
        
        if 'wer' in results:
            logger.info(f"Word Error Rate (WER): {results['wer']:.4f}")
        
        if 'cer' in results:
            logger.info(f"Character Error Rate (CER): {results['cer']:.4f}")
        
        if 'overall_wer' in results:
            logger.info(f"Overall WER: {results['overall_wer']:.4f}")
        
        if 'overall_cer' in results:
            logger.info(f"Overall CER: {results['overall_cer']:.4f}")
        
        if 'avg_confidence' in results:
            logger.info(f"Average confidence: {results['avg_confidence']:.4f}")
        
        if 'evaluation_time' in results:
            logger.info(f"Evaluation time: {results['evaluation_time']:.1f}s")
        
        logger.info("=" * 50)
        logger.info(f"Results saved to: {results_file}")
        
        if args.save_predictions and output_file:
            logger.info(f"Predictions saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())