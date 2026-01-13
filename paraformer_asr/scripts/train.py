#!/usr/bin/env python3
"""
Training script for Paraformer ASR.

Provides command-line interface for training ASR models with
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
from data.toy_seq2seq import ToySeq2SeqDataset
from data.tokenizer import create_default_tokenizer
from data.utils import collate_seq2seq_batch
from training.trainer import create_trainer_from_config
from utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def create_dataloaders(config, tokenizer):
    """Create training and validation data loaders."""
    data_config = config.get('data', {})
    
    # Create datasets
    train_dataset = ToySeq2SeqDataset(
        vocab_size=len(tokenizer),
        num_samples=data_config.get('train_samples', 1000),
        min_seq_len=data_config.get('min_seq_len', 10),
        max_seq_len=data_config.get('max_seq_len', 100),
        feature_dim=config.get('model', {}).get('input_dim', 80),
        feature_noise=data_config.get('feature_noise', 0.1),
        correlation_strength=data_config.get('correlation_strength', 0.8),
        difficulty_level=data_config.get('difficulty_level', 0.5),
        tokenizer=tokenizer
    )
    
    val_dataset = ToySeq2SeqDataset(
        vocab_size=len(tokenizer),
        num_samples=data_config.get('val_samples', 200),
        min_seq_len=data_config.get('min_seq_len', 10),
        max_seq_len=data_config.get('max_seq_len', 100),
        feature_dim=config.get('model', {}).get('input_dim', 80),
        feature_noise=data_config.get('feature_noise', 0.1),
        correlation_strength=data_config.get('correlation_strength', 0.8),
        difficulty_level=data_config.get('difficulty_level', 0.5),
        tokenizer=tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 16),
        shuffle=True,
        num_workers=data_config.get('num_workers', 0),
        collate_fn=collate_seq2seq_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 16),
        shuffle=False,
        num_workers=data_config.get('num_workers', 0),
        collate_fn=collate_seq2seq_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train Paraformer ASR model')
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Training device (cuda/cpu)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logger(args.output_dir / 'train.log', level=log_level)
    
    logger.info("Starting Paraformer ASR training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Setup device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    try:
        # Create tokenizer
        vocab_size = config.model.vocab_size
        tokenizer = create_default_tokenizer(vocab_size)
        logger.info(f"Created tokenizer with vocab_size={len(tokenizer)}")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(config.__dict__, tokenizer)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer_from_config(
            config=config.__dict__,
            output_dir=args.output_dir,
            device=device,
            resume_from=args.resume
        )
        
        # Run training
        logger.info("Starting training...")
        results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            tokenizer=tokenizer
        )
        
        # Save final model
        trainer.save_model()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final results: {results['final_metrics']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())