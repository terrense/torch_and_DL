"""
ASR Trainer class for Paraformer training.

Provides high-level training interface with configuration-based setup,
automatic device management, and integrated logging and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ExponentialLR,
    OneCycleLR,
    ReduceLROnPlateau
)
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

from ..models.paraformer import ParaformerASR, create_paraformer_from_config
from ..losses.seq_loss import CombinedASRLoss, create_sequence_loss
from ..data.tokenizer import CharTokenizer
from ..utils.checkpoint import CheckpointManager
from ..utils.logging_utils import setup_logger
from ..utils.reproducibility import set_seed, set_deterministic
from .train_loop import run_training

logger = logging.getLogger(__name__)


class ASRTrainer:
    """
    High-level trainer for Paraformer ASR models.
    
    Handles model creation, optimizer setup, data loading,
    and training orchestration with comprehensive logging.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        device: Optional[torch.device] = None,
        resume_from: Optional[Path] = None
    ):
        """
        Initialize ASR trainer.
        
        Args:
            config: Training configuration dictionary
            output_dir: Output directory for logs and checkpoints
            device: Training device (auto-detect if None)
            resume_from: Path to checkpoint to resume from (optional)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Setup reproducibility
        if config.get('experiment', {}).get('seed') is not None:
            set_seed(config['experiment']['seed'])
        
        if config.get('experiment', {}).get('deterministic', False):
            set_deterministic(True)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.checkpoint_manager = None
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup logging
        log_file = self.output_dir / 'training.log'
        setup_logger(log_file, level=logging.INFO)
        
        # Save configuration
        config_file = self.output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Resume from checkpoint if specified
        if resume_from is not None:
            self._resume_from_checkpoint(resume_from)
    
    def setup_model(self, tokenizer: CharTokenizer) -> ParaformerASR:
        """
        Setup model from configuration.
        
        Args:
            tokenizer: Character tokenizer for vocabulary size
            
        Returns:
            model: Configured Paraformer model
        """
        model_config = self.config.get('model', {})
        
        # Update vocab size from tokenizer
        model_config['vocab_size'] = len(tokenizer)
        model_config['input_dim'] = model_config.get('input_dim', 80)
        
        # Create model
        self.model = create_paraformer_from_config(model_config)
        self.model.to(self.device)
        
        # Log model info
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {self.model.__class__.__name__}")
        logger.info(f"Total parameters: {num_params:,}")
        logger.info(f"Trainable parameters: {num_trainable:,}")
        
        return self.model
    
    def setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup optimizer from configuration.
        
        Returns:
            optimizer: Configured optimizer
        """
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")
        
        train_config = self.config.get('training', {})
        
        optimizer_name = train_config.get('optimizer', 'adamw').lower()
        lr = train_config.get('learning_rate', 1e-4)
        weight_decay = train_config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'sgd':
            momentum = train_config.get('momentum', 0.9)
            self.optimizer = SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        logger.info(f"Optimizer: {optimizer_name} (lr={lr}, weight_decay={weight_decay})")
        
        return self.optimizer
    
    def setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Setup learning rate scheduler from configuration.
        
        Returns:
            scheduler: Configured scheduler or None
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be setup before scheduler")
        
        train_config = self.config.get('training', {})
        scheduler_name = train_config.get('scheduler', 'none').lower()
        
        if scheduler_name == 'none' or scheduler_name is None:
            self.scheduler = None
        elif scheduler_name == 'cosine':
            num_epochs = train_config.get('num_epochs', 100)
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=train_config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'step':
            step_size = train_config.get('step_size', 30)
            gamma = train_config.get('gamma', 0.1)
            self.scheduler = StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_name == 'exponential':
            gamma = train_config.get('gamma', 0.95)
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
        elif scheduler_name == 'onecycle':
            num_epochs = train_config.get('num_epochs', 100)
            steps_per_epoch = train_config.get('steps_per_epoch', 100)
            max_lr = train_config.get('learning_rate', 1e-4)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=num_epochs,
                steps_per_epoch=steps_per_epoch
            )
        elif scheduler_name == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=train_config.get('factor', 0.5),
                patience=train_config.get('patience', 10),
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        if self.scheduler is not None:
            logger.info(f"Scheduler: {scheduler_name}")
        
        return self.scheduler
    
    def setup_loss_function(self, tokenizer: CharTokenizer) -> CombinedASRLoss:
        """
        Setup loss function from configuration.
        
        Args:
            tokenizer: Character tokenizer for special tokens
            
        Returns:
            loss_fn: Configured loss function
        """
        train_config = self.config.get('training', {})
        
        self.loss_fn = create_sequence_loss(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            label_smoothing=train_config.get('label_smoothing', 0.0),
            predictor_loss_weight=train_config.get('predictor_loss_weight', 0.1),
            predictor_type='boundary'
        )
        
        logger.info(f"Loss function: CombinedASRLoss")
        logger.info(f"Label smoothing: {train_config.get('label_smoothing', 0.0)}")
        logger.info(f"Predictor loss weight: {train_config.get('predictor_loss_weight', 0.1)}")
        
        return self.loss_fn
    
    def setup_checkpoint_manager(self) -> CheckpointManager:
        """
        Setup checkpoint manager.
        
        Returns:
            checkpoint_manager: Configured checkpoint manager
        """
        checkpoint_dir = self.output_dir / 'checkpoints'
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=self.config.get('experiment', {}).get('max_checkpoints', 5),
            save_best=True
        )
        
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        
        return self.checkpoint_manager
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[CharTokenizer] = None
    ) -> Dict[str, Any]:
        """
        Run complete training process.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            tokenizer: Character tokenizer (optional, will create default if None)
            
        Returns:
            Dictionary with training results and history
        """
        # Setup tokenizer if not provided
        if tokenizer is None:
            from ..data.tokenizer import create_default_tokenizer
            vocab_size = self.config.get('model', {}).get('vocab_size', 1000)
            tokenizer = create_default_tokenizer(vocab_size)
            logger.info(f"Created default tokenizer with vocab_size={len(tokenizer)}")
        
        self.tokenizer = tokenizer
        
        # Setup all components
        logger.info("Setting up training components...")
        
        self.setup_model(tokenizer)
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_loss_function(tokenizer)
        self.setup_checkpoint_manager()
        
        # Training configuration
        train_config = self.config.get('training', {})
        experiment_config = self.config.get('experiment', {})
        
        num_epochs = train_config.get('num_epochs', 100)
        gradient_clip = train_config.get('gradient_clip')
        log_interval = experiment_config.get('log_every', 10)
        eval_interval = experiment_config.get('eval_every', 1)
        early_stopping_patience = train_config.get('early_stopping_patience')
        
        # Results file
        results_file = self.output_dir / 'training_results.csv'
        
        # Run training
        logger.info("Starting training...")
        
        history = run_training(
            model=self.model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            tokenizer=tokenizer,
            num_epochs=num_epochs,
            device=self.device,
            checkpoint_manager=self.checkpoint_manager,
            log_interval=log_interval,
            eval_interval=eval_interval,
            gradient_clip=gradient_clip,
            early_stopping_patience=early_stopping_patience,
            results_file=results_file
        )
        
        # Save final results
        results = {
            'config': self.config,
            'history': history,
            'final_metrics': {
                'best_val_loss': min(h for h in history['val_loss'] if h is not None) if any(h is not None for h in history['val_loss']) else None,
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                'final_train_accuracy': history['train_token_accuracy'][-1] if history['train_token_accuracy'] else None
            }
        }
        
        results_json = self.output_dir / 'final_results.json'
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed! Results saved to {self.output_dir}")
        
        return results
    
    def _resume_from_checkpoint(self, checkpoint_path: Path):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Update start epoch
        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Resuming from epoch {self.start_epoch}")
    
    def save_model(self, path: Optional[Path] = None):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        if path is None:
            path = self.output_dir / 'final_model.pt'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config(),
            'tokenizer_vocab': self.tokenizer.vocab if self.tokenizer else None,
            'config': self.config
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path, tokenizer: Optional[CharTokenizer] = None):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Setup tokenizer if not provided
        if tokenizer is None and 'tokenizer_vocab' in checkpoint:
            from ..data.tokenizer import CharTokenizer
            tokenizer = CharTokenizer(vocab_size=len(checkpoint['tokenizer_vocab']))
            tokenizer.vocab = checkpoint['tokenizer_vocab']
            tokenizer.char_to_id = {char: i for i, char in enumerate(tokenizer.vocab)}
            tokenizer.id_to_char = {i: char for i, char in enumerate(tokenizer.vocab)}
        
        # Setup model
        if self.model is None:
            self.setup_model(tokenizer)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {path}")


def create_trainer_from_config(
    config: Dict[str, Any],
    output_dir: Path,
    device: Optional[torch.device] = None,
    resume_from: Optional[Path] = None
) -> ASRTrainer:
    """
    Create trainer from configuration dictionary.
    
    Args:
        config: Training configuration
        output_dir: Output directory
        device: Training device (optional)
        resume_from: Checkpoint to resume from (optional)
        
    Returns:
        trainer: Configured ASRTrainer instance
    """
    return ASRTrainer(
        config=config,
        output_dir=output_dir,
        device=device,
        resume_from=resume_from
    )


if __name__ == "__main__":
    # Test trainer creation
    print("Testing ASR trainer...")
    
    # Example configuration
    config = {
        'model': {
            'input_dim': 80,
            'hidden_dim': 256,
            'vocab_size': 1000,
            'encoder_layers': 4,
            'decoder_layers': 2
        },
        'training': {
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine'
        },
        'experiment': {
            'seed': 42,
            'log_every': 5
        }
    }
    
    # Create trainer
    output_dir = Path('test_output')
    trainer = create_trainer_from_config(config, output_dir)
    
    print("Trainer created successfully!")
    print(f"Output directory: {trainer.output_dir}")
    print(f"Device: {trainer.device}")
    
    # Cleanup
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print("Trainer test completed!")