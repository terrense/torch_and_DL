"""Comprehensive segmentation trainer with full feature set."""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
import logging

from ..losses import get_loss_function
from ..metrics import SegmentationMetrics
from ..utils.checkpoint import CheckpointManager
from ..utils.logging_utils import setup_logger, log_metrics


class SegmentationTrainer:
    """
    Comprehensive segmentation trainer with AdamW optimizer, schedulers,
    gradient clipping, mixed precision, checkpoint management, and metrics logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "runs"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Segmentation model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration dictionary
            device: Training device
            output_dir: Output directory for logs and checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            name="trainer",
            log_file=self.log_dir / "training.log",
            level=logging.INFO
        )
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_metrics()
        self._setup_mixed_precision()
        self._setup_checkpoint_manager()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.training_history = []
        
        # CSV logging
        self.csv_file = self.output_dir / "training_results.csv"
        self._setup_csv_logging()
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer."""
        optimizer_config = self.config.get('optimizer', {})
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config.get('learning_rate', 1e-3),
            weight_decay=optimizer_config.get('weight_decay', 1e-4),
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8)
        )
        
        self.logger.info(f"Initialized AdamW optimizer with lr={self.optimizer.param_groups[0]['lr']}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        else:
            self.scheduler = None
        
        if self.scheduler:
            self.logger.info(f"Initialized {scheduler_type} scheduler")
    
    def _setup_loss_function(self):
        """Setup loss function."""
        loss_config = self.config.get('loss', {})
        num_classes = self.config.get('num_classes', 3)
        
        self.criterion = get_loss_function(
            loss_type=loss_config.get('type', 'dice_bce'),
            num_classes=num_classes,
            **loss_config.get('params', {})
        )
        
        self.logger.info(f"Initialized loss function: {type(self.criterion).__name__}")
    
    def _setup_metrics(self):
        """Setup evaluation metrics."""
        num_classes = self.config.get('num_classes', 3)
        class_names = self.config.get('class_names', None)
        ignore_index = self.config.get('ignore_index', None)
        
        self.train_metrics = SegmentationMetrics(
            num_classes=num_classes,
            class_names=class_names,
            ignore_index=ignore_index
        )
        
        if self.val_loader:
            self.val_metrics = SegmentationMetrics(
                num_classes=num_classes,
                class_names=class_names,
                ignore_index=ignore_index
            )
        
        self.logger.info(f"Initialized metrics for {num_classes} classes")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        self.use_amp = self.config.get('mixed_precision', False)
        
        if self.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler()
            self.logger.info("Enabled mixed precision training")
        else:
            self.scaler = None
            if self.use_amp:
                self.logger.warning("Mixed precision requested but CUDA not available")
    
    def _setup_checkpoint_manager(self):
        """Setup checkpoint management."""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints=self.config.get('max_checkpoints', 5)
        )
    
    def _setup_csv_logging(self):
        """Setup CSV logging for training results."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = [
                    'epoch', 'train_loss', 'train_pixel_acc', 'train_mean_iou', 'train_mean_dice'
                ]
                if self.val_loader:
                    headers.extend([
                        'val_loss', 'val_pixel_acc', 'val_mean_iou', 'val_mean_dice'
                    ])
                headers.extend(['learning_rate', 'epoch_time'])
                writer.writerow(headers)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Gradient clipping setup
        max_grad_norm = self.config.get('gradient_clip', None)
        accumulate_grad_batches = self.config.get('accumulate_grad_batches', 1)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            if self.use_amp and self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss = loss / accumulate_grad_batches
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulate_grad_batches == 0:
                    if max_grad_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / accumulate_grad_batches
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulate_grad_batches == 0:
                    if max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            # Update metrics
            epoch_loss += loss.item() * accumulate_grad_batches
            self.train_metrics.update(outputs.detach(), targets)
            
            # Log progress
            if batch_idx % self.config.get('log_every', 10) == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item() * accumulate_grad_batches:.4f}"
                )
        
        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        metrics = self.train_metrics.compute()
        
        return {
            'loss': avg_loss,
            'pixel_accuracy': metrics['pixel_accuracy'],
            'mean_iou': metrics['mean_iou'],
            'mean_dice': metrics['mean_dice']
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        self.val_metrics.reset()
        
        epoch_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_amp and self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Update metrics
                epoch_loss += loss.item()
                self.val_metrics.update(outputs, targets)
        
        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        metrics = self.val_metrics.compute()
        
        return {
            'loss': avg_loss,
            'pixel_accuracy': metrics['pixel_accuracy'],
            'mean_iou': metrics['mean_iou'],
            'mean_dice': metrics['mean_dice']
        }
    
    def train(self, num_epochs: int) -> List[Dict[str, Any]]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history list
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_results = self.train_epoch()
            
            # Validation phase
            val_results = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric_to_monitor = val_results.get('mean_iou', train_results['mean_iou'])
                    self.scheduler.step(metric_to_monitor)
                else:
                    self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s")
            self.logger.info(f"Train - Loss: {train_results['loss']:.4f}, "
                           f"IoU: {train_results['mean_iou']:.4f}, "
                           f"Dice: {train_results['mean_dice']:.4f}")
            
            if val_results:
                self.logger.info(f"Val - Loss: {val_results['loss']:.4f}, "
                               f"IoU: {val_results['mean_iou']:.4f}, "
                               f"Dice: {val_results['mean_dice']:.4f}")
            
            # Save checkpoint
            val_metric = val_results.get('mean_iou', train_results['mean_iou'])
            is_best = val_metric > self.best_val_metric
            
            if is_best:
                self.best_val_metric = val_metric
            
            if (epoch + 1) % self.config.get('save_every', 20) == 0 or is_best:
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'best_val_metric': self.best_val_metric,
                    'config': self.config
                }
                
                self.checkpoint_manager.save_checkpoint(
                    checkpoint_data,
                    epoch + 1,
                    is_best=is_best
                )
            
            # Log to CSV
            self._log_to_csv(epoch + 1, train_results, val_results, current_lr, epoch_time)
            
            # Store history
            epoch_data = {
                'epoch': epoch + 1,
                'train': train_results,
                'val': val_results,
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            self.training_history.append(epoch_data)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation metric: {self.best_val_metric:.4f}")
        
        return self.training_history
    
    def _log_to_csv(
        self,
        epoch: int,
        train_results: Dict[str, float],
        val_results: Dict[str, float],
        learning_rate: float,
        epoch_time: float
    ):
        """Log epoch results to CSV file."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                epoch,
                train_results['loss'],
                train_results['pixel_accuracy'],
                train_results['mean_iou'],
                train_results['mean_dice']
            ]
            
            if val_results:
                row.extend([
                    val_results['loss'],
                    val_results['pixel_accuracy'],
                    val_results['mean_iou'],
                    val_results['mean_dice']
                ])
            else:
                row.extend([None, None, None, None])
            
            row.extend([learning_rate, epoch_time])
            writer.writerow(row)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint