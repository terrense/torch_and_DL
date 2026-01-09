"""Checkpoint management utilities for model saving and loading."""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import torch
import torch.nn as nn


class CheckpointManager:
    """Comprehensive checkpoint management with metadata and versioning."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        monitor_metric: str = "val_loss",
        mode: str = "min"
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best checkpoint separately
            monitor_metric: Metric to monitor for best checkpoint
            mode: 'min' or 'max' for best checkpoint selection
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # Track best metric value
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
        
        # Track saved checkpoints
        self.checkpoint_history: List[Dict[str, Any]] = []
        
        # Load existing checkpoint history if available
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self) -> None:
        """Load checkpoint history from metadata file."""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    data = json.load(f)
                    self.checkpoint_history = data.get('checkpoints', [])
                    self.best_metric = data.get('best_metric', self.best_metric)
                    self.best_epoch = data.get('best_epoch', self.best_epoch)
            except (json.JSONDecodeError, KeyError):
                self.checkpoint_history = []
    
    def _save_checkpoint_history(self) -> None:
        """Save checkpoint history to metadata file."""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        data = {
            'checkpoints': self.checkpoint_history,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
        }
        
        with open(history_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            epoch: Current epoch number
            metrics: Current metrics dictionary
            scheduler: Learning rate scheduler (optional)
            extra_state: Additional state to save (optional)
            filename: Custom filename (optional)
            
        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().isoformat()
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp,
        }
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint_data['extra_state'] = extra_state
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update checkpoint history
        checkpoint_info = {
            'filename': filename,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': timestamp,
            'path': str(checkpoint_path),
        }
        
        self.checkpoint_history.append(checkpoint_info)
        
        # Check if this is the best checkpoint
        is_best = False
        if self.monitor_metric in metrics:
            current_metric = metrics[self.monitor_metric]
            
            if self.mode == 'min':
                is_best = current_metric < self.best_metric
            else:
                is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                
                if self.save_best:
                    best_path = self.checkpoint_dir / "best_checkpoint.pth"
                    shutil.copy2(checkpoint_path, best_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save updated history
        self._save_checkpoint_history()
        
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by epoch (oldest first)
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['epoch'])
        
        # Remove oldest checkpoints
        to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Remove from history
            self.checkpoint_history.remove(checkpoint_info)
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = False,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: PyTorch model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            checkpoint_path: Path to specific checkpoint (optional)
            load_best: Whether to load best checkpoint (optional)
            map_location: Device to map tensors to (optional)
            
        Returns:
            Dictionary containing loaded checkpoint metadata
        """
        # Determine checkpoint path
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pth"
        elif checkpoint_path is None:
            # Load latest checkpoint
            if not self.checkpoint_history:
                raise FileNotFoundError("No checkpoints found")
            
            latest_checkpoint = max(self.checkpoint_history, key=lambda x: x['epoch'])
            checkpoint_path = Path(latest_checkpoint['path'])
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint_data.get('epoch', 0),
            'metrics': checkpoint_data.get('metrics', {}),
            'timestamp': checkpoint_data.get('timestamp', ''),
            'extra_state': checkpoint_data.get('extra_state', {}),
        }
    
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self.checkpoint_history:
            return None
        
        latest_checkpoint = max(self.checkpoint_history, key=lambda x: x['epoch'])
        return Path(latest_checkpoint['path'])
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        return best_path if best_path.exists() else None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.checkpoint_history.copy()
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best checkpoint information."""
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
        }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_path: Union[str, Path],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    extra_state: Optional[Dict[str, Any]] = None
) -> None:
    """
    Simple checkpoint saving function.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        metrics: Current metrics dictionary
        checkpoint_path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        extra_state: Additional state to save (optional)
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    if extra_state is not None:
        checkpoint_data['extra_state'] = extra_state
    
    torch.save(checkpoint_data, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Simple checkpoint loading function.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        map_location: Device to map tensors to (optional)
        
    Returns:
        Dictionary containing loaded checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
    
    # Load model state
    model.load_state_dict(checkpoint_data['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint_data.get('epoch', 0),
        'metrics': checkpoint_data.get('metrics', {}),
        'timestamp': checkpoint_data.get('timestamp', ''),
        'extra_state': checkpoint_data.get('extra_state', {}),
    }