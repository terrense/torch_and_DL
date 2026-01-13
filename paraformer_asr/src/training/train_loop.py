"""
Training loop implementation for Paraformer ASR.

Implements training loop with optimizer, scheduler, gradient clipping,
token accuracy calculation, sequence-level metrics, and comprehensive
logging and checkpoint management.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import csv

from ..losses.seq_loss import CombinedASRLoss, compute_token_accuracy
from ..decode.greedy import greedy_decode
from ..data.tokenizer import CharTokenizer
from ..utils.tensor_utils import assert_shape, check_nan_inf
from ..utils.checkpoint import CheckpointManager
from ..utils.logging_utils import setup_logger, log_metrics

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: CombinedASRLoss,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    gradient_clip: Optional[float] = None,
    device: torch.device = torch.device('cpu'),
    log_interval: int = 10,
    tokenizer: Optional[CharTokenizer] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Paraformer ASR model
        dataloader: Training data loader
        loss_fn: Combined ASR loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        gradient_clip: Gradient clipping value (optional)
        device: Training device
        log_interval: Logging interval in steps
        tokenizer: Tokenizer for text decoding (optional)
        epoch: Current epoch number
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    # Initialize metrics
    total_loss = 0.0
    total_decoder_loss = 0.0
    total_predictor_loss = 0.0
    total_token_accuracy = 0.0
    num_batches = 0
    num_tokens = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        features = batch['features'].to(device)  # [B, T, F]
        feature_lengths = batch['feature_lengths'].to(device)  # [B]
        tokens = batch['tokens'].to(device)  # [B, S]
        token_lengths = batch['token_lengths'].to(device)  # [B]
        
        # Optional predictor targets
        predictor_targets = batch.get('predictor_targets')
        if predictor_targets is not None:
            predictor_targets = predictor_targets.to(device)
        
        # Input validation
        assert_shape(features, "B,T,F", "features")
        assert_shape(tokens, "B,S", "tokens")
        check_nan_inf(features, "features")
        
        B, T, F = features.shape
        S = tokens.shape[1]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        try:
            outputs = model(
                features=features,
                feature_lengths=feature_lengths,
                target_tokens=tokens,
                target_lengths=token_lengths,
                return_predictor_output=True
            )
            
            # Compute loss
            loss_dict = loss_fn(
                decoder_logits=outputs['logits'],
                target_tokens=tokens,
                target_lengths=token_lengths,
                predictor_predictions=outputs.get('predictor_predictions'),
                predictor_targets=predictor_targets,
                feature_mask=~outputs['padding_mask']  # Convert to valid mask
            )
            
            total_loss_val = loss_dict['total_loss']
            
            # Check for NaN/Inf in loss
            if not torch.isfinite(total_loss_val):
                logger.warning(f"Non-finite loss detected at batch {batch_idx}: {total_loss_val}")
                continue
            
            # Backward pass
            total_loss_val.backward()
            
            # Gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Optimizer step
            optimizer.step()
            
            # Scheduler step (if per-step scheduler)
            if scheduler is not None and hasattr(scheduler, 'step_update'):
                scheduler.step_update(epoch * len(dataloader) + batch_idx)
            
            # Update metrics
            total_loss += total_loss_val.item()
            total_decoder_loss += loss_dict['decoder_loss'].item()
            if 'predictor_loss' in loss_dict:
                total_predictor_loss += loss_dict['predictor_loss'].item()
            total_token_accuracy += loss_dict['token_accuracy'].item()
            
            # Count tokens for perplexity calculation
            valid_tokens = (tokens != loss_fn.decoder_loss.ignore_index).sum().item()
            num_tokens += valid_tokens
            num_batches += 1
            
            # Logging
            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx:4d}/{len(dataloader)} | "
                    f"Loss: {total_loss_val.item():.4f} | "
                    f"Acc: {loss_dict['token_accuracy'].item():.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
                
                # Sample text generation (occasionally)
                if batch_idx % (log_interval * 5) == 0 and tokenizer is not None:
                    try:
                        with torch.no_grad():
                            # Take first sample from batch
                            sample_logits = outputs['logits'][:1]  # [1, S, V]
                            sample_tokens = tokens[:1]  # [1, S]
                            sample_lengths = token_lengths[:1]  # [1]
                            
                            # Decode prediction and target
                            pred_texts = greedy_decode(
                                sample_logits, tokenizer, sample_lengths
                            )
                            target_texts = tokenizer.batch_decode(
                                sample_tokens, sample_lengths
                            )
                            
                            logger.info(f"Sample - Target: '{target_texts[0]}'")
                            logger.info(f"Sample - Prediction: '{pred_texts[0]}'")
                    except Exception as e:
                        logger.debug(f"Sample generation failed: {e}")
        
        except Exception as e:
            logger.error(f"Error in training step {batch_idx}: {e}")
            continue
    
    # Per-epoch scheduler step
    if scheduler is not None and not hasattr(scheduler, 'step_update'):
        scheduler.step()
    
    # Calculate average metrics
    if num_batches > 0:
        avg_metrics = {
            'train_loss': total_loss / num_batches,
            'train_decoder_loss': total_decoder_loss / num_batches,
            'train_predictor_loss': total_predictor_loss / num_batches,
            'train_token_accuracy': total_token_accuracy / num_batches,
            'train_perplexity': torch.exp(torch.tensor(total_decoder_loss / num_batches)).item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'num_tokens': num_tokens,
            'epoch_time': time.time() - start_time
        }
    else:
        avg_metrics = {
            'train_loss': float('inf'),
            'train_decoder_loss': float('inf'),
            'train_predictor_loss': 0.0,
            'train_token_accuracy': 0.0,
            'train_perplexity': float('inf'),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'num_tokens': 0,
            'epoch_time': time.time() - start_time
        }
    
    return avg_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: CombinedASRLoss,
    device: torch.device = torch.device('cpu'),
    tokenizer: Optional[CharTokenizer] = None,
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: Paraformer ASR model
        dataloader: Validation data loader
        loss_fn: Combined ASR loss function
        device: Validation device
        tokenizer: Tokenizer for text decoding (optional)
        max_samples: Maximum number of samples to evaluate (optional)
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    # Initialize metrics
    total_loss = 0.0
    total_decoder_loss = 0.0
    total_predictor_loss = 0.0
    total_token_accuracy = 0.0
    num_batches = 0
    num_tokens = 0
    
    # For sequence-level metrics
    total_sequence_accuracy = 0.0
    num_sequences = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Early stopping for validation
            if max_samples is not None and num_sequences >= max_samples:
                break
            
            # Move batch to device
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths'].to(device)
            tokens = batch['tokens'].to(device)
            token_lengths = batch['token_lengths'].to(device)
            
            predictor_targets = batch.get('predictor_targets')
            if predictor_targets is not None:
                predictor_targets = predictor_targets.to(device)
            
            B, T, F = features.shape
            
            try:
                # Forward pass
                outputs = model(
                    features=features,
                    feature_lengths=feature_lengths,
                    target_tokens=tokens,
                    target_lengths=token_lengths,
                    return_predictor_output=True
                )
                
                # Compute loss
                loss_dict = loss_fn(
                    decoder_logits=outputs['logits'],
                    target_tokens=tokens,
                    target_lengths=token_lengths,
                    predictor_predictions=outputs.get('predictor_predictions'),
                    predictor_targets=predictor_targets,
                    feature_mask=~outputs['padding_mask']
                )
                
                # Update metrics
                total_loss += loss_dict['total_loss'].item()
                total_decoder_loss += loss_dict['decoder_loss'].item()
                if 'predictor_loss' in loss_dict:
                    total_predictor_loss += loss_dict['predictor_loss'].item()
                total_token_accuracy += loss_dict['token_accuracy'].item()
                
                # Sequence-level accuracy
                if tokenizer is not None:
                    try:
                        # Generate predictions
                        pred_texts = greedy_decode(
                            outputs['logits'], tokenizer, token_lengths
                        )
                        target_texts = tokenizer.batch_decode(tokens, token_lengths)
                        
                        # Count exact matches
                        for pred, target in zip(pred_texts, target_texts):
                            if pred.strip() == target.strip():
                                total_sequence_accuracy += 1
                        
                        num_sequences += len(pred_texts)
                    except Exception as e:
                        logger.debug(f"Sequence accuracy calculation failed: {e}")
                        num_sequences += B
                else:
                    num_sequences += B
                
                # Count tokens
                valid_tokens = (tokens != loss_fn.decoder_loss.ignore_index).sum().item()
                num_tokens += valid_tokens
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error in validation step {batch_idx}: {e}")
                continue
    
    # Calculate average metrics
    if num_batches > 0:
        avg_metrics = {
            'val_loss': total_loss / num_batches,
            'val_decoder_loss': total_decoder_loss / num_batches,
            'val_predictor_loss': total_predictor_loss / num_batches,
            'val_token_accuracy': total_token_accuracy / num_batches,
            'val_perplexity': torch.exp(torch.tensor(total_decoder_loss / num_batches)).item(),
            'val_sequence_accuracy': total_sequence_accuracy / max(num_sequences, 1),
            'num_val_tokens': num_tokens,
            'val_time': time.time() - start_time
        }
    else:
        avg_metrics = {
            'val_loss': float('inf'),
            'val_decoder_loss': float('inf'),
            'val_predictor_loss': 0.0,
            'val_token_accuracy': 0.0,
            'val_perplexity': float('inf'),
            'val_sequence_accuracy': 0.0,
            'num_val_tokens': 0,
            'val_time': time.time() - start_time
        }
    
    return avg_metrics


def run_training(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    loss_fn: CombinedASRLoss,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    tokenizer: CharTokenizer,
    num_epochs: int,
    device: torch.device,
    checkpoint_manager: CheckpointManager,
    log_interval: int = 10,
    eval_interval: int = 1,
    gradient_clip: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    results_file: Optional[Path] = None
) -> Dict[str, List[float]]:
    """
    Run complete training loop with logging and checkpointing.
    
    Args:
        model: Paraformer ASR model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
        loss_fn: Combined ASR loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        tokenizer: Character tokenizer
        num_epochs: Number of training epochs
        device: Training device
        checkpoint_manager: Checkpoint manager
        log_interval: Logging interval in steps
        eval_interval: Evaluation interval in epochs
        gradient_clip: Gradient clipping value (optional)
        early_stopping_patience: Early stopping patience (optional)
        results_file: Path to save training results CSV (optional)
        
    Returns:
        Dictionary with training history
    """
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_token_accuracy': [],
        'val_loss': [],
        'val_token_accuracy': [],
        'val_sequence_accuracy': [],
        'learning_rate': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Results CSV file
    csv_file = None
    csv_writer = None
    if results_file is not None:
        results_file.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(results_file, 'w', newline='')
        fieldnames = [
            'epoch', 'train_loss', 'train_decoder_loss', 'train_predictor_loss',
            'train_token_accuracy', 'train_perplexity', 'val_loss', 'val_decoder_loss',
            'val_predictor_loss', 'val_token_accuracy', 'val_perplexity',
            'val_sequence_accuracy', 'learning_rate', 'epoch_time'
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
    
    try:
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = train_epoch(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_clip=gradient_clip,
                device=device,
                log_interval=log_interval,
                tokenizer=tokenizer,
                epoch=epoch
            )
            
            # Validation
            val_metrics = {}
            if val_dataloader is not None and (epoch + 1) % eval_interval == 0:
                val_metrics = validate_epoch(
                    model=model,
                    dataloader=val_dataloader,
                    loss_fn=loss_fn,
                    device=device,
                    tokenizer=tokenizer
                )
                
                logger.info(
                    f"Validation - Loss: {val_metrics['val_loss']:.4f} | "
                    f"Token Acc: {val_metrics['val_token_accuracy']:.4f} | "
                    f"Seq Acc: {val_metrics['val_sequence_accuracy']:.4f}"
                )
            
            # Update history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_token_accuracy'].append(train_metrics['train_token_accuracy'])
            history['learning_rate'].append(train_metrics['learning_rate'])
            
            if val_metrics:
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_token_accuracy'].append(val_metrics['val_token_accuracy'])
                history['val_sequence_accuracy'].append(val_metrics['val_sequence_accuracy'])
            else:
                history['val_loss'].append(None)
                history['val_token_accuracy'].append(None)
                history['val_sequence_accuracy'].append(None)
            
            # Save to CSV
            if csv_writer is not None:
                row = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
                csv_writer.writerow(row)
                csv_file.flush()
            
            # Checkpointing
            is_best = False
            if val_metrics and val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                is_best = True
                patience_counter = 0
            elif val_metrics:
                patience_counter += 1
            
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics={**train_metrics, **val_metrics},
                is_best=is_best
            )
            
            # Early stopping
            if (early_stopping_patience is not None and 
                patience_counter >= early_stopping_patience):
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            logger.info(
                f"Epoch {epoch + 1} completed - "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_token_accuracy']:.4f}"
            )
    
    finally:
        if csv_file is not None:
            csv_file.close()
    
    logger.info("Training completed!")
    return history


if __name__ == "__main__":
    # Test training functions
    print("Testing training loop functions...")
    
    # This would require a full model setup, so we'll just test imports
    print("Training loop imports successful!")