"""
Smoke tests for Paraformer ASR end-to-end training.

Tests that run 30-100 training steps on toy data to verify:
- Loss decrease during training
- Proper convergence behavior  
- Checkpoint save/load functionality
- Reproducibility
"""
import pytest
import torch
import tempfile
import os
from pathlib import Path
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.toy_seq2seq import ToySeq2SeqDataset
from src.data.tokenizer import CharTokenizer
from src.data.utils import collate_seq2seq_batch
from src.models.paraformer import create_paraformer_from_config
from src.losses.seq_loss import create_sequence_loss
from src.training.train_loop import train_epoch, validate_epoch
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import CheckpointManager


@pytest.fixture
def toy_dataset():
    """Create a small toy dataset for smoke testing."""
    return ToySeq2SeqDataset(
        num_samples=40,
        vocab_size=50,
        max_token_len=20,
        feature_dim=40,
        max_feat_len=80,
        complexity='simple'
    )


@pytest.fixture
def tokenizer():
    """Create a simple tokenizer for testing."""
    return CharTokenizer(vocab_size=50)


@pytest.fixture
def model_config():
    """Create a minimal model config for smoke tests."""
    return {
        'input_dim': 40,
        'hidden_dim': 128,
        'vocab_size': 50,
        'encoder_layers': 2,
        'encoder_heads': 4,
        'decoder_layers': 1,
        'decoder_heads': 4,
        'dropout': 0.1,
        'predictor_hidden_dim': 64
    }


def test_training_loss_decreases(toy_dataset, tokenizer, model_config):
    """Test that loss decreases over 30-50 training steps."""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and loss
    model = create_paraformer_from_config(model_config)
    model.to(device)
    
    loss_fn = create_sequence_loss(
        vocab_size=model_config['vocab_size'],
        pad_token_id=tokenizer.pad_token_id,
        label_smoothing=0.0,
        predictor_loss_weight=0.1
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_seq2seq_batch
    )
    
    # Train for limited steps
    model.train()
    initial_losses = []
    final_losses = []
    
    step = 0
    max_steps = 50
    
    for epoch in range(3):
        for batch in train_loader:
            if step >= max_steps:
                break
            
            # Move batch to device
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths'].to(device)
            tokens = batch['tokens'].to(device)
            token_lengths = batch['token_lengths'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
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
                predictor_targets=None,
                feature_mask=~outputs['padding_mask']
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            if step < 5:
                initial_losses.append(total_loss.item())
            elif step >= max_steps - 5:
                final_losses.append(total_loss.item())
            
            step += 1
        
        if step >= max_steps:
            break
    
    # Verify loss decreased
    avg_initial = sum(initial_losses) / len(initial_losses)
    avg_final = sum(final_losses) / len(final_losses)
    
    assert avg_final < avg_initial, \
        f"Loss did not decrease: {avg_initial:.4f} -> {avg_final:.4f}"
    
    print(f"Loss decreased from {avg_initial:.4f} to {avg_final:.4f}")


def test_checkpoint_save_load(toy_dataset, tokenizer, model_config):
    """Test that checkpoints can be saved and loaded correctly."""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and optimizer
    model = create_paraformer_from_config(model_config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    loss_fn = create_sequence_loss(
        vocab_size=model_config['vocab_size'],
        pad_token_id=tokenizer.pad_token_id
    )
    
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(tmpdir),
            max_checkpoints=3,
            save_best=True
        )
        
        # Train a few steps
        model.train()
        for i, batch in enumerate(train_loader):
            if i >= 5:
                break
            
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths'].to(device)
            tokens = batch['tokens'].to(device)
            token_lengths = batch['token_lengths'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                features=features,
                feature_lengths=feature_lengths,
                target_tokens=tokens,
                target_lengths=token_lengths
            )
            
            loss_dict = loss_fn(
                decoder_logits=outputs['logits'],
                target_tokens=tokens,
                target_lengths=token_lengths,
                feature_mask=~outputs['padding_mask']
            )
            
            loss_dict['total_loss'].backward()
            optimizer.step()
        
        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            metrics={'train_loss': 1.5},
            is_best=True
        )
        
        # Get saved model weights
        original_weights = {
            name: param.clone() 
            for name, param in model.state_dict().items()
        }
        
        # Create new model and load checkpoint
        model2 = create_paraformer_from_config(model_config)
        model2.to(device)
        
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        
        # Load checkpoint
        checkpoint_path = Path(tmpdir) / 'best_model.pt'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model2.load_state_dict(checkpoint['model_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Verify models have same weights
        for name, param in model2.state_dict().items():
            assert torch.allclose(param, original_weights[name], atol=1e-6), \
                f"Checkpoint weights don't match for {name}"
        
        print("Checkpoint save/load successful")


def test_reproducibility(toy_dataset, tokenizer, model_config):
    """Test that training is reproducible with same seed."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_n_steps(seed, n_steps=15):
        set_seed(seed)
        
        model = create_paraformer_from_config(model_config)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        loss_fn = create_sequence_loss(
            vocab_size=model_config['vocab_size'],
            pad_token_id=tokenizer.pad_token_id
        )
        
        train_loader = torch.utils.data.DataLoader(
            toy_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_seq2seq_batch
        )
        
        model.train()
        losses = []
        
        for i, batch in enumerate(train_loader):
            if i >= n_steps:
                break
            
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths'].to(device)
            tokens = batch['tokens'].to(device)
            token_lengths = batch['token_lengths'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                features=features,
                feature_lengths=feature_lengths,
                target_tokens=tokens,
                target_lengths=token_lengths
            )
            
            loss_dict = loss_fn(
                decoder_logits=outputs['logits'],
                target_tokens=tokens,
                target_lengths=token_lengths,
                feature_mask=~outputs['padding_mask']
            )
            
            total_loss = loss_dict['total_loss']
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
        
        return losses
    
    # Train twice with same seed
    losses1 = train_n_steps(seed=42, n_steps=15)
    losses2 = train_n_steps(seed=42, n_steps=15)
    
    # Verify losses are identical (or very close due to GPU non-determinism)
    for i, (l1, l2) in enumerate(zip(losses1, losses2)):
        assert abs(l1 - l2) < 1e-4, \
            f"Training not reproducible at step {i}: {l1} != {l2}"
    
    print("Reproducibility test passed")


def test_overfitting_on_single_batch(toy_dataset, tokenizer, model_config):
    """Test that model can overfit on a single batch (sanity check)."""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_paraformer_from_config(model_config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    loss_fn = create_sequence_loss(
        vocab_size=model_config['vocab_size'],
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Get single batch
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )
    
    batch = next(iter(train_loader))
    # Store the batch data
    features_data = batch['features'].to(device)
    feature_lengths_data = batch['feature_lengths'].to(device)
    tokens_data = batch['tokens'].to(device)
    token_lengths_data = batch['token_lengths'].to(device)
    
    # Train on same batch repeatedly
    model.train()
    losses = []
    
    for i in range(100):
        optimizer.zero_grad()
        
        # Use the data without creating a persistent graph
        outputs = model(
            features=features_data,
            feature_lengths=feature_lengths_data,
            target_tokens=tokens_data,
            target_lengths=token_lengths_data
        )
        
        loss_dict = loss_fn(
            decoder_logits=outputs['logits'],
            target_tokens=tokens_data,
            target_lengths=token_lengths_data,
            feature_mask=~outputs['padding_mask']
        )
        
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
    
    # Loss should decrease significantly
    initial_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10
    
    assert final_loss < initial_loss * 0.5, \
        f"Model failed to overfit: {initial_loss:.4f} -> {final_loss:.4f}"
    
    print(f"Overfitting test passed: {initial_loss:.4f} -> {final_loss:.4f}")


def test_train_epoch_function(toy_dataset, tokenizer, model_config):
    """Test the train_epoch function from train_loop."""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_paraformer_from_config(model_config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    loss_fn = create_sequence_loss(
        vocab_size=model_config['vocab_size'],
        pad_token_id=tokenizer.pad_token_id
    )
    
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_seq2seq_batch
    )
    
    # Run one epoch
    metrics = train_epoch(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        gradient_clip=1.0,
        device=device,
        log_interval=2,
        tokenizer=tokenizer,
        epoch=0
    )
    
    # Verify metrics are returned
    assert 'train_loss' in metrics
    assert 'train_token_accuracy' in metrics
    assert 'train_decoder_loss' in metrics
    assert 'learning_rate' in metrics
    
    # Verify metrics are reasonable
    assert metrics['train_loss'] > 0
    assert 0 <= metrics['train_token_accuracy'] <= 1
    
    print(f"Train epoch metrics: {metrics}")


def test_validate_epoch_function(toy_dataset, tokenizer, model_config):
    """Test the validate_epoch function from train_loop."""
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_paraformer_from_config(model_config)
    model.to(device)
    
    loss_fn = create_sequence_loss(
        vocab_size=model_config['vocab_size'],
        pad_token_id=tokenizer.pad_token_id
    )
    
    val_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )
    
    # Run validation
    metrics = validate_epoch(
        model=model,
        dataloader=val_loader,
        loss_fn=loss_fn,
        device=device,
        tokenizer=tokenizer,
        max_samples=None
    )
    
    # Verify metrics are returned
    assert 'val_loss' in metrics
    assert 'val_token_accuracy' in metrics
    assert 'val_sequence_accuracy' in metrics
    
    # Verify metrics are reasonable
    assert metrics['val_loss'] > 0
    assert 0 <= metrics['val_token_accuracy'] <= 1
    assert 0 <= metrics['val_sequence_accuracy'] <= 1
    
    print(f"Validation metrics: {metrics}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
