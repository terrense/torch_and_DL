"""
Simplified Smoke Tests for Paraformer ASR
简化的Paraformer ASR冒烟测试

These tests avoid the computation graph reuse issue by creating fresh model instances.
这些测试通过创建新的模型实例来避免计算图重用问题。
"""
import pytest
import torch
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.toy_seq2seq import ToySeq2SeqDataset
from src.data.tokenizer import CharTokenizer
from src.data.utils import collate_seq2seq_batch
from src.models.paraformer import create_paraformer_from_config
from src.losses.seq_loss import create_sequence_loss
from src.utils.reproducibility import set_seed
from src.utils.checkpoint import CheckpointManager


@pytest.fixture
def toy_dataset():
    """
    Create a small toy dataset for smoke testing.
    创建用于冒烟测试的小型玩具数据集。
    """
    return ToySeq2SeqDataset(
        num_samples=20,  # Smaller for faster tests
        vocab_size=50,
        max_token_len=15,
        feature_dim=40,
        max_feat_len=60,
        complexity='simple'
    )


@pytest.fixture
def tokenizer():
    """
    Create a simple tokenizer for testing.
    创建用于测试的简单分词器。
    """
    return CharTokenizer(vocab_size=50)


@pytest.fixture
def model_config():
    """
    Create a minimal model config for smoke tests.
    创建用于冒烟测试的最小模型配置。
    """
    return {
        'input_dim': 40,
        'hidden_dim': 64,  # Smaller for faster tests
        'vocab_size': 50,
        'encoder_layers': 1,  # Minimal layers
        'encoder_heads': 2,
        'decoder_layers': 1,
        'decoder_heads': 2,
        'dropout': 0.1,
        'predictor_hidden_dim': 32
    }


def test_model_forward_pass(toy_dataset, tokenizer, model_config):
    """
    Test that model can perform forward pass without errors.
    测试模型可以执行前向传播而不出错。
    
    This is the most basic smoke test - just verify the model runs.
    这是最基本的冒烟测试 - 只验证模型运行。
    """
    set_seed(42)
    device = torch.device('cpu')  # Use CPU to avoid CUDA issues
    
    # Create model
    model = create_paraformer_from_config(model_config)
    model.to(device)
    model.eval()  # Eval mode to avoid dropout randomness
    
    # Get a batch
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )
    
    batch = next(iter(train_loader))
    features = batch['features'].to(device)
    feature_lengths = batch['feature_lengths'].to(device)
    tokens = batch['tokens'].to(device)
    token_lengths = batch['token_lengths'].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            features=features,
            feature_lengths=feature_lengths,
            target_tokens=tokens,
            target_lengths=token_lengths
        )
    
    # Verify outputs
    assert 'logits' in outputs
    assert outputs['logits'].shape[0] == 2  # Batch size
    assert outputs['logits'].shape[2] == model_config['vocab_size']
    assert not torch.isnan(outputs['logits']).any()
    
    print("✓ Model forward pass successful")


def test_single_training_step(toy_dataset, tokenizer, model_config):
    """
    Test that a single training step works.
    测试单个训练步骤是否有效。
    
    Creates a fresh model for each step to avoid graph reuse.
    为每个步骤创建新模型以避免图重用。
    """
    set_seed(42)
    device = torch.device('cpu')
    
    # Get a batch
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )
    
    batch = next(iter(train_loader))
    features = batch['features'].to(device)
    feature_lengths = batch['feature_lengths'].to(device)
    tokens = batch['tokens'].to(device)
    token_lengths = batch['token_lengths'].to(device)
    
    # Create fresh model and optimizer
    model = create_paraformer_from_config(model_config)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = create_sequence_loss(
        vocab_size=model_config['vocab_size'],
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Single training step
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
    
    # Verify loss is finite
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0
    
    print(f"✓ Single training step successful, loss={total_loss.item():.4f}")


def test_loss_decreases_with_fresh_models(toy_dataset, tokenizer, model_config):
    """
    Test that loss decreases by training on different batches.
    通过在不同批次上训练来测试损失是否下降。
    
    Uses different batches to avoid graph reuse.
    使用不同批次以避免图重用。
    """
    set_seed(42)
    device = torch.device('cpu')
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )
    
    # Create model and optimizer
    model = create_paraformer_from_config(model_config)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = create_sequence_loss(
        vocab_size=model_config['vocab_size'],
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Train for 10 steps on different batches
    losses = []
    for step, batch in enumerate(train_loader):
        if step >= 10:
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
    
    # Compare first and last losses
    initial_loss = sum(losses[:3]) / 3
    final_loss = sum(losses[-3:]) / 3
    
    print(f"Initial loss (avg of first 3): {initial_loss:.4f}")
    print(f"Final loss (avg of last 3): {final_loss:.4f}")
    print(f"Decrease: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    
    assert final_loss < initial_loss, \
        f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    
    print("✓ Loss decrease test passed")


def test_checkpoint_save_load(toy_dataset, tokenizer, model_config):
    """
    Test checkpoint save and load functionality.
    测试检查点保存和加载功能。
    """
    set_seed(42)
    device = torch.device('cpu')
    
    # Create model and optimizer
    model1 = create_paraformer_from_config(model_config)
    model1.to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(tmpdir),
            max_checkpoints=3
        )
        
        # Save checkpoint using correct API
        checkpoint_manager.save_checkpoint(
            model=model1,
            optimizer=optimizer1,
            epoch=1,
            metrics={'loss': 1.0}
        )
        
        # Load into new model
        model2 = create_paraformer_from_config(model_config)
        model2.to(device)
        
        checkpoint_path = Path(tmpdir) / 'checkpoint_epoch_001.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), 
            model2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, atol=1e-6), \
                f"Weights don't match for {name1}"
        
        print("✓ Checkpoint save/load test passed")


def test_reproducibility_simple(toy_dataset, tokenizer, model_config):
    """
    Test basic reproducibility with same seed.
    测试使用相同种子的基本可复现性。
    """
    device = torch.device('cpu')
    
    # Get a batch
    train_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )
    
    batch = next(iter(train_loader))
    features = batch['features'].to(device)
    feature_lengths = batch['feature_lengths'].to(device)
    tokens = batch['tokens'].to(device)
    token_lengths = batch['token_lengths'].to(device)
    
    loss_fn = create_sequence_loss(
        vocab_size=model_config['vocab_size'],
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Run 1
    set_seed(42)
    model1 = create_paraformer_from_config(model_config)
    model1.to(device)
    model1.eval()
    
    with torch.no_grad():
        outputs1 = model1(
            features=features,
            feature_lengths=feature_lengths,
            target_tokens=tokens,
            target_lengths=token_lengths
        )
        loss_dict1 = loss_fn(
            decoder_logits=outputs1['logits'],
            target_tokens=tokens,
            target_lengths=token_lengths,
            feature_mask=~outputs1['padding_mask']
        )
        loss1 = loss_dict1['total_loss'].item()
    
    # Run 2
    set_seed(42)
    model2 = create_paraformer_from_config(model_config)
    model2.to(device)
    model2.eval()
    
    with torch.no_grad():
        outputs2 = model2(
            features=features,
            feature_lengths=feature_lengths,
            target_tokens=tokens,
            target_lengths=token_lengths
        )
        loss_dict2 = loss_fn(
            decoder_logits=outputs2['logits'],
            target_tokens=tokens,
            target_lengths=token_lengths,
            feature_mask=~outputs2['padding_mask']
        )
        loss2 = loss_dict2['total_loss'].item()
    
    # Verify reproducibility
    print(f"Loss 1: {loss1:.6f}")
    print(f"Loss 2: {loss2:.6f}")
    print(f"Difference: {abs(loss1 - loss2):.10f}")
    
    assert abs(loss1 - loss2) < 1e-5, \
        f"Not reproducible: {loss1} != {loss2}"
    
    print("✓ Reproducibility test passed")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
