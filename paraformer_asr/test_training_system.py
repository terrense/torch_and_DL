#!/usr/bin/env python3
"""
Test script for ASR training and inference systems.

Verifies that all components work together correctly with a small-scale test.
"""

import sys
from pathlib import Path
import torch
import tempfile
import shutil
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config, ModelConfig, DataConfig, TrainingConfig, ExperimentConfig
from data.toy_seq2seq import ToySeq2SeqDataset
from data.tokenizer import create_default_tokenizer
from data.utils import collate_seq2seq_batch
from models.paraformer import create_paraformer_from_config
from losses.seq_loss import create_sequence_loss
from decode.greedy import greedy_decode
from training.trainer import ASRTrainer
from evaluation.evaluator import ASREvaluator
from evaluation.inference import ASRInference
from utils.logging_utils import setup_logger
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_loss_functions():
    """Test sequence loss functions."""
    print("\n" + "="*50)
    print("Testing Loss Functions")
    print("="*50)
    
    from losses.seq_loss import MaskedCrossEntropyLoss, PredictorLoss, CombinedASRLoss
    
    # Test parameters
    B, S, T, V = 2, 10, 20, 50
    pad_token_id = 0
    
    # Create test data
    logits = torch.randn(B, S, V)
    targets = torch.randint(1, V, (B, S))
    targets[:, -2:] = pad_token_id  # Add padding
    lengths = torch.tensor([S-2, S-1])
    
    predictor_preds = torch.randn(B, T, 1)
    predictor_targets = torch.randint(0, 2, (T,)).float().unsqueeze(0).expand(B, T)
    feature_mask = torch.ones(B, T, dtype=torch.bool)
    
    # Test combined loss
    combined_loss = CombinedASRLoss(
        vocab_size=V,
        pad_token_id=pad_token_id,
        label_smoothing=0.1,
        predictor_loss_weight=0.1
    )
    
    loss_dict = combined_loss(
        decoder_logits=logits,
        target_tokens=targets,
        target_lengths=lengths,
        predictor_predictions=predictor_preds,
        predictor_targets=predictor_targets,
        feature_mask=feature_mask
    )
    
    print(f"✓ Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"✓ Decoder loss: {loss_dict['decoder_loss'].item():.4f}")
    print(f"✓ Token accuracy: {loss_dict['token_accuracy'].item():.4f}")
    
    return True


def test_greedy_decoding():
    """Test greedy decoding system."""
    print("\n" + "="*50)
    print("Testing Greedy Decoding")
    print("="*50)
    
    # Create tokenizer
    tokenizer = create_default_tokenizer(vocab_size=50)
    
    # Test parameters
    B, S, V = 2, 15, len(tokenizer)
    
    # Create test logits
    logits = torch.randn(B, S, V)
    lengths = torch.tensor([12, 10])
    
    # Test greedy decoding
    texts = greedy_decode(logits, tokenizer, lengths)
    
    print(f"✓ Decoded {len(texts)} sequences:")
    for i, text in enumerate(texts):
        print(f"  Sequence {i}: '{text}'")
    
    return True


def test_model_creation():
    """Test model creation and forward pass."""
    print("\n" + "="*50)
    print("Testing Model Creation")
    print("="*50)
    
    # Create tokenizer
    tokenizer = create_default_tokenizer(vocab_size=100)
    
    # Model configuration
    model_config = {
        'input_dim': 80,
        'vocab_size': len(tokenizer),
        'encoder_dim': 128,
        'encoder_layers': 2,
        'decoder_layers': 1,
        'encoder_heads': 4,
        'decoder_heads': 4
    }
    
    # Create model
    model = create_paraformer_from_config(model_config)
    
    # Test forward pass
    B, T, F = 2, 50, 80
    S = 20
    
    features = torch.randn(B, T, F)
    feature_lengths = torch.tensor([45, 40])
    tokens = torch.randint(1, len(tokenizer)-1, (B, S))
    token_lengths = torch.tensor([18, 15])
    
    # Forward pass
    outputs = model(
        features=features,
        feature_lengths=feature_lengths,
        target_tokens=tokens,
        target_lengths=token_lengths
    )
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Forward pass successful")
    print(f"✓ Output logits shape: {outputs['logits'].shape}")
    print(f"✓ Predictor predictions shape: {outputs['predictor_predictions'].shape}")
    
    return True, model, tokenizer


def test_data_pipeline():
    """Test data pipeline."""
    print("\n" + "="*50)
    print("Testing Data Pipeline")
    print("="*50)
    
    # Create tokenizer
    tokenizer = create_default_tokenizer(vocab_size=100)
    
    # Create dataset
    dataset = ToySeq2SeqDataset(
        vocab_size=len(tokenizer),
        num_samples=50,
        min_seq_len=10,
        max_seq_len=30,
        feature_dim=80,
        tokenizer=tokenizer
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_seq2seq_batch
    )
    
    # Test batch
    batch = next(iter(dataloader))
    
    print(f"✓ Dataset created with {len(dataset)} samples")
    print(f"✓ Batch features shape: {batch['features'].shape}")
    print(f"✓ Batch tokens shape: {batch['tokens'].shape}")
    print(f"✓ Feature lengths: {batch['feature_lengths'].tolist()}")
    print(f"✓ Token lengths: {batch['token_lengths'].tolist()}")
    
    return True, dataloader, tokenizer


def test_training_loop():
    """Test training loop with minimal setup."""
    print("\n" + "="*50)
    print("Testing Training Loop")
    print("="*50)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create configuration
        config = {
            'model': {
                'input_dim': 80,
                'vocab_size': 100,
                'encoder_dim': 64,
                'encoder_layers': 1,
                'decoder_layers': 1,
                'encoder_heads': 2,
                'decoder_heads': 2
            },
            'training': {
                'num_epochs': 2,
                'learning_rate': 1e-3,
                'optimizer': 'adamw',
                'gradient_clip': 1.0
            },
            'data': {
                'batch_size': 4,
                'train_samples': 20,
                'val_samples': 10
            },
            'experiment': {
                'seed': 42,
                'log_every': 5
            }
        }
        
        # Create tokenizer
        tokenizer = create_default_tokenizer(vocab_size=config['model']['vocab_size'])
        
        # Create datasets
        train_dataset = ToySeq2SeqDataset(
            vocab_size=len(tokenizer),
            num_samples=config['data']['train_samples'],
            min_seq_len=10,
            max_seq_len=20,
            feature_dim=config['model']['input_dim'],
            tokenizer=tokenizer
        )
        
        val_dataset = ToySeq2SeqDataset(
            vocab_size=len(tokenizer),
            num_samples=config['data']['val_samples'],
            min_seq_len=10,
            max_seq_len=20,
            feature_dim=config['model']['input_dim'],
            tokenizer=tokenizer
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            collate_fn=collate_seq2seq_batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            collate_fn=collate_seq2seq_batch
        )
        
        # Create trainer
        trainer = ASRTrainer(
            config=config,
            output_dir=temp_dir,
            device=torch.device('cpu')
        )
        
        # Run training
        results = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            tokenizer=tokenizer
        )
        
        print(f"✓ Training completed successfully")
        print(f"✓ Final train loss: {results['final_metrics']['final_train_loss']:.4f}")
        print(f"✓ Final train accuracy: {results['final_metrics']['final_train_accuracy']:.4f}")
        
        return True, trainer.model, tokenizer
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_evaluation():
    """Test evaluation pipeline."""
    print("\n" + "="*50)
    print("Testing Evaluation Pipeline")
    print("="*50)
    
    # Use model from previous test
    success, model, tokenizer = test_model_creation()
    if not success:
        return False
    
    # Create test dataset
    dataset = ToySeq2SeqDataset(
        vocab_size=len(tokenizer),
        num_samples=20,
        min_seq_len=10,
        max_seq_len=20,
        feature_dim=80,
        tokenizer=tokenizer
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_seq2seq_batch
    )
    
    # Create loss function
    loss_fn = create_sequence_loss(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Create evaluator
    evaluator = ASREvaluator(
        model=model,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        device=torch.device('cpu')
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataloader=dataloader,
        max_batches=2
    )
    
    print(f"✓ Evaluation completed")
    print(f"✓ Samples evaluated: {results['num_samples']}")
    print(f"✓ Sequence accuracy: {results['sequence_accuracy']:.4f}")
    print(f"✓ WER: {results['wer']:.4f}")
    print(f"✓ CER: {results['cer']:.4f}")
    
    return True


def test_inference():
    """Test inference pipeline."""
    print("\n" + "="*50)
    print("Testing Inference Pipeline")
    print("="*50)
    
    # Use model from previous test
    success, model, tokenizer = test_model_creation()
    if not success:
        return False
    
    # Create inference instance
    inference = ASRInference(
        model=model,
        tokenizer=tokenizer,
        device=torch.device('cpu')
    )
    
    # Generate test features
    T, F = 50, 80
    features = torch.randn(T, F)
    
    # Run inference
    result = inference.infer_features(
        features=features,
        return_confidence=True,
        return_tokens=True
    )
    
    print(f"✓ Inference completed")
    print(f"✓ Predicted text: '{result['text']}'")
    print(f"✓ Token length: {result['token_length']}")
    print(f"✓ Average confidence: {result.get('avg_confidence', 0.0):.4f}")
    
    return True


def main():
    """Run all tests."""
    print("Starting ASR Training and Inference System Tests")
    print("="*60)
    
    tests = [
        ("Loss Functions", test_loss_functions),
        ("Greedy Decoding", test_greedy_decoding),
        ("Model Creation", lambda: test_model_creation()[0]),
        ("Data Pipeline", lambda: test_data_pipeline()[0]),
        ("Training Loop", lambda: test_training_loop()[0]),
        ("Evaluation", test_evaluation),
        ("Inference", test_inference)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name} test...")
            success = test_func()
            if success:
                print(f"✓ {test_name} test PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} test FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} test FAILED with error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! ASR training and inference system is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())