#!/usr/bin/env python3
"""
Simple test for ASR training and inference systems.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test loss functions
        from losses.seq_loss import MaskedCrossEntropyLoss, CombinedASRLoss
        print("✓ Loss functions imported")
        
        # Test greedy decoding
        from decode.greedy import GreedyDecoder, greedy_decode
        print("✓ Greedy decoding imported")
        
        # Test training
        from training.trainer import ASRTrainer
        from training.train_loop import train_epoch, validate_epoch
        print("✓ Training modules imported")
        
        # Test evaluation
        from evaluation.evaluator import ASREvaluator, compute_wer, compute_cer
        from evaluation.inference import ASRInference
        print("✓ Evaluation modules imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_loss_functions():
    """Test loss functions work."""
    print("\nTesting loss functions...")
    
    try:
        from losses.seq_loss import CombinedASRLoss
        
        # Test parameters
        B, S, V = 2, 10, 50
        
        # Create test data
        logits = torch.randn(B, S, V)
        targets = torch.randint(1, V, (B, S))
        lengths = torch.tensor([8, 9])
        
        # Create loss function
        loss_fn = CombinedASRLoss(vocab_size=V, pad_token_id=0)
        
        # Compute loss
        loss_dict = loss_fn(
            decoder_logits=logits,
            target_tokens=targets,
            target_lengths=lengths
        )
        
        print(f"✓ Loss computed: {loss_dict['total_loss'].item():.4f}")
        print(f"✓ Token accuracy: {loss_dict['token_accuracy'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_greedy_decoding():
    """Test greedy decoding."""
    print("\nTesting greedy decoding...")
    
    try:
        from decode.greedy import greedy_decode
        from data.tokenizer import create_default_tokenizer
        
        # Create tokenizer
        tokenizer = create_default_tokenizer(vocab_size=50)
        
        # Test parameters
        B, S, V = 2, 10, len(tokenizer)
        
        # Create test logits
        logits = torch.randn(B, S, V)
        lengths = torch.tensor([8, 9])
        
        # Decode
        texts = greedy_decode(logits, tokenizer, lengths)
        
        print(f"✓ Decoded {len(texts)} sequences")
        for i, text in enumerate(texts):
            print(f"  Sequence {i}: '{text}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Greedy decoding test failed: {e}")
        return False


def test_wer_cer():
    """Test WER and CER computation."""
    print("\nTesting WER/CER computation...")
    
    try:
        from evaluation.evaluator import compute_wer, compute_cer
        
        predictions = ["hello world", "this is test", "good morning"]
        references = ["hello world", "this is a test", "good morning"]
        
        wer = compute_wer(predictions, references)
        cer = compute_cer(predictions, references)
        
        print(f"✓ WER: {wer:.4f}")
        print(f"✓ CER: {cer:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ WER/CER test failed: {e}")
        return False


def main():
    """Run simple tests."""
    print("Running Simple ASR Training System Tests")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Loss Functions", test_loss_functions),
        ("Greedy Decoding", test_greedy_decoding),
        ("WER/CER Computation", test_wer_cer)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
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
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All basic tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())