"""
Unit tests for data pipeline components in Paraformer ASR.

Tests tokenizer, dataset generation, and sequence utilities.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.tokenizer import CharTokenizer, create_default_tokenizer
from src.data.toy_seq2seq import ToySeq2SeqDataset, create_toy_datasets
from src.data.utils import collate_seq2seq, create_attention_mask, pad_sequences


class TestCharTokenizer:
    """Test character-level tokenizer."""
    
    def test_tokenizer_creation(self):
        """Test basic tokenizer creation."""
        tokenizer = CharTokenizer(vocab_size=50)
        
        assert len(tokenizer) == 50
        assert tokenizer.get_vocab_size() == 50
        
        # Check special tokens
        special_tokens = tokenizer.get_special_tokens()
        assert 'pad_token_id' in special_tokens
        assert 'unk_token_id' in special_tokens
        assert 'sos_token_id' in special_tokens
        assert 'eos_token_id' in special_tokens
    
    def test_encode_decode(self):
        """Test encoding and decoding functionality."""
        tokenizer = CharTokenizer(vocab_size=100)
        
        test_texts = [
            "hello world",
            "test 123",
            "a",
            ""
        ]
        
        for text in test_texts:
            # Test encoding
            tokens = tokenizer.encode(text, add_special_tokens=True)
            assert tokens.dtype == torch.long
            assert len(tokens) >= 2  # At least SOS + EOS
            
            # Test decoding
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            assert decoded == text
            
            # Test without special tokens
            tokens_no_special = tokenizer.encode(text, add_special_tokens=False)
            decoded_no_special = tokenizer.decode(tokens_no_special, skip_special_tokens=False)
            assert decoded_no_special == text
    
    def test_batch_operations(self):
        """Test batch encoding and decoding."""
        tokenizer = CharTokenizer(vocab_size=100)
        
        texts = ["hello", "world test", "a", "longer text example"]
        
        # Test batch encoding
        batch_tokens, lengths = tokenizer.batch_encode(
            texts, 
            padding=True, 
            return_lengths=True
        )
        
        assert batch_tokens.shape[0] == len(texts)
        assert batch_tokens.dtype == torch.long
        assert lengths.dtype == torch.long
        assert len(lengths) == len(texts)
        
        # Check padding
        max_len = batch_tokens.shape[1]
        for i, length in enumerate(lengths):
            # Non-padded positions should not be pad token
            assert not torch.all(batch_tokens[i, :length] == tokenizer.pad_token_id)
            # Padded positions should be pad token
            if length < max_len:
                assert torch.all(batch_tokens[i, length:] == tokenizer.pad_token_id)
        
        # Test batch decoding
        decoded_texts = tokenizer.batch_decode(batch_tokens, lengths)
        assert decoded_texts == texts
    
    def test_unknown_characters(self):
        """Test handling of unknown characters."""
        # Create tokenizer with limited character set
        tokenizer = CharTokenizer(
            vocab_size=20,
            characters="abc123 "  # Very limited set
        )
        
        # Test with unknown characters
        text_with_unknown = "abc xyz 123"
        tokens = tokenizer.encode(text_with_unknown)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Unknown characters should be replaced with UNK
        assert tokenizer.unk_token in decoded or len(decoded) < len(text_with_unknown)
    
    def test_max_length_truncation(self):
        """Test maximum length truncation."""
        tokenizer = CharTokenizer(vocab_size=100)
        
        long_text = "a" * 100
        max_len = 20
        
        tokens = tokenizer.encode(long_text, max_length=max_len, add_special_tokens=True)
        
        assert len(tokens) == max_len
        # Should still have SOS and EOS tokens
        assert tokens[0] == tokenizer.sos_token_id
        assert tokens[-1] == tokenizer.eos_token_id
    
    def test_reproducibility(self):
        """Test tokenizer reproducibility."""
        tokenizer1 = CharTokenizer(vocab_size=50, characters="abcdef123 ")
        tokenizer2 = CharTokenizer(vocab_size=50, characters="abcdef123 ")
        
        text = "abc 123"
        tokens1 = tokenizer1.encode(text)
        tokens2 = tokenizer2.encode(text)
        
        assert torch.equal(tokens1, tokens2)
    
    def test_save_load_vocabulary(self):
        """Test vocabulary saving and loading."""
        import tempfile
        import os
        
        tokenizer = CharTokenizer(vocab_size=30, characters="hello world")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            vocab_path = os.path.join(temp_dir, "vocab.json")
            
            # Save vocabulary
            tokenizer.save_vocabulary(vocab_path)
            assert os.path.exists(vocab_path)
            
            # Load vocabulary
            loaded_tokenizer = CharTokenizer.load_vocabulary(vocab_path)
            
            # Test that loaded tokenizer works the same
            test_text = "hello world"
            original_tokens = tokenizer.encode(test_text)
            loaded_tokens = loaded_tokenizer.encode(test_text)
            
            assert torch.equal(original_tokens, loaded_tokens)
            assert loaded_tokenizer.get_vocab_size() == tokenizer.get_vocab_size()


class TestToySeq2SeqDataset:
    """Test toy sequence-to-sequence dataset."""
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        dataset = ToySeq2SeqDataset(
            num_samples=10,
            vocab_size=50,
            feature_dim=80,
            seed=42
        )
        
        assert len(dataset) == 10
        assert dataset.vocab_size == 50
        assert dataset.feature_dim == 80
    
    def test_sample_generation(self):
        """Test individual sample generation."""
        dataset = ToySeq2SeqDataset(
            num_samples=5,
            vocab_size=30,
            feature_dim=40,
            min_feat_len=20,
            max_feat_len=100,
            min_token_len=5,
            max_token_len=25,
            seed=42
        )
        
        features, tokens, feat_len, token_len = dataset[0]
        
        # Check shapes
        assert features.shape[1] == 40  # feature_dim
        assert features.shape[0] >= feat_len  # May be padded
        assert tokens.shape[0] >= token_len  # May be padded
        
        # Check data types
        assert features.dtype == torch.float32
        assert tokens.dtype == torch.long
        
        # Check lengths are within bounds
        assert 20 <= feat_len <= 100
        assert 5 <= token_len <= 25
        
        # Check token values are valid
        assert tokens.min() >= 0
        assert tokens.max() < 30
        
        # Check for NaN/Inf in features
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    def test_reproducibility(self):
        """Test dataset reproducibility with same seed."""
        dataset1 = ToySeq2SeqDataset(num_samples=3, seed=123)
        dataset2 = ToySeq2SeqDataset(num_samples=3, seed=123)
        
        for i in range(3):
            feat1, tok1, flen1, tlen1 = dataset1[i]
            feat2, tok2, flen2, tlen2 = dataset2[i]
            
            assert torch.allclose(feat1, feat2, atol=1e-6)
            assert torch.equal(tok1, tok2)
            assert flen1 == flen2
            assert tlen1 == tlen2
    
    def test_complexity_levels(self):
        """Test different complexity levels."""
        complexities = ['simple', 'medium', 'high']
        
        for complexity in complexities:
            dataset = ToySeq2SeqDataset(
                num_samples=5,
                complexity=complexity,
                seed=42
            )
            
            # All should generate valid samples
            features, tokens, feat_len, token_len = dataset[0]
            
            assert features.shape[1] == dataset.feature_dim
            assert not torch.isnan(features).any()
            assert not torch.isinf(features).any()
            assert tokens.min() >= 0
            assert tokens.max() < dataset.vocab_size
    
    def test_feature_token_correlation(self):
        """Test that features are correlated with tokens."""
        dataset = ToySeq2SeqDataset(
            num_samples=10,
            complexity='simple',  # Less noise for clearer correlation
            feature_noise=0.01,   # Minimal noise
            seed=42
        )
        
        # Generate multiple samples and check basic correlation properties
        alignment_ratios = []
        
        for i in range(10):
            features, tokens, feat_len, token_len = dataset[i]
            ratio = feat_len / token_len
            alignment_ratios.append(ratio)
            
            # Features should have reasonable alignment ratio
            assert 1.0 <= ratio <= 10.0  # Reasonable range for speech
        
        # Alignment ratios should have some variance but be reasonable
        ratios_tensor = torch.tensor(alignment_ratios)
        assert ratios_tensor.std() > 0.1  # Some variance
        assert ratios_tensor.mean() > 1.5  # Features generally longer than tokens
    
    def test_sample_info(self):
        """Test sample information retrieval."""
        dataset = ToySeq2SeqDataset(num_samples=3, seed=42)
        
        info = dataset.get_sample_info(0)
        
        required_keys = [
            'index', 'feat_len', 'token_len', 'feat_shape', 
            'token_shape', 'alignment_ratio', 'feature_stats'
        ]
        
        for key in required_keys:
            assert key in info
        
        # Check feature stats
        stats = info['feature_stats']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        # Stats should be reasonable
        assert not np.isnan(stats['mean'])
        assert stats['std'] >= 0
    
    def test_dataset_splits(self):
        """Test dataset split creation."""
        train_dataset, val_dataset, test_dataset = create_toy_datasets(
            train_size=100,
            val_size=20,
            test_size=10,
            vocab_size=50
        )
        
        assert len(train_dataset) == 100
        assert len(val_dataset) == 20
        assert len(test_dataset) == 10
        
        # All should have same configuration
        assert train_dataset.vocab_size == val_dataset.vocab_size == test_dataset.vocab_size
        assert train_dataset.feature_dim == val_dataset.feature_dim == test_dataset.feature_dim
        
        # But should generate different samples (different seeds)
        train_sample = train_dataset[0]
        val_sample = val_dataset[0]
        
        assert not torch.allclose(train_sample[0], val_sample[0])


class TestSequenceUtilities:
    """Test sequence processing utilities."""
    
    def test_pad_sequences(self):
        """Test sequence padding utility."""
        sequences = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8, 9])
        ]
        
        padded, lengths = pad_sequences(sequences, pad_value=0)
        
        # Check output shapes
        assert padded.shape == (3, 4)  # 3 sequences, max length 4
        assert lengths.shape == (3,)
        
        # Check padding
        expected_padded = torch.tensor([
            [1, 2, 3, 0],
            [4, 5, 0, 0],
            [6, 7, 8, 9]
        ])
        expected_lengths = torch.tensor([3, 2, 4])
        
        assert torch.equal(padded, expected_padded)
        assert torch.equal(lengths, expected_lengths)
    
    def test_create_attention_mask(self):
        """Test attention mask creation."""
        lengths = torch.tensor([3, 2, 4])
        max_len = 4
        
        mask = create_attention_mask(lengths, max_len)
        
        expected_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1]
        ], dtype=torch.bool)
        
        assert torch.equal(mask, expected_mask)
        assert mask.dtype == torch.bool
    
    def test_collate_seq2seq(self):
        """Test sequence-to-sequence collation function."""
        # Create sample batch data
        batch_data = [
            (torch.randn(10, 80), torch.tensor([1, 2, 3]), 10, 3),
            (torch.randn(15, 80), torch.tensor([4, 5]), 15, 2),
            (torch.randn(8, 80), torch.tensor([6, 7, 8, 9]), 8, 4)
        ]
        
        collated = collate_seq2seq(batch_data)
        
        # Check that all required keys are present
        required_keys = [
            'features', 'tokens', 'feature_lengths', 'token_lengths',
            'feature_mask', 'token_mask'
        ]
        
        for key in required_keys:
            assert key in collated
        
        # Check shapes
        batch_size = 3
        max_feat_len = 15
        max_token_len = 4
        feature_dim = 80
        
        assert collated['features'].shape == (batch_size, max_feat_len, feature_dim)
        assert collated['tokens'].shape == (batch_size, max_token_len)
        assert collated['feature_lengths'].shape == (batch_size,)
        assert collated['token_lengths'].shape == (batch_size,)
        assert collated['feature_mask'].shape == (batch_size, max_feat_len)
        assert collated['token_mask'].shape == (batch_size, max_token_len)
        
        # Check data types
        assert collated['features'].dtype == torch.float32
        assert collated['tokens'].dtype == torch.long
        assert collated['feature_lengths'].dtype == torch.long
        assert collated['token_lengths'].dtype == torch.long
        assert collated['feature_mask'].dtype == torch.bool
        assert collated['token_mask'].dtype == torch.bool


class TestDataLoading:
    """Test data loading with DataLoader."""
    
    def test_dataloader_integration(self):
        """Test DataLoader integration with toy dataset."""
        dataset = ToySeq2SeqDataset(num_samples=8, seed=42)
        
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            collate_fn=collate_seq2seq,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        
        # Check batch structure
        assert isinstance(batch, dict)
        assert 'features' in batch
        assert 'tokens' in batch
        assert 'feature_lengths' in batch
        assert 'token_lengths' in batch
        assert 'feature_mask' in batch
        assert 'token_mask' in batch
        
        # Check batch size
        assert batch['features'].shape[0] == 3
        assert batch['tokens'].shape[0] == 3
        
        # Check that masks are consistent with lengths
        for i in range(3):
            feat_len = batch['feature_lengths'][i].item()
            token_len = batch['token_lengths'][i].item()
            
            # Check feature mask
            feat_mask = batch['feature_mask'][i]
            assert feat_mask[:feat_len].all()  # Should be True for valid positions
            if feat_len < feat_mask.shape[0]:
                assert not feat_mask[feat_len:].any()  # Should be False for padded positions
            
            # Check token mask
            token_mask = batch['token_mask'][i]
            assert token_mask[:token_len].all()  # Should be True for valid positions
            if token_len < token_mask.shape[0]:
                assert not token_mask[token_len:].any()  # Should be False for padded positions
    
    def test_variable_batch_sizes(self):
        """Test handling of variable batch sizes."""
        dataset = ToySeq2SeqDataset(num_samples=7, seed=42)
        
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_seq2seq,
            num_workers=0
        )
        
        batches = list(dataloader)
        
        # Should have 3 batches: [3, 3, 1]
        assert len(batches) == 3
        assert batches[0]['features'].shape[0] == 3
        assert batches[1]['features'].shape[0] == 3
        assert batches[2]['features'].shape[0] == 1


if __name__ == "__main__":
    # Run basic tests
    print("Running ASR data pipeline tests...")
    
    # Test tokenizer
    test_tokenizer = TestCharTokenizer()
    test_tokenizer.test_tokenizer_creation()
    test_tokenizer.test_encode_decode()
    test_tokenizer.test_batch_operations()
    print("✓ Tokenizer tests passed")
    
    # Test dataset
    test_dataset = TestToySeq2SeqDataset()
    test_dataset.test_dataset_creation()
    test_dataset.test_sample_generation()
    test_dataset.test_reproducibility()
    print("✓ Dataset tests passed")
    
    # Test utilities
    test_utils = TestSequenceUtilities()
    test_utils.test_pad_sequences()
    test_utils.test_create_attention_mask()
    test_utils.test_collate_seq2seq()
    print("✓ Sequence utilities tests passed")
    
    # Test data loading
    test_loading = TestDataLoading()
    test_loading.test_dataloader_integration()
    test_loading.test_variable_batch_sizes()
    print("✓ DataLoader tests passed")
    
    print("All ASR data pipeline tests completed successfully!")