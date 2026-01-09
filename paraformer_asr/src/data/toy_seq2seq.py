"""
Toy sequence-to-sequence dataset for Paraformer ASR training.

Generates synthetic speech-like features correlated to token sequences
with controllable difficulty and variable lengths.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
from torch.utils.data import Dataset


class ToySeq2SeqDataset(Dataset):
    """
    Synthetic sequence-to-sequence dataset for ASR training.
    
    Generates speech-like features that are correlated with character sequences,
    allowing end-to-end training without real audio data.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        vocab_size: int = 100,
        feature_dim: int = 80,
        min_feat_len: int = 50,
        max_feat_len: int = 300,
        min_token_len: int = 10,
        max_token_len: int = 60,
        feature_noise: float = 0.05,
        complexity: str = 'medium',
        seed: Optional[int] = None
    ):
        """
        Initialize toy dataset.
        
        Args:
            num_samples: Number of samples in dataset
            vocab_size: Size of character vocabulary
            feature_dim: Dimension of feature vectors (like mel-spectrogram)
            min_feat_len: Minimum feature sequence length
            max_feat_len: Maximum feature sequence length
            min_token_len: Minimum token sequence length
            max_token_len: Maximum token sequence length
            feature_noise: Standard deviation of noise added to features
            complexity: Dataset complexity ('simple', 'medium', 'high')
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.min_feat_len = min_feat_len
        self.max_feat_len = max_feat_len
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.feature_noise = feature_noise
        self.complexity = complexity
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Create character embeddings for feature generation
        self.char_embeddings = nn.Embedding(vocab_size, feature_dim)
        nn.init.normal_(self.char_embeddings.weight, std=0.1)
        
        # Complexity-dependent parameters
        self.complexity_params = self._get_complexity_params()
        
        # Pre-generate all samples for consistency
        self.samples = []
        for i in range(num_samples):
            sample = self._generate_sample(i)
            self.samples.append(sample)
    
    def _get_complexity_params(self) -> dict:
        """Get parameters based on complexity level."""
        if self.complexity == 'simple':
            return {
                'alignment_noise': 0.1,
                'duration_variance': 0.2,
                'feature_smoothing': 0.8,
                'repetition_prob': 0.1
            }
        elif self.complexity == 'medium':
            return {
                'alignment_noise': 0.2,
                'duration_variance': 0.4,
                'feature_smoothing': 0.6,
                'repetition_prob': 0.2
            }
        else:  # high
            return {
                'alignment_noise': 0.3,
                'duration_variance': 0.6,
                'feature_smoothing': 0.4,
                'repetition_prob': 0.3
            }
    
    def _generate_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Generate a single sample with correlated features and tokens.
        
        Returns:
            features: [T_feat, F] speech-like features
            tokens: [T_token] token sequence
            feat_len: actual feature length
            token_len: actual token length
        """
        # Set sample-specific seed for reproducibility
        torch.manual_seed(index + 42)
        np.random.seed(index + 42)
        
        # Generate token sequence
        token_len = torch.randint(self.min_token_len, self.max_token_len + 1, (1,)).item()
        tokens = torch.randint(0, self.vocab_size, (token_len,))
        
        # Generate feature sequence length (roughly correlated with token length)
        base_feat_len = int(token_len * (2.0 + torch.randn(1).item() * self.complexity_params['duration_variance']))
        feat_len = max(self.min_feat_len, min(self.max_feat_len, base_feat_len))
        
        # Generate features correlated with tokens
        features = self._generate_correlated_features(tokens, feat_len)
        
        return features, tokens, feat_len, token_len
    
    def _generate_correlated_features(self, tokens: torch.Tensor, feat_len: int) -> torch.Tensor:
        """
        Generate speech-like features correlated with token sequence.
        
        Args:
            tokens: [T_token] token sequence
            feat_len: desired feature sequence length
            
        Returns:
            features: [T_feat, F] correlated features
        """
        token_len = len(tokens)
        
        # Get character embeddings
        char_embeds = self.char_embeddings(tokens)  # [T_token, F]
        
        # Create alignment between tokens and features
        alignment = self._create_alignment(token_len, feat_len)
        
        # Interpolate character embeddings to feature length
        features = torch.zeros(feat_len, self.feature_dim)
        for t in range(feat_len):
            # Find which token(s) this feature frame corresponds to
            token_weights = alignment[t]  # [T_token]
            features[t] = torch.sum(char_embeds * token_weights.unsqueeze(1), dim=0)
        
        # Add temporal smoothing
        features = self._apply_temporal_smoothing(features)
        
        # Add noise
        if self.feature_noise > 0:
            noise = torch.randn_like(features) * self.feature_noise
            features = features + noise
        
        # Add some repetition patterns (common in speech)
        if torch.rand(1).item() < self.complexity_params['repetition_prob']:
            features = self._add_repetition_patterns(features)
        
        return features
    
    def _create_alignment(self, token_len: int, feat_len: int) -> torch.Tensor:
        """
        Create soft alignment between tokens and feature frames.
        
        Args:
            token_len: number of tokens
            feat_len: number of feature frames
            
        Returns:
            alignment: [T_feat, T_token] soft alignment weights
        """
        # Create monotonic alignment with some noise
        alignment = torch.zeros(feat_len, token_len)
        
        # Base alignment: linear interpolation
        for t in range(feat_len):
            # Map feature frame to token position
            token_pos = (t / feat_len) * token_len
            
            # Add alignment noise
            noise = torch.randn(1).item() * self.complexity_params['alignment_noise']
            token_pos = max(0, min(token_len - 1, token_pos + noise))
            
            # Soft alignment around the position
            for k in range(token_len):
                distance = abs(k - token_pos)
                weight = torch.exp(torch.tensor(-distance * 2.0))  # Gaussian-like weighting
                alignment[t, k] = weight
        
        # Normalize alignment weights
        alignment = alignment / (alignment.sum(dim=1, keepdim=True) + 1e-8)
        
        return alignment
    
    def _apply_temporal_smoothing(self, features: torch.Tensor) -> torch.Tensor:
        """Apply temporal smoothing to make features more speech-like."""
        smoothing = self.complexity_params['feature_smoothing']
        
        if smoothing <= 0:
            return features
        
        # Simple moving average smoothing
        kernel_size = 3
        padding = kernel_size // 2
        
        # Pad features
        padded = torch.cat([
            features[:padding].flip(0),
            features,
            features[-padding:].flip(0)
        ], dim=0)
        
        # Apply smoothing
        smoothed = torch.zeros_like(features)
        for i in range(len(features)):
            window = padded[i:i+kernel_size]
            smoothed[i] = window.mean(dim=0)
        
        # Blend with original
        features = smoothing * smoothed + (1 - smoothing) * features
        
        return features
    
    def _add_repetition_patterns(self, features: torch.Tensor) -> torch.Tensor:
        """Add repetition patterns common in speech."""
        feat_len = len(features)
        
        # Choose a random segment to repeat
        seg_len = torch.randint(2, min(10, feat_len // 4), (1,)).item()
        start_pos = torch.randint(0, feat_len - seg_len, (1,)).item()
        
        # Repeat the segment 2-3 times
        num_repeats = torch.randint(2, 4, (1,)).item()
        segment = features[start_pos:start_pos + seg_len]
        
        # Find a place to insert repetitions
        min_insert_pos = start_pos + seg_len
        max_insert_pos = feat_len - seg_len * num_repeats
        
        # Skip repetition if not enough space
        if min_insert_pos >= max_insert_pos:
            return features
            
        insert_pos = torch.randint(min_insert_pos, max_insert_pos, (1,)).item()
        
        # Insert repetitions with slight variations
        for i in range(num_repeats):
            pos = insert_pos + i * seg_len
            if pos + seg_len <= feat_len:
                # Add slight variation to each repetition
                variation = torch.randn_like(segment) * 0.05
                features[pos:pos + seg_len] = segment + variation
        
        return features
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Get a single sample.
        
        Returns:
            features: [T_feat, F] feature sequence
            tokens: [T_token] token sequence  
            feat_len: actual feature length
            token_len: actual token length
        """
        return self.samples[index]
    
    def get_sample_info(self, index: int) -> dict:
        """Get detailed information about a sample."""
        features, tokens, feat_len, token_len = self.samples[index]
        
        return {
            'index': index,
            'feat_len': feat_len,
            'token_len': token_len,
            'feat_shape': features.shape,
            'token_shape': tokens.shape,
            'alignment_ratio': feat_len / token_len,
            'feature_stats': {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'min': features.min().item(),
                'max': features.max().item()
            }
        }
    
    def visualize_sample(self, index: int, save_path: Optional[str] = None):
        """Visualize a sample (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        features, tokens, feat_len, token_len = self.samples[index]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot features
        ax1.imshow(features[:feat_len].T, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title(f'Features (length={feat_len})')
        ax1.set_xlabel('Time frames')
        ax1.set_ylabel('Feature dimension')
        
        # Plot tokens
        ax2.plot(tokens[:token_len].numpy(), 'o-')
        ax2.set_title(f'Tokens (length={token_len})')
        ax2.set_xlabel('Token position')
        ax2.set_ylabel('Token ID')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def create_toy_datasets(
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    **kwargs
) -> Tuple[ToySeq2SeqDataset, ToySeq2SeqDataset, ToySeq2SeqDataset]:
    """
    Create train/val/test splits of toy dataset.
    
    Args:
        train_size: Number of training samples
        val_size: Number of validation samples  
        test_size: Number of test samples
        **kwargs: Additional arguments for ToySeq2SeqDataset
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    train_dataset = ToySeq2SeqDataset(num_samples=train_size, seed=42, **kwargs)
    val_dataset = ToySeq2SeqDataset(num_samples=val_size, seed=1337, **kwargs)
    test_dataset = ToySeq2SeqDataset(num_samples=test_size, seed=9999, **kwargs)
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Example usage and testing
    print("Creating toy dataset...")
    dataset = ToySeq2SeqDataset(num_samples=100, complexity='medium')
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a few samples
    for i in range(3):
        features, tokens, feat_len, token_len = dataset[i]
        info = dataset.get_sample_info(i)
        
        print(f"\nSample {i}:")
        print(f"  Features: {features.shape} (actual length: {feat_len})")
        print(f"  Tokens: {tokens.shape} (actual length: {token_len})")
        print(f"  Alignment ratio: {info['alignment_ratio']:.2f}")
        print(f"  Feature stats: mean={info['feature_stats']['mean']:.3f}, "
              f"std={info['feature_stats']['std']:.3f}")
    
    print("\nDataset creation successful!")