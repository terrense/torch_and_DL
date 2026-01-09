"""
Sequence utilities for Paraformer ASR data loading.

Includes collation functions, padding utilities, and mask generation
for variable-length sequences.
"""

import torch
from typing import List, Tuple, Dict, Any, Optional
from torch.nn.utils.rnn import pad_sequence


def create_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Create padding mask for variable-length sequences.
    
    Args:
        lengths: [batch_size] actual sequence lengths
        max_len: maximum sequence length (if None, use max of lengths)
        
    Returns:
        mask: [batch_size, max_len] boolean mask (True for valid positions)
    """
    if max_len is None:
        max_len = lengths.max().item()
    
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    
    return mask


def create_causal_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create causal (lower triangular) mask for autoregressive decoding.
    
    Args:
        size: sequence length
        device: torch device
        
    Returns:
        mask: [size, size] boolean mask (True for allowed positions)
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask


def create_attention_mask(
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    causal: bool = False
) -> torch.Tensor:
    """
    Create attention mask for cross-attention or self-attention.
    
    Args:
        query_lengths: [batch_size] query sequence lengths
        key_lengths: [batch_size] key sequence lengths  
        causal: whether to apply causal masking
        
    Returns:
        mask: [batch_size, max_query_len, max_key_len] attention mask
    """
    batch_size = query_lengths.size(0)
    max_query_len = query_lengths.max().item()
    max_key_len = key_lengths.max().item()
    
    # Create padding masks
    query_mask = create_padding_mask(query_lengths, max_query_len)  # [B, Q]
    key_mask = create_padding_mask(key_lengths, max_key_len)  # [B, K]
    
    # Combine masks: [B, Q, K]
    attention_mask = query_mask.unsqueeze(2) & key_mask.unsqueeze(1)
    
    # Apply causal mask if needed (for self-attention)
    if causal:
        assert max_query_len == max_key_len, "Causal mask requires same query/key lengths"
        causal_mask = create_causal_mask(max_query_len, device=query_lengths.device)
        attention_mask = attention_mask & causal_mask.unsqueeze(0)
    
    return attention_mask


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch: List of (features, tokens, feat_len, token_len) tuples
        
    Returns:
        batch_dict: Dictionary with padded tensors and masks
    """
    features_list = []
    tokens_list = []
    feat_lengths = []
    token_lengths = []
    
    # Separate batch components
    for features, tokens, feat_len, token_len in batch:
        features_list.append(features)
        tokens_list.append(tokens)
        feat_lengths.append(feat_len)
        token_lengths.append(token_len)
    
    # Convert lengths to tensors
    feat_lengths = torch.tensor(feat_lengths, dtype=torch.long)
    token_lengths = torch.tensor(token_lengths, dtype=torch.long)
    
    # Pad sequences
    # Features: pad to [batch_size, max_feat_len, feature_dim]
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    
    # Tokens: pad to [batch_size, max_token_len]
    tokens_padded = pad_sequence(tokens_list, batch_first=True, padding_value=0)
    
    # Create masks
    feat_mask = create_padding_mask(feat_lengths, features_padded.size(1))
    token_mask = create_padding_mask(token_lengths, tokens_padded.size(1))
    
    return {
        'features': features_padded,
        'tokens': tokens_padded,
        'feat_lengths': feat_lengths,
        'token_lengths': token_lengths,
        'feat_mask': feat_mask,
        'token_mask': token_mask
    }


def collate_inference_fn(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Collate function for inference (features only).
    
    Args:
        batch: List of feature tensors
        
    Returns:
        batch_dict: Dictionary with padded features and masks
    """
    feat_lengths = torch.tensor([len(features) for features in batch], dtype=torch.long)
    
    # Pad features
    features_padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
    
    # Create mask
    feat_mask = create_padding_mask(feat_lengths, features_padded.size(1))
    
    return {
        'features': features_padded,
        'feat_lengths': feat_lengths,
        'feat_mask': feat_mask
    }


def apply_mask(tensor: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """
    Apply mask to tensor by setting masked positions to fill_value.
    
    Args:
        tensor: [..., seq_len, ...] tensor to mask
        mask: [..., seq_len] boolean mask (True for valid positions)
        fill_value: value to fill masked positions
        
    Returns:
        masked_tensor: tensor with masked positions filled
    """
    # Expand mask to match tensor dimensions
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    
    mask = mask.expand_as(tensor)
    
    return tensor.masked_fill(~mask, fill_value)


def truncate_or_pad_sequence(
    sequence: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
    truncate_side: str = 'right'
) -> torch.Tensor:
    """
    Truncate or pad sequence to target length.
    
    Args:
        sequence: [seq_len, ...] input sequence
        target_length: desired sequence length
        pad_value: value for padding
        truncate_side: 'left' or 'right' for truncation
        
    Returns:
        sequence: [target_length, ...] processed sequence
    """
    current_length = sequence.size(0)
    
    if current_length == target_length:
        return sequence
    elif current_length > target_length:
        # Truncate
        if truncate_side == 'right':
            return sequence[:target_length]
        else:  # left
            return sequence[-target_length:]
    else:
        # Pad
        pad_length = target_length - current_length
        pad_shape = (pad_length,) + sequence.shape[1:]
        padding = torch.full(pad_shape, pad_value, dtype=sequence.dtype, device=sequence.device)
        
        if truncate_side == 'right':
            return torch.cat([sequence, padding], dim=0)
        else:  # left
            return torch.cat([padding, sequence], dim=0)


def compute_sequence_stats(lengths: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for sequence lengths.
    
    Args:
        lengths: [batch_size] sequence lengths
        
    Returns:
        stats: dictionary with length statistics
    """
    lengths_float = lengths.float()
    
    return {
        'mean': lengths_float.mean().item(),
        'std': lengths_float.std().item(),
        'min': lengths.min().item(),
        'max': lengths.max().item(),
        'median': lengths.median().item()
    }


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Move batch tensors to specified device.
    
    Args:
        batch: batch dictionary
        device: target device
        
    Returns:
        batch: batch with tensors moved to device
    """
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value 
            for key, value in batch.items()}


def validate_batch(batch: Dict[str, torch.Tensor]) -> bool:
    """
    Validate batch consistency.
    
    Args:
        batch: batch dictionary
        
    Returns:
        valid: whether batch is valid
    """
    try:
        batch_size = batch['features'].size(0)
        
        # Check all tensors have same batch size
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                if tensor.size(0) != batch_size:
                    print(f"Batch size mismatch for {key}: {tensor.size(0)} vs {batch_size}")
                    return False
        
        # Check mask shapes match sequence shapes
        if 'feat_mask' in batch:
            feat_seq_len = batch['features'].size(1)
            mask_seq_len = batch['feat_mask'].size(1)
            if feat_seq_len != mask_seq_len:
                print(f"Feature mask shape mismatch: {mask_seq_len} vs {feat_seq_len}")
                return False
        
        if 'token_mask' in batch:
            token_seq_len = batch['tokens'].size(1)
            mask_seq_len = batch['token_mask'].size(1)
            if token_seq_len != mask_seq_len:
                print(f"Token mask shape mismatch: {mask_seq_len} vs {token_seq_len}")
                return False
        
        # Check lengths are consistent
        if 'feat_lengths' in batch and 'feat_mask' in batch:
            max_feat_len = batch['feat_lengths'].max().item()
            mask_len = batch['feat_mask'].size(1)
            if max_feat_len > mask_len:
                print(f"Feature length exceeds mask: {max_feat_len} vs {mask_len}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Batch validation error: {e}")
        return False


if __name__ == "__main__":
    # Example usage and testing
    print("Testing sequence utilities...")
    
    # Test padding mask
    lengths = torch.tensor([5, 3, 7, 2])
    mask = create_padding_mask(lengths, max_len=8)
    print(f"Padding mask shape: {mask.shape}")
    print(f"Mask:\n{mask}")
    
    # Test causal mask
    causal_mask = create_causal_mask(4)
    print(f"\nCausal mask:\n{causal_mask}")
    
    # Test attention mask
    query_lengths = torch.tensor([3, 2])
    key_lengths = torch.tensor([4, 3])
    attn_mask = create_attention_mask(query_lengths, key_lengths)
    print(f"\nAttention mask shape: {attn_mask.shape}")
    
    # Test collation with dummy data
    print("\nTesting collation...")
    batch = [
        (torch.randn(5, 10), torch.randint(0, 50, (3,)), 5, 3),
        (torch.randn(7, 10), torch.randint(0, 50, (4,)), 7, 4),
        (torch.randn(3, 10), torch.randint(0, 50, (2,)), 3, 2)
    ]
    
    collated = collate_fn(batch)
    print(f"Collated features shape: {collated['features'].shape}")
    print(f"Collated tokens shape: {collated['tokens'].shape}")
    print(f"Feature lengths: {collated['feat_lengths']}")
    print(f"Token lengths: {collated['token_lengths']}")
    
    # Validate batch
    is_valid = validate_batch(collated)
    print(f"Batch is valid: {is_valid}")
    
    print("\nSequence utilities testing successful!")