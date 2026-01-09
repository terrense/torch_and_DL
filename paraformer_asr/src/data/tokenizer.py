"""
Character-level tokenizer for Paraformer ASR.

Implements vocabulary management, encoding/decoding utilities,
and proper handling of special tokens.
"""

import torch
from typing import List, Dict, Optional, Union, Tuple
import json
import os


class CharTokenizer:
    """
    Character-level tokenizer with special token support.
    
    Handles encoding text to token IDs and decoding back to text,
    with proper management of padding, start/end tokens, and unknown characters.
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        pad_token: str = '<PAD>',
        unk_token: str = '<UNK>',
        sos_token: str = '<SOS>',
        eos_token: str = '<EOS>',
        characters: Optional[str] = None
    ):
        """
        Initialize character tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            pad_token: Padding token
            unk_token: Unknown token for out-of-vocabulary characters
            sos_token: Start of sequence token
            eos_token: End of sequence token
            characters: Custom character set (if None, uses common characters)
        """
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        # Build vocabulary
        self._build_vocabulary(characters)
        
        # Special token IDs
        self.pad_token_id = self.char_to_id[pad_token]
        self.unk_token_id = self.char_to_id[unk_token]
        self.sos_token_id = self.char_to_id[sos_token]
        self.eos_token_id = self.char_to_id[eos_token]
    
    def _build_vocabulary(self, characters: Optional[str] = None):
        """Build character vocabulary."""
        # Start with special tokens
        vocab = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        
        if characters is None:
            # Default character set: lowercase letters, digits, space, punctuation
            characters = (
                'abcdefghijklmnopqrstuvwxyz'
                '0123456789'
                ' .,!?;:-\'\"()[]{}/@#$%^&*+=_~`|\\<>'
            )
        
        # Add characters up to vocab_size
        for char in characters:
            if char not in vocab and len(vocab) < self.vocab_size:
                vocab.append(char)
        
        # Fill remaining slots with generated characters if needed
        while len(vocab) < self.vocab_size:
            vocab.append(f'<CHAR_{len(vocab)}>')
        
        # Create mappings
        self.char_to_id = {char: i for i, char in enumerate(vocab)}
        self.id_to_char = {i: char for i, char in enumerate(vocab)}
        self.vocab = vocab
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add SOS/EOS tokens
            max_length: Maximum sequence length (truncate if longer)
            
        Returns:
            tokens: [seq_len] tensor of token IDs
        """
        # Convert characters to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.sos_token_id)
        
        for char in text:
            token_id = self.char_to_id.get(char, self.unk_token_id)
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        # Truncate if necessary
        if max_length is not None and len(token_ids) > max_length:
            if add_special_tokens:
                # Keep SOS, truncate middle, keep EOS
                token_ids = [token_ids[0]] + token_ids[1:max_length-1] + [token_ids[-1]]
            else:
                token_ids = token_ids[:max_length]
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(
        self, 
        tokens: Union[torch.Tensor, List[int]], 
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: Token IDs as tensor or list
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            text: Decoded text string
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        characters = []
        for token_id in tokens:
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and char in [
                    self.pad_token, self.unk_token, self.sos_token, self.eos_token
                ]:
                    continue
                
                characters.append(char)
        
        return ''.join(characters)
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_lengths: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
            add_special_tokens: Whether to add SOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences to same length
            return_lengths: Whether to return actual lengths
            
        Returns:
            tokens: [batch_size, seq_len] padded token sequences
            lengths: [batch_size] actual sequence lengths (if return_lengths=True)
        """
        # Encode all texts
        encoded_texts = []
        lengths = []
        
        for text in texts:
            tokens = self.encode(text, add_special_tokens, max_length)
            encoded_texts.append(tokens)
            lengths.append(len(tokens))
        
        if not padding:
            if return_lengths:
                return encoded_texts, torch.tensor(lengths, dtype=torch.long)
            return encoded_texts
        
        # Pad to same length
        if max_length is None:
            max_length = max(lengths)
        
        batch_tokens = torch.full(
            (len(texts), max_length), 
            self.pad_token_id, 
            dtype=torch.long
        )
        
        for i, tokens in enumerate(encoded_texts):
            seq_len = min(len(tokens), max_length)
            batch_tokens[i, :seq_len] = tokens[:seq_len]
        
        if return_lengths:
            return batch_tokens, torch.tensor(lengths, dtype=torch.long)
        
        return batch_tokens
    
    def batch_decode(
        self,
        batch_tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token sequences.
        
        Args:
            batch_tokens: [batch_size, seq_len] token sequences
            lengths: [batch_size] actual sequence lengths
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            texts: List of decoded text strings
        """
        texts = []
        
        for i in range(batch_tokens.size(0)):
            tokens = batch_tokens[i]
            
            # Use actual length if provided
            if lengths is not None:
                tokens = tokens[:lengths[i]]
            
            text = self.decode(tokens, skip_special_tokens)
            texts.append(text)
        
        return texts
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'sos_token_id': self.sos_token_id,
            'eos_token_id': self.eos_token_id
        }
    
    def save_vocabulary(self, path: str):
        """Save vocabulary to file."""
        vocab_data = {
            'vocab': self.vocab,
            'char_to_id': self.char_to_id,
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'sos_token': self.sos_token,
                'eos_token': self.eos_token
            }
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_vocabulary(cls, path: str) -> 'CharTokenizer':
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Create tokenizer with loaded vocabulary
        tokenizer = cls.__new__(cls)
        tokenizer.vocab = vocab_data['vocab']
        tokenizer.char_to_id = vocab_data['char_to_id']
        tokenizer.id_to_char = {int(k): v for k, v in vocab_data['char_to_id'].items()}
        
        # Set special tokens
        special_tokens = vocab_data['special_tokens']
        tokenizer.pad_token = special_tokens['pad_token']
        tokenizer.unk_token = special_tokens['unk_token']
        tokenizer.sos_token = special_tokens['sos_token']
        tokenizer.eos_token = special_tokens['eos_token']
        
        # Set special token IDs
        tokenizer.pad_token_id = tokenizer.char_to_id[tokenizer.pad_token]
        tokenizer.unk_token_id = tokenizer.char_to_id[tokenizer.unk_token]
        tokenizer.sos_token_id = tokenizer.char_to_id[tokenizer.sos_token]
        tokenizer.eos_token_id = tokenizer.char_to_id[tokenizer.eos_token]
        
        tokenizer.vocab_size = len(tokenizer.vocab)
        
        return tokenizer
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


def create_default_tokenizer(vocab_size: int = 100) -> CharTokenizer:
    """Create a default character tokenizer."""
    return CharTokenizer(vocab_size=vocab_size)


if __name__ == "__main__":
    # Example usage and testing
    print("Creating character tokenizer...")
    tokenizer = CharTokenizer(vocab_size=50)
    
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.get_special_tokens()}")
    
    # Test encoding/decoding
    test_texts = [
        "hello world",
        "this is a test",
        "123 abc!",
        ""
    ]
    
    print("\nTesting single text encoding/decoding:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"  '{text}' -> {tokens.tolist()} -> '{decoded}'")
    
    print("\nTesting batch encoding/decoding:")
    batch_tokens, lengths = tokenizer.batch_encode(
        test_texts, 
        return_lengths=True,
        padding=True
    )
    print(f"Batch tokens shape: {batch_tokens.shape}")
    print(f"Lengths: {lengths.tolist()}")
    
    decoded_texts = tokenizer.batch_decode(batch_tokens, lengths)
    for orig, decoded in zip(test_texts, decoded_texts):
        print(f"  '{orig}' -> '{decoded}'")
    
    print("\nTokenizer testing successful!")