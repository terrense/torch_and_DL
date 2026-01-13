"""
Greedy decoding system for Paraformer ASR.

Implements greedy decoding from logits with proper masking,
tokenizer integration for text generation, and complete
inference pipeline from features to decoded text.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Dict, Any
import logging

from ..data.tokenizer import CharTokenizer
from ..utils.tensor_utils import assert_shape, check_nan_inf

logger = logging.getLogger(__name__)


class GreedyDecoder:
    """
    Greedy decoder for sequence generation from logits.
    
    Implements greedy decoding with proper handling of special tokens,
    sequence length limits, and early stopping conditions.
    """
    
    def __init__(
        self,
        tokenizer: CharTokenizer,
        max_length: int = 200,
        temperature: float = 1.0,
        min_length: int = 1
    ):
        """
        Initialize greedy decoder.
        
        Args:
            tokenizer: Character tokenizer for text conversion
            max_length: Maximum generation length
            temperature: Sampling temperature (1.0 = no scaling)
            min_length: Minimum generation length before EOS
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.min_length = min_length
        
        # Special token IDs
        self.pad_token_id = tokenizer.pad_token_id
        self.sos_token_id = tokenizer.sos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.unk_token_id = tokenizer.unk_token_id
    
    def decode_logits(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode token sequences from logits using greedy selection.
        
        Args:
            logits: [B, S, vocab_size] model output logits
            lengths: [B] actual sequence lengths (optional)
            return_scores: Whether to return confidence scores
            
        Returns:
            tokens: [B, S] decoded token sequences
            scores: [B, S] confidence scores (if return_scores=True)
        """
        # Input validation
        assert_shape(logits, "B,S,V", "logits")
        check_nan_inf(logits, "logits")
        
        B, S, V = logits.shape
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Get probabilities and predictions
        probs = F.softmax(logits, dim=-1)  # [B, S, V]
        tokens = torch.argmax(logits, dim=-1)  # [B, S]
        
        # Get confidence scores (max probability)
        scores = torch.gather(probs, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)  # [B, S]
        
        # Apply length masking if provided
        if lengths is not None:
            assert_shape(lengths, "B", "lengths")
            mask = torch.arange(S, device=tokens.device)[None, :] >= lengths[:, None]
            tokens = tokens.masked_fill(mask, self.pad_token_id)
            scores = scores.masked_fill(mask, 0.0)
        
        if return_scores:
            return tokens, scores
        else:
            return tokens
    
    def generate_step_by_step(
        self,
        model: torch.nn.Module,
        encoder_features: torch.Tensor,
        predictor_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        do_sample: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences step by step using the model.
        
        Args:
            model: Paraformer model with decoder
            encoder_features: [B, T, D] encoded audio features
            predictor_output: [B, T, 1] predictor alignment output
            encoder_mask: [B, T] encoder padding mask
            max_length: Maximum generation length
            do_sample: Whether to use sampling instead of greedy
            
        Returns:
            generated_tokens: [B, S] generated token sequences
            generated_lengths: [B] actual generation lengths
        """
        # Input validation
        assert_shape(encoder_features, "B,T,D", "encoder_features")
        assert_shape(predictor_output, "B,T,1", "predictor_output")
        
        B, T, D = encoder_features.shape
        max_len = max_length or self.max_length
        
        device = encoder_features.device
        
        # Initialize generation
        generated_tokens = torch.full(
            (B, max_len), 
            self.pad_token_id, 
            dtype=torch.long, 
            device=device
        )
        generated_lengths = torch.zeros(B, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Start with SOS token
        generated_tokens[:, 0] = self.sos_token_id
        current_length = 1
        
        model.eval()
        with torch.no_grad():
            for step in range(1, max_len):
                # Get current sequence
                current_tokens = generated_tokens[:, :current_length]
                
                # Forward pass through decoder
                decoder_output = model.decoder(
                    encoder_features=encoder_features,
                    predictor_output=predictor_output,
                    target_tokens=current_tokens,
                    encoder_mask=encoder_mask,
                    target_lengths=None  # Let decoder handle variable lengths
                )
                
                # Get logits for next token
                if isinstance(decoder_output, tuple):
                    logits, _ = decoder_output
                else:
                    logits = decoder_output
                
                next_token_logits = logits[:, -1, :]  # [B, V]
                
                # Apply temperature
                if self.temperature != 1.0:
                    next_token_logits = next_token_logits / self.temperature
                
                # Generate next token
                if do_sample:
                    # Sampling
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    # Greedy
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Update sequences
                generated_tokens[:, current_length] = next_tokens
                current_length += 1
                
                # Check for EOS tokens
                is_eos = (next_tokens == self.eos_token_id)
                is_min_length = (current_length >= self.min_length)
                can_finish = is_eos & is_min_length & ~finished
                
                # Update finished sequences
                finished = finished | can_finish
                generated_lengths = torch.where(
                    can_finish,
                    torch.tensor(current_length, device=device),
                    generated_lengths
                )
                
                # Stop if all sequences are finished
                if finished.all():
                    break
        
        # Set lengths for unfinished sequences
        unfinished_mask = ~finished
        generated_lengths = torch.where(
            unfinished_mask,
            torch.tensor(current_length, device=device),
            generated_lengths
        )
        
        return generated_tokens, generated_lengths
    
    def decode_to_text(
        self,
        tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Convert token sequences to text strings.
        
        Args:
            tokens: [B, S] token sequences
            lengths: [B] actual sequence lengths
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            texts: List of decoded text strings
        """
        # Input validation
        assert_shape(tokens, "B,S", "tokens")
        
        B, S = tokens.shape
        
        if lengths is not None:
            assert_shape(lengths, "B", "lengths")
        
        texts = []
        for i in range(B):
            # Get sequence
            if lengths is not None:
                seq_tokens = tokens[i, :lengths[i]]
            else:
                seq_tokens = tokens[i]
            
            # Decode to text
            text = self.tokenizer.decode(seq_tokens, skip_special_tokens=skip_special_tokens)
            texts.append(text)
        
        return texts


def greedy_decode(
    logits: torch.Tensor,
    tokenizer: CharTokenizer,
    lengths: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    return_scores: bool = False
) -> Union[List[str], Tuple[List[str], torch.Tensor]]:
    """
    Convenience function for greedy decoding from logits to text.
    
    Args:
        logits: [B, S, vocab_size] model output logits
        tokenizer: Character tokenizer
        lengths: [B] actual sequence lengths
        temperature: Sampling temperature
        return_scores: Whether to return confidence scores
        
    Returns:
        texts: List of decoded text strings
        scores: [B, S] confidence scores (if return_scores=True)
    """
    decoder = GreedyDecoder(tokenizer, temperature=temperature)
    
    if return_scores:
        tokens, scores = decoder.decode_logits(logits, lengths, return_scores=True)
        texts = decoder.decode_to_text(tokens, lengths)
        return texts, scores
    else:
        tokens = decoder.decode_logits(logits, lengths, return_scores=False)
        texts = decoder.decode_to_text(tokens, lengths)
        return texts


class InferencePipeline:
    """
    Complete inference pipeline from audio features to decoded text.
    
    Integrates model forward pass, greedy decoding, and text generation
    into a single convenient interface.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: CharTokenizer,
        max_length: int = 200,
        temperature: float = 1.0,
        do_sample: bool = False
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained Paraformer ASR model
            tokenizer: Character tokenizer
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling instead of greedy
        """
        self.model = model
        self.tokenizer = tokenizer
        self.decoder = GreedyDecoder(
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature
        )
        self.do_sample = do_sample
        
        # Set model to eval mode
        self.model.eval()
    
    def __call__(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
        return_alignment: bool = False,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete inference pipeline.
        
        Args:
            features: [B, T, F] input audio features
            feature_lengths: [B] feature sequence lengths
            return_alignment: Whether to return predictor alignment
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing:
            - texts: List of decoded text strings
            - tokens: [B, S] generated token sequences
            - lengths: [B] generation lengths
            - alignment: [B, T, 1] predictor alignment (if requested)
            - confidence: [B, S] confidence scores (if requested)
        """
        with torch.no_grad():
            # 1. Forward pass through model
            if hasattr(self.model, 'generate'):
                # Use model's built-in generation
                if return_alignment:
                    generated_tokens, alignment = self.model.generate(
                        features=features,
                        feature_lengths=feature_lengths,
                        max_length=self.decoder.max_length,
                        temperature=self.decoder.temperature,
                        do_sample=self.do_sample,
                        return_alignment=True
                    )
                else:
                    generated_tokens = self.model.generate(
                        features=features,
                        feature_lengths=feature_lengths,
                        max_length=self.decoder.max_length,
                        temperature=self.decoder.temperature,
                        do_sample=self.do_sample,
                        return_alignment=False
                    )
                    alignment = None
                
                # Compute lengths (find EOS tokens)
                B, S = generated_tokens.shape
                lengths = torch.full((B,), S, dtype=torch.long, device=generated_tokens.device)
                
                for i in range(B):
                    eos_positions = (generated_tokens[i] == self.tokenizer.eos_token_id).nonzero()
                    if len(eos_positions) > 0:
                        lengths[i] = eos_positions[0].item() + 1
            
            else:
                # Manual generation using decoder
                model_outputs = self.model(
                    features=features,
                    feature_lengths=feature_lengths,
                    return_predictor_output=True
                )
                
                encoder_features = model_outputs['encoder_features']
                predictor_output = model_outputs['predictor_probabilities']
                encoder_mask = model_outputs['padding_mask']
                
                generated_tokens, lengths = self.decoder.generate_step_by_step(
                    model=self.model,
                    encoder_features=encoder_features,
                    predictor_output=predictor_output,
                    encoder_mask=encoder_mask,
                    do_sample=self.do_sample
                )
                
                alignment = predictor_output if return_alignment else None
            
            # 2. Decode to text
            texts = self.decoder.decode_to_text(generated_tokens, lengths)
            
            # 3. Compute confidence scores if requested
            confidence = None
            if return_confidence:
                # Re-run model to get logits for confidence calculation
                # This is a simplified approach - in practice, you might want to
                # store logits during generation for efficiency
                model_outputs = self.model(
                    features=features,
                    feature_lengths=feature_lengths,
                    target_tokens=generated_tokens,
                    target_lengths=lengths
                )
                
                if 'logits' in model_outputs:
                    logits = model_outputs['logits']
                    _, confidence = self.decoder.decode_logits(
                        logits, lengths, return_scores=True
                    )
        
        # Prepare results
        results = {
            'texts': texts,
            'tokens': generated_tokens,
            'lengths': lengths
        }
        
        if return_alignment and alignment is not None:
            results['alignment'] = alignment
        
        if return_confidence and confidence is not None:
            results['confidence'] = confidence
        
        return results
    
    def infer_single(
        self,
        features: torch.Tensor,
        return_alignment: bool = False
    ) -> Dict[str, Any]:
        """
        Convenience method for single sequence inference.
        
        Args:
            features: [T, F] single sequence features
            return_alignment: Whether to return alignment
            
        Returns:
            Dictionary with single sequence results
        """
        # Add batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, T, F]
        
        # Run inference
        results = self(features, return_alignment=return_alignment)
        
        # Remove batch dimension from results
        return {
            'text': results['texts'][0],
            'tokens': results['tokens'][0],
            'length': results['lengths'][0],
            'alignment': results.get('alignment', [None])[0] if return_alignment else None
        }


def create_inference_pipeline(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    max_length: int = 200,
    temperature: float = 1.0,
    do_sample: bool = False
) -> InferencePipeline:
    """
    Create an inference pipeline with default settings.
    
    Args:
        model: Trained Paraformer ASR model
        tokenizer: Character tokenizer
        max_length: Maximum generation length
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        pipeline: Configured InferencePipeline instance
    """
    return InferencePipeline(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature,
        do_sample=do_sample
    )


if __name__ == "__main__":
    # Test the greedy decoder
    print("Testing greedy decoding system...")
    
    # Create test tokenizer
    from ..data.tokenizer import CharTokenizer
    tokenizer = CharTokenizer(vocab_size=50)
    
    # Test parameters
    B, S, V = 2, 15, tokenizer.get_vocab_size()
    
    # Create test logits
    logits = torch.randn(B, S, V)
    lengths = torch.tensor([12, 10])
    
    # Test greedy decoding
    print("\n1. Testing greedy decoding from logits:")
    texts = greedy_decode(logits, tokenizer, lengths)
    for i, text in enumerate(texts):
        print(f"   Sequence {i}: '{text}'")
    
    # Test with confidence scores
    print("\n2. Testing with confidence scores:")
    texts, scores = greedy_decode(logits, tokenizer, lengths, return_scores=True)
    for i, (text, score_seq) in enumerate(zip(texts, scores)):
        avg_confidence = score_seq[:lengths[i]].mean().item()
        print(f"   Sequence {i}: '{text}' (avg confidence: {avg_confidence:.3f})")
    
    # Test GreedyDecoder class
    print("\n3. Testing GreedyDecoder class:")
    decoder = GreedyDecoder(tokenizer, max_length=20, temperature=1.0)
    
    tokens = decoder.decode_logits(logits, lengths)
    decoded_texts = decoder.decode_to_text(tokens, lengths)
    
    for i, text in enumerate(decoded_texts):
        print(f"   Decoded {i}: '{text}'")
    
    print("\nGreedy decoding tests passed!")