"""
Evaluation pipeline for Paraformer ASR.

Implements evaluation pipeline with token accuracy and loss calculation,
word error rate (WER) and character error rate (CER) metrics,
and comprehensive result reporting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import csv
import time
from collections import defaultdict
"""
Evaluation pipeline for Paraformer ASR.

Implements evaluation pipeline with token accuracy and loss calculation,
word error rate (WER) and character error rate (CER) metrics,
and comprehensive result reporting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import csv
import time
from collections import defaultdict

from ..losses.seq_loss import CombinedASRLoss, compute_token_accuracy
from ..decode.greedy import greedy_decode, create_inference_pipeline
from ..data.tokenizer import CharTokenizer
from ..utils.tensor_utils import assert_shape

logger = logging.getLogger(__name__)


def edit_distance(s1: List[str], s2: List[str]) -> int:
    """
    Compute edit distance between two sequences using dynamic programming.
    
    Args:
        s1: First sequence (list of strings)
        s2: Second sequence (list of strings)
        
    Returns:
        edit_distance: Minimum number of edits to transform s1 to s2
    """
    m, n = len(s1), len(s2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    return dp[m][n]


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate (WER).
    
    Args:
        predictions: List of predicted text strings
        references: List of reference text strings
        
    Returns:
        wer: Word Error Rate [0, 1]
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    total_words = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        # Split into words
        pred_words = pred.strip().split()
        ref_words = ref.strip().split()
        
        # Compute edit distance
        errors = edit_distance(pred_words, ref_words)
        
        total_errors += errors
        total_words += len(ref_words)
    
    if total_words == 0:
        return 0.0
    
    return total_errors / total_words


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate (CER).
    
    Args:
        predictions: List of predicted text strings
        references: List of reference text strings
        
    Returns:
        cer: Character Error Rate [0, 1]
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    total_chars = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        # Remove spaces for character-level comparison
        pred_chars = list(pred.replace(' ', ''))
        ref_chars = list(ref.replace(' ', ''))
        
        # Compute edit distance
        errors = edit_distance(pred_chars, ref_chars)
        
        total_errors += errors
        total_chars += len(ref_chars)
    
    if total_chars == 0:
        return 0.0
    
    return total_errors / total_chars


class ASREvaluator:
    """
    Comprehensive evaluator for ASR models.
    
    Provides evaluation pipeline with multiple metrics including
    token accuracy, sequence accuracy, WER, CER, and loss calculation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: CharTokenizer,
        loss_fn: Optional[CombinedASRLoss] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize ASR evaluator.
        
        Args:
            model: Trained ASR model
            tokenizer: Character tokenizer
            loss_fn: Loss function for loss calculation (optional)
            device: Evaluation device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        # Create inference pipeline
        self.inference_pipeline = create_inference_pipeline(
            model=model,
            tokenizer=tokenizer,
            max_length=200,
            temperature=1.0,
            do_sample=False
        )
    
    def evaluate_batch(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        target_tokens: torch.Tensor,
        target_lengths: torch.Tensor,
        predictor_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single batch.
        
        Args:
            features: [B, T, F] input features
            feature_lengths: [B] feature sequence lengths
            target_tokens: [B, S] target token sequences
            target_lengths: [B] target sequence lengths
            predictor_targets: [B, T] predictor targets (optional)
            
        Returns:
            Dictionary with batch evaluation results
        """
        with torch.no_grad():
            B, T, F = features.shape
            S = target_tokens.shape[1]
            
            # Forward pass for loss calculation
            loss_results = {}
            if self.loss_fn is not None:
                outputs = self.model(
                    features=features,
                    feature_lengths=feature_lengths,
                    target_tokens=target_tokens,
                    target_lengths=target_lengths,
                    return_predictor_output=True
                )
                
                loss_dict = self.loss_fn(
                    decoder_logits=outputs['logits'],
                    target_tokens=target_tokens,
                    target_lengths=target_lengths,
                    predictor_predictions=outputs.get('predictor_predictions'),
                    predictor_targets=predictor_targets,
                    feature_mask=~outputs['padding_mask']
                )
                
                loss_results = {
                    'loss': loss_dict['total_loss'].item(),
                    'decoder_loss': loss_dict['decoder_loss'].item(),
                    'predictor_loss': loss_dict.get('predictor_loss', torch.tensor(0.0)).item(),
                    'token_accuracy': loss_dict['token_accuracy'].item()
                }
            
            # Generate predictions
            inference_results = self.inference_pipeline(
                features=features,
                feature_lengths=feature_lengths,
                return_alignment=True,
                return_confidence=True
            )
            
            predicted_texts = inference_results['texts']
            predicted_tokens = inference_results['tokens']
            predicted_lengths = inference_results['lengths']
            confidence_scores = inference_results.get('confidence')
            
            # Decode target texts
            target_texts = self.tokenizer.batch_decode(
                target_tokens, target_lengths, skip_special_tokens=True
            )
            
            # Compute sequence accuracy
            sequence_matches = 0
            for pred, target in zip(predicted_texts, target_texts):
                if pred.strip() == target.strip():
                    sequence_matches += 1
            
            sequence_accuracy = sequence_matches / B
            
            # Compute WER and CER
            wer = compute_wer(predicted_texts, target_texts)
            cer = compute_cer(predicted_texts, target_texts)
            
            # Compute average confidence
            avg_confidence = 0.0
            if confidence_scores is not None:
                total_confidence = 0.0
                total_tokens = 0
                for i, length in enumerate(predicted_lengths):
                    if length > 0:
                        total_confidence += confidence_scores[i, :length].sum().item()
                        total_tokens += length.item()
                
                if total_tokens > 0:
                    avg_confidence = total_confidence / total_tokens
            
            # Prepare results
            results = {
                'batch_size': B,
                'sequence_accuracy': sequence_accuracy,
                'wer': wer,
                'cer': cer,
                'avg_confidence': avg_confidence,
                'predictions': predicted_texts,
                'targets': target_texts,
                'predicted_tokens': predicted_tokens.cpu(),
                'predicted_lengths': predicted_lengths.cpu()
            }
            
            # Add loss results if available
            results.update(loss_results)
            
            return results
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        save_predictions: bool = False,
        output_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate complete dataset.
        
        Args:
            dataloader: Data loader for evaluation
            max_batches: Maximum number of batches to evaluate (optional)
            save_predictions: Whether to save individual predictions
            output_file: File to save detailed results (optional)
            
        Returns:
            Dictionary with aggregated evaluation results
        """
        logger.info("Starting dataset evaluation...")
        
        # Initialize accumulators
        total_samples = 0
        total_loss = 0.0
        total_decoder_loss = 0.0
        total_predictor_loss = 0.0
        total_token_accuracy = 0.0
        total_sequence_accuracy = 0.0
        total_wer = 0.0
        total_cer = 0.0
        total_confidence = 0.0
        
        all_predictions = []
        all_targets = []
        detailed_results = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Move batch to device
                features = batch['features'].to(self.device)
                feature_lengths = batch['feature_lengths'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                token_lengths = batch['token_lengths'].to(self.device)
                
                predictor_targets = batch.get('predictor_targets')
                if predictor_targets is not None:
                    predictor_targets = predictor_targets.to(self.device)
                
                try:
                    # Evaluate batch
                    batch_results = self.evaluate_batch(
                        features=features,
                        feature_lengths=feature_lengths,
                        target_tokens=tokens,
                        target_lengths=token_lengths,
                        predictor_targets=predictor_targets
                    )
                    
                    # Accumulate metrics
                    batch_size = batch_results['batch_size']
                    total_samples += batch_size
                    
                    if 'loss' in batch_results:
                        total_loss += batch_results['loss'] * batch_size
                        total_decoder_loss += batch_results['decoder_loss'] * batch_size
                        total_predictor_loss += batch_results['predictor_loss'] * batch_size
                        total_token_accuracy += batch_results['token_accuracy'] * batch_size
                    
                    total_sequence_accuracy += batch_results['sequence_accuracy'] * batch_size
                    total_wer += batch_results['wer'] * batch_size
                    total_cer += batch_results['cer'] * batch_size
                    total_confidence += batch_results['avg_confidence'] * batch_size
                    
                    # Store predictions and targets
                    all_predictions.extend(batch_results['predictions'])
                    all_targets.extend(batch_results['targets'])
                    
                    # Store detailed results if requested
                    if save_predictions:
                        for i in range(batch_size):
                            detailed_results.append({
                                'batch_idx': batch_idx,
                                'sample_idx': i,
                                'prediction': batch_results['predictions'][i],
                                'target': batch_results['targets'][i],
                                'confidence': batch_results['avg_confidence']
                            })
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"Evaluated {batch_idx + 1} batches, {total_samples} samples")
                
                except Exception as e:
                    logger.error(f"Error evaluating batch {batch_idx}: {e}")
                    continue
        
        # Calculate final metrics
        if total_samples > 0:
            final_results = {
                'num_samples': total_samples,
                'avg_loss': total_loss / total_samples if total_loss > 0 else None,
                'avg_decoder_loss': total_decoder_loss / total_samples if total_decoder_loss > 0 else None,
                'avg_predictor_loss': total_predictor_loss / total_samples if total_predictor_loss > 0 else None,
                'avg_token_accuracy': total_token_accuracy / total_samples if total_token_accuracy > 0 else None,
                'sequence_accuracy': total_sequence_accuracy / total_samples,
                'wer': total_wer / total_samples,
                'cer': total_cer / total_samples,
                'avg_confidence': total_confidence / total_samples,
                'evaluation_time': time.time() - start_time
            }
            
            # Compute overall WER and CER on all predictions
            if all_predictions and all_targets:
                overall_wer = compute_wer(all_predictions, all_targets)
                overall_cer = compute_cer(all_predictions, all_targets)
                final_results['overall_wer'] = overall_wer
                final_results['overall_cer'] = overall_cer
        else:
            final_results = {
                'num_samples': 0,
                'error': 'No samples evaluated successfully'
            }
        
        # Save detailed results if requested
        if output_file is not None and detailed_results:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.suffix == '.json':
                with open(output_file, 'w') as f:
                    json.dump({
                        'summary': final_results,
                        'detailed_results': detailed_results
                    }, f, indent=2)
            elif output_file.suffix == '.csv':
                with open(output_file, 'w', newline='') as f:
                    if detailed_results:
                        writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
                        writer.writeheader()
                        writer.writerows(detailed_results)
        
        logger.info(f"Evaluation completed: {total_samples} samples in {final_results.get('evaluation_time', 0):.1f}s")
        
        return final_results
    
    def evaluate_samples(
        self,
        features_list: List[torch.Tensor],
        target_texts: List[str],
        feature_lengths: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate individual samples.
        
        Args:
            features_list: List of feature tensors [T, F]
            target_texts: List of target text strings
            feature_lengths: List of feature lengths (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        if len(features_list) != len(target_texts):
            raise ValueError("Features and targets must have same length")
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for i, (features, target_text) in enumerate(zip(features_list, target_texts)):
                # Add batch dimension
                if features.dim() == 2:
                    features = features.unsqueeze(0)  # [1, T, F]
                
                features = features.to(self.device)
                
                # Get feature length
                if feature_lengths is not None:
                    feat_len = torch.tensor([feature_lengths[i]], device=self.device)
                else:
                    feat_len = torch.tensor([features.shape[1]], device=self.device)
                
                # Run inference
                results = self.inference_pipeline(
                    features=features,
                    feature_lengths=feat_len,
                    return_confidence=True
                )
                
                predictions.append(results['texts'][0])
                
                if results.get('confidence') is not None:
                    conf = results['confidence'][0, :results['lengths'][0]].mean().item()
                    confidences.append(conf)
                else:
                    confidences.append(0.0)
        
        # Compute metrics
        wer = compute_wer(predictions, target_texts)
        cer = compute_cer(predictions, target_texts)
        
        sequence_matches = sum(1 for p, t in zip(predictions, target_texts) if p.strip() == t.strip())
        sequence_accuracy = sequence_matches / len(predictions)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'num_samples': len(predictions),
            'predictions': predictions,
            'targets': target_texts,
            'confidences': confidences,
            'sequence_accuracy': sequence_accuracy,
            'wer': wer,
            'cer': cer,
            'avg_confidence': avg_confidence
        }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer: CharTokenizer,
    loss_fn: Optional[CombinedASRLoss] = None,
    device: torch.device = torch.device('cpu'),
    max_batches: Optional[int] = None,
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Trained ASR model
        dataloader: Evaluation data loader
        tokenizer: Character tokenizer
        loss_fn: Loss function (optional)
        device: Evaluation device
        max_batches: Maximum batches to evaluate (optional)
        output_file: Output file for detailed results (optional)
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = ASREvaluator(
        model=model,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        device=device
    )
    
    return evaluator.evaluate_dataset(
        dataloader=dataloader,
        max_batches=max_batches,
        save_predictions=True,
        output_file=output_file
    )


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation functions...")
    
    # Test WER and CER
    predictions = ["hello world", "this is test", "good morning"]
    references = ["hello world", "this is a test", "good morning"]
    
    wer = compute_wer(predictions, references)
    cer = compute_cer(predictions, references)
    
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")
    
    print("Evaluation functions test completed!")