"""
Inference utilities for Paraformer ASR.

Provides inference interface for feature sequences with text output,
model loading utilities, and optional FastAPI service for JSON input/output.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import time

from ..models.paraformer import ParaformerASR
from ..data.tokenizer import CharTokenizer
from ..decode.greedy import create_inference_pipeline, InferencePipeline
from ..utils.tensor_utils import assert_shape

logger = logging.getLogger(__name__)


class ASRInference:
    """
    High-level inference interface for Paraformer ASR.
    
    Provides convenient methods for running inference on audio features
    with automatic preprocessing, postprocessing, and result formatting.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: CharTokenizer,
        device: torch.device = torch.device('cpu'),
        max_length: int = 200,
        temperature: float = 1.0,
        do_sample: bool = False
    ):
        """
        Initialize ASR inference.
        
        Args:
            model: Trained Paraformer ASR model
            tokenizer: Character tokenizer
            device: Inference device
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling instead of greedy
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        # Create inference pipeline
        self.pipeline = create_inference_pipeline(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample
        )
        
        logger.info(f"ASR inference initialized on {device}")
    
    def infer_features(
        self,
        features: Union[torch.Tensor, np.ndarray],
        feature_length: Optional[int] = None,
        return_alignment: bool = False,
        return_confidence: bool = False,
        return_tokens: bool = False
    ) -> Dict[str, Any]:
        """
        Run inference on audio features.
        
        Args:
            features: [T, F] or [1, T, F] audio features
            feature_length: Actual feature length (optional)
            return_alignment: Whether to return predictor alignment
            return_confidence: Whether to return confidence scores
            return_tokens: Whether to return token sequences
            
        Returns:
            Dictionary with inference results
        """
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Ensure correct shape
        if features.dim() == 2:
            features = features.unsqueeze(0)  # Add batch dimension
        elif features.dim() != 3:
            raise ValueError(f"Features must be 2D or 3D, got {features.dim()}D")
        
        # Move to device
        features = features.to(self.device)
        
        # Prepare feature lengths
        feature_lengths = None
        if feature_length is not None:
            feature_lengths = torch.tensor([feature_length], device=self.device)
        
        # Run inference
        with torch.no_grad():
            results = self.pipeline(
                features=features,
                feature_lengths=feature_lengths,
                return_alignment=return_alignment,
                return_confidence=return_confidence
            )
        
        # Format output
        output = {
            'text': results['texts'][0],
            'inference_time': time.time()  # Placeholder for timing
        }
        
        if return_tokens:
            output['tokens'] = results['tokens'][0].cpu().tolist()
            output['token_length'] = results['lengths'][0].item()
        
        if return_alignment and 'alignment' in results:
            output['alignment'] = results['alignment'][0].cpu().numpy()
        
        if return_confidence and 'confidence' in results:
            confidence = results['confidence'][0, :results['lengths'][0]]
            output['confidence'] = confidence.cpu().numpy()
            output['avg_confidence'] = confidence.mean().item()
        
        return output
    
    def infer_batch(
        self,
        features_batch: Union[torch.Tensor, np.ndarray, List[torch.Tensor]],
        feature_lengths: Optional[List[int]] = None,
        return_alignment: bool = False,
        return_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of features.
        
        Args:
            features_batch: Batch of features [B, T, F] or list of [T, F]
            feature_lengths: List of actual feature lengths (optional)
            return_alignment: Whether to return predictor alignment
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of inference results for each sample
        """
        # Handle different input formats
        if isinstance(features_batch, list):
            # List of individual feature tensors
            max_length = max(f.shape[0] for f in features_batch)
            batch_size = len(features_batch)
            feature_dim = features_batch[0].shape[1]
            
            # Create padded batch tensor
            batch_tensor = torch.zeros(batch_size, max_length, feature_dim)
            batch_lengths = []
            
            for i, features in enumerate(features_batch):
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features).float()
                
                length = features.shape[0]
                batch_tensor[i, :length] = features
                batch_lengths.append(length)
            
            features_batch = batch_tensor
            if feature_lengths is None:
                feature_lengths = batch_lengths
        
        # Convert to tensor if needed
        if isinstance(features_batch, np.ndarray):
            features_batch = torch.from_numpy(features_batch).float()
        
        # Move to device
        features_batch = features_batch.to(self.device)
        
        # Prepare feature lengths tensor
        feature_lengths_tensor = None
        if feature_lengths is not None:
            feature_lengths_tensor = torch.tensor(feature_lengths, device=self.device)
        
        # Run batch inference
        with torch.no_grad():
            results = self.pipeline(
                features=features_batch,
                feature_lengths=feature_lengths_tensor,
                return_alignment=return_alignment,
                return_confidence=return_confidence
            )
        
        # Format outputs
        batch_results = []
        batch_size = features_batch.shape[0]
        
        for i in range(batch_size):
            output = {
                'text': results['texts'][i]
            }
            
            if return_alignment and 'alignment' in results:
                if feature_lengths is not None:
                    align_length = feature_lengths[i]
                    output['alignment'] = results['alignment'][i, :align_length].cpu().numpy()
                else:
                    output['alignment'] = results['alignment'][i].cpu().numpy()
            
            if return_confidence and 'confidence' in results:
                token_length = results['lengths'][i].item()
                confidence = results['confidence'][i, :token_length]
                output['confidence'] = confidence.cpu().numpy()
                output['avg_confidence'] = confidence.mean().item()
            
            batch_results.append(output)
        
        return batch_results
    
    def infer_from_dict(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference from dictionary input (useful for API).
        
        Args:
            input_dict: Dictionary with 'features' and optional parameters
            
        Returns:
            Dictionary with inference results
        """
        # Extract features
        features = input_dict['features']
        if isinstance(features, list):
            features = np.array(features)
        
        # Extract optional parameters
        feature_length = input_dict.get('feature_length')
        return_alignment = input_dict.get('return_alignment', False)
        return_confidence = input_dict.get('return_confidence', False)
        return_tokens = input_dict.get('return_tokens', False)
        
        return self.infer_features(
            features=features,
            feature_length=feature_length,
            return_alignment=return_alignment,
            return_confidence=return_confidence,
            return_tokens=return_tokens
        )


def create_inference_from_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
    max_length: int = 200,
    temperature: float = 1.0,
    do_sample: bool = False
) -> ASRInference:
    """
    Create inference instance from saved checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Inference device (auto-detect if None)
        max_length: Maximum generation length
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        inference: Configured ASRInference instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create tokenizer
    if 'tokenizer_vocab' in checkpoint:
        tokenizer = CharTokenizer(vocab_size=len(checkpoint['tokenizer_vocab']))
        tokenizer.vocab = checkpoint['tokenizer_vocab']
        tokenizer.char_to_id = {char: i for i, char in enumerate(tokenizer.vocab)}
        tokenizer.id_to_char = {i: char for i, char in enumerate(tokenizer.vocab)}
        
        # Set special token IDs
        tokenizer.pad_token_id = tokenizer.char_to_id.get('<PAD>', 0)
        tokenizer.sos_token_id = tokenizer.char_to_id.get('<SOS>', 1)
        tokenizer.eos_token_id = tokenizer.char_to_id.get('<EOS>', 2)
        tokenizer.unk_token_id = tokenizer.char_to_id.get('<UNK>', 3)
    else:
        # Create default tokenizer
        from ..data.tokenizer import create_default_tokenizer
        tokenizer = create_default_tokenizer()
        logger.warning("No tokenizer found in checkpoint, using default")
    
    # Create model
    if 'model_config' in checkpoint:
        from ..models.paraformer import create_paraformer_from_config
        model = create_paraformer_from_config(checkpoint['model_config'])
    else:
        # Try to infer from state dict
        state_dict = checkpoint['model_state_dict']
        vocab_size = len(tokenizer)
        
        # Basic model configuration
        model_config = {
            'input_dim': 80,  # Default
            'vocab_size': vocab_size,
            'encoder_dim': 512,
            'encoder_layers': 6,
            'decoder_layers': 2
        }
        
        from ..models.paraformer import create_paraformer_from_config
        model = create_paraformer_from_config(model_config)
        logger.warning("No model config found, using default configuration")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create inference instance
    inference = ASRInference(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        temperature=temperature,
        do_sample=do_sample
    )
    
    logger.info("Inference instance created successfully")
    
    return inference


# Optional FastAPI service
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional
    import uvicorn
    
    class InferenceRequest(BaseModel):
        features: List[List[float]]
        feature_length: Optional[int] = None
        return_alignment: bool = False
        return_confidence: bool = False
        return_tokens: bool = False
    
    class InferenceResponse(BaseModel):
        text: str
        tokens: Optional[List[int]] = None
        token_length: Optional[int] = None
        alignment: Optional[List[List[float]]] = None
        confidence: Optional[List[float]] = None
        avg_confidence: Optional[float] = None
        inference_time: Optional[float] = None
    
    def create_fastapi_service(
        inference: ASRInference,
        host: str = "0.0.0.0",
        port: int = 8000
    ) -> FastAPI:
        """
        Create FastAPI service for ASR inference.
        
        Args:
            inference: ASRInference instance
            host: Service host
            port: Service port
            
        Returns:
            app: FastAPI application
        """
        app = FastAPI(title="Paraformer ASR Inference Service")
        
        @app.post("/infer", response_model=InferenceResponse)
        async def infer_endpoint(request: InferenceRequest):
            try:
                # Convert request to inference input
                input_dict = {
                    'features': request.features,
                    'feature_length': request.feature_length,
                    'return_alignment': request.return_alignment,
                    'return_confidence': request.return_confidence,
                    'return_tokens': request.return_tokens
                }
                
                # Run inference
                result = inference.infer_from_dict(input_dict)
                
                # Convert numpy arrays to lists for JSON serialization
                if 'alignment' in result:
                    result['alignment'] = result['alignment'].tolist()
                if 'confidence' in result:
                    result['confidence'] = result['confidence'].tolist()
                
                return InferenceResponse(**result)
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy"}
        
        return app
    
    def run_fastapi_service(
        checkpoint_path: Path,
        host: str = "0.0.0.0",
        port: int = 8000,
        device: Optional[torch.device] = None
    ):
        """
        Run FastAPI service for ASR inference.
        
        Args:
            checkpoint_path: Path to model checkpoint
            host: Service host
            port: Service port
            device: Inference device
        """
        # Create inference instance
        inference = create_inference_from_checkpoint(checkpoint_path, device)
        
        # Create FastAPI app
        app = create_fastapi_service(inference, host, port)
        
        # Run service
        logger.info(f"Starting ASR inference service on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

except ImportError:
    logger.info("FastAPI not available, skipping service functionality")
    
    def create_fastapi_service(*args, **kwargs):
        raise ImportError("FastAPI is required for service functionality")
    
    def run_fastapi_service(*args, **kwargs):
        raise ImportError("FastAPI is required for service functionality")


if __name__ == "__main__":
    # Test inference functionality
    print("Testing ASR inference...")
    
    # This would require a trained model, so we'll just test imports
    print("ASR inference imports successful!")
    
    # Test WER/CER functions from evaluator
    from .evaluator import compute_wer, compute_cer
    
    predictions = ["hello world", "this is test"]
    references = ["hello world", "this is a test"]
    
    wer = compute_wer(predictions, references)
    cer = compute_cer(predictions, references)
    
    print(f"Test WER: {wer:.4f}")
    print(f"Test CER: {cer:.4f}")
    
    print("Inference testing completed!")