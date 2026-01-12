"""
Paraformer ASR Model Components

This module contains the core model components for the Paraformer ASR system:
- Transformer layers with multi-head attention
- Encoder stack for feature processing
- Predictor for alignment estimation
- Decoder for token generation
"""

from .transformer import (
    MultiHeadAttention, 
    FeedForward, 
    TransformerLayer,
    create_causal_mask,
    create_padding_mask
)
from .encoder import (
    ParaformerEncoder, 
    PositionalEncoding,
    InputProjection,
    create_encoder_from_config
)
from .predictor import (
    AlignmentPredictor,
    CTCAlignmentPredictor, 
    create_predictor_from_config
)
from .decoder import (
    ParaformerDecoder,
    DecoderLayer,
    PredictorIntegration,
    create_decoder_from_config
)
from .paraformer import (
    ParaformerASR,
    create_paraformer_from_config
)

__all__ = [
    # Transformer components
    'MultiHeadAttention',
    'FeedForward', 
    'TransformerLayer',
    'create_causal_mask',
    'create_padding_mask',
    
    # Encoder components
    'ParaformerEncoder',
    'PositionalEncoding',
    'InputProjection',
    'create_encoder_from_config',
    
    # Predictor components
    'AlignmentPredictor',
    'CTCAlignmentPredictor',
    'create_predictor_from_config',
    
    # Decoder components
    'ParaformerDecoder',
    'DecoderLayer', 
    'PredictorIntegration',
    'create_decoder_from_config',
    
    # Complete model
    'ParaformerASR',
    'create_paraformer_from_config'
]