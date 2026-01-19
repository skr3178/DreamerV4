"""
DreamerV4 Models

Unified architecture where tokenizer and dynamics share the same
block-causal transformer architecture.
"""

from .transformer import (
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
    BlockCausalAttention,
    TransformerBlock,
    BlockCausalTransformer,
)
from .tokenizer import CausalTokenizer
from .dynamics import DynamicsModel
from .heads import PolicyHead, ValueHead, RewardHead
from .embeddings import PatchEmbedding, ActionEmbedding, SignalEmbedding

__all__ = [
    # Transformer components
    "RMSNorm",
    "RotaryPositionEmbedding", 
    "SwiGLU",
    "BlockCausalAttention",
    "TransformerBlock",
    "BlockCausalTransformer",
    # Main models
    "CausalTokenizer",
    "DynamicsModel",
    # Heads
    "PolicyHead",
    "ValueHead", 
    "RewardHead",
    # Embeddings
    "PatchEmbedding",
    "ActionEmbedding",
    "SignalEmbedding",
]
