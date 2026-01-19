"""
DreamerV4 Implementation - From Scratch
Based on: "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)

Key Features:
- Unified block-causal transformer architecture for tokenizer and dynamics
- Shortcut forcing objective for efficient generation
- PMPO for policy optimization in imagination
- TD(λ) for value learning
"""

from .models import BlockCausalTransformer, CausalTokenizer, DynamicsModel
from .models import PolicyHead, ValueHead, RewardHead
from .imagination import ImaginationRollout
from .losses import PMPOLoss, TDLambdaLoss

__version__ = "0.1.0"
__all__ = [
    # Models
    "BlockCausalTransformer",
    "CausalTokenizer", 
    "DynamicsModel",
    "PolicyHead",
    "ValueHead",
    "RewardHead",
    # Imagination
    "ImaginationRollout",
    # Losses
    "PMPOLoss",
    "TDLambdaLoss",
]
