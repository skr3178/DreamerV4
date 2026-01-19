"""
DreamerV4 Loss Functions

Implements the key loss functions from the paper:
- Eq. 5: Tokenizer loss (MAE + LPIPS)
- Eq. 7: Shortcut forcing loss
- Eq. 9: Behavior cloning + reward prediction
- Eq. 10: TD(λ) value loss
- Eq. 11: PMPO policy loss
"""

from .tokenizer_loss import TokenizerLoss
from .shortcut_loss import ShortcutForcingLoss
from .agent_loss import BehaviorCloningLoss, RewardPredictionLoss, AgentFinetuningLoss
from .pmpo_loss import PMPOLoss, PPOClipLoss
from .value_loss import TDLambdaLoss, ValueSymlogLoss, CombinedAgentLoss

__all__ = [
    "TokenizerLoss",
    "ShortcutForcingLoss",
    "BehaviorCloningLoss",
    "RewardPredictionLoss",
    "AgentFinetuningLoss",
    "PMPOLoss",
    "PPOClipLoss",
    "TDLambdaLoss",
    "ValueSymlogLoss",
    "CombinedAgentLoss",
]
