"""
DreamerV4 Data Loading

Dataset loaders for various environments.
"""

from .minerl_dataset import MineRLDataset, create_dataloader

# Pretokenized dataset (optional - for faster training with pre-encoded Cosmos latents)
try:
    from .pretokenized_dataset import PretokenizedDataset, create_pretokenized_dataloader
    _pretokenized_available = True
except ImportError:
    _pretokenized_available = False
    PretokenizedDataset = None
    create_pretokenized_dataloader = None

__all__ = [
    "MineRLDataset",
    "create_dataloader",
]

if _pretokenized_available:
    __all__.extend(["PretokenizedDataset", "create_pretokenized_dataloader"])
