"""
Utility functions for DreamerV4
"""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def create_block_causal_mask(
    seq_len: int,
    block_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a block-causal attention mask.
    
    In block-causal attention:
    - Tokens within the same block can attend to each other
    - Tokens can attend to all tokens in previous blocks  
    - Tokens cannot attend to future blocks
    
    Args:
        seq_len: Total sequence length
        block_size: Size of each block
        device: Device to create mask on
    
    Returns:
        Boolean mask of shape (seq_len, seq_len)
        True = can attend, False = masked
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    
    num_blocks = (seq_len + block_size - 1) // block_size
    
    for i in range(num_blocks):
        block_start = i * block_size
        block_end = min((i + 1) * block_size, seq_len)
        
        # Can attend to all previous blocks and current block
        mask[block_start:block_end, :block_end] = True
    
    return mask


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric logarithm for handling large value ranges.
    
    symlog(x) = sign(x) * ln(|x| + 1)
    
    Args:
        x: Input tensor
    
    Returns:
        Transformed tensor
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of symlog.
    
    symexp(x) = sign(x) * (exp(|x|) - 1)
    
    Args:
        x: Input tensor
    
    Returns:
        Transformed tensor
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """
    Compute λ-returns for value learning (used in Equation 10).
    
    R_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} R_t^{(n)}
    
    Args:
        rewards: Rewards (batch, time)
        values: Value estimates (batch, time)
        dones: Done flags (batch, time)
        gamma: Discount factor
        lambda_: λ parameter for TD(λ)
    
    Returns:
        λ-returns (batch, time)
    """
    batch_size, time_steps = rewards.shape
    
    # Initialize returns
    returns = torch.zeros_like(rewards)
    
    # Bootstrap from last value
    next_value = values[:, -1]
    next_return = next_value
    
    # Backward pass to compute returns
    for t in reversed(range(time_steps)):
        # TD target
        td_target = rewards[:, t] + gamma * (1 - dones[:, t].float()) * next_value
        
        # λ-return
        returns[:, t] = (1 - lambda_) * td_target + lambda_ * (
            rewards[:, t] + gamma * (1 - dones[:, t].float()) * next_return
        )
        
        next_value = values[:, t]
        next_return = returns[:, t]
    
    return returns


def soft_update(
    target: nn.Module,
    source: nn.Module,
    tau: float = 0.005,
):
    """
    Soft update target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        target: Target network
        source: Source network
        tau: Update rate
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1 - tau) * target_param.data
        )


def freeze_module(module: nn.Module):
    """
    Freeze all parameters in a module.
    
    Args:
        module: Module to freeze
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module):
    """
    Unfreeze all parameters in a module.
    
    Args:
        module: Module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
