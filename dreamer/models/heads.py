"""
Agent Heads for DreamerV4 Phase 2

MLP heads attached to the frozen transformer for:
- Policy (action prediction)
- Value (state value estimation)
- Reward (reward prediction)

These heads are trained during Phase 2 (Agent Finetuning) and Phase 3 (Imagination RL)
while the main transformer remains frozen.
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class PolicyHead(nn.Module):
    """
    Policy head for action prediction.
    
    Outputs action distribution conditioned on latent state.
    Supports both discrete and continuous action spaces.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_discrete_actions: Optional[int] = None,
        continuous_action_dim: Optional[int] = None,
        num_layers: int = 2,
    ):
        """
        Args:
            input_dim: Dimension of input latent representation
            hidden_dim: Hidden layer dimension
            num_discrete_actions: Number of discrete actions (if discrete)
            continuous_action_dim: Dimension of continuous actions (if continuous)
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_discrete_actions = num_discrete_actions
        self.continuous_action_dim = continuous_action_dim
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output heads
        if num_discrete_actions is not None:
            self.action_head = nn.Linear(hidden_dim, num_discrete_actions)
        
        if continuous_action_dim is not None:
            # Output mean and log_std for Gaussian policy
            self.mean_head = nn.Linear(hidden_dim, continuous_action_dim)
            self.log_std_head = nn.Linear(hidden_dim, continuous_action_dim)
    
    def forward(
        self,
        latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute action distribution from latent state.
        
        Args:
            latents: Latent representation (batch, ..., input_dim)
        
        Returns:
            Dictionary containing distribution parameters
        """
        # Flatten if needed
        original_shape = latents.shape[:-1]
        latents = latents.reshape(-1, self.input_dim)
        
        # MLP forward
        hidden = self.mlp(latents)
        
        result = {}
        
        if self.num_discrete_actions is not None:
            logits = self.action_head(hidden)
            logits = logits.reshape(*original_shape, self.num_discrete_actions)
            result["logits"] = logits
            result["probs"] = F.softmax(logits, dim=-1)
        
        if self.continuous_action_dim is not None:
            mean = self.mean_head(hidden)
            log_std = self.log_std_head(hidden)
            log_std = torch.clamp(log_std, -20, 2)  # Stability
            
            mean = mean.reshape(*original_shape, self.continuous_action_dim)
            log_std = log_std.reshape(*original_shape, self.continuous_action_dim)
            
            result["mean"] = mean
            result["log_std"] = log_std
            result["std"] = torch.exp(log_std)
        
        return result
    
    def sample(
        self,
        latents: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            latents: Latent representation
            deterministic: If True, return mode of distribution
        
        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of sampled actions
        """
        dist_params = self.forward(latents)
        
        if self.num_discrete_actions is not None:
            logits = dist_params["logits"]
            dist = Categorical(logits=logits)
            
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                actions = dist.sample()
            
            log_probs = dist.log_prob(actions)
            
        else:  # Continuous
            mean = dist_params["mean"]
            std = dist_params["std"]
            dist = Normal(mean, std)
            
            if deterministic:
                actions = mean
            else:
                actions = dist.rsample()  # Reparameterized sample
            
            log_probs = dist.log_prob(actions).sum(dim=-1)
        
        return actions, log_probs
    
    def log_prob(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of actions under the policy.
        
        Args:
            latents: Latent representation
            actions: Actions to evaluate
        
        Returns:
            log_probs: Log probabilities
        """
        dist_params = self.forward(latents)
        
        if self.num_discrete_actions is not None:
            dist = Categorical(logits=dist_params["logits"])
            return dist.log_prob(actions)
        else:
            dist = Normal(dist_params["mean"], dist_params["std"])
            return dist.log_prob(actions).sum(dim=-1)


class ValueHead(nn.Module):
    """
    Value head for state value estimation V(s).
    
    Outputs a scalar value estimate for each state.
    Uses distributional value learning with symlog encoding.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_bins: int = 255,
        value_range: Tuple[float, float] = (-20.0, 20.0),
    ):
        """
        Args:
            input_dim: Dimension of input latent representation
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            num_bins: Number of bins for distributional value
            value_range: (min, max) value range
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.value_range = value_range
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output head (distributional)
        self.value_head = nn.Linear(hidden_dim, num_bins)
        
        # Precompute bin centers
        bin_centers = torch.linspace(value_range[0], value_range[1], num_bins)
        self.register_buffer("bin_centers", bin_centers)
    
    def forward(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute value distribution and expected value.
        
        Args:
            latents: Latent representation (batch, ..., input_dim)
        
        Returns:
            Dictionary containing:
                - logits: Distribution logits
                - probs: Distribution probabilities
                - value: Expected value
        """
        original_shape = latents.shape[:-1]
        latents = latents.reshape(-1, self.input_dim)
        
        hidden = self.mlp(latents)
        logits = self.value_head(hidden)
        
        probs = F.softmax(logits, dim=-1)
        value = (probs * self.bin_centers).sum(dim=-1)
        
        # Reshape outputs
        logits = logits.reshape(*original_shape, self.num_bins)
        probs = probs.reshape(*original_shape, self.num_bins)
        value = value.reshape(*original_shape)
        
        return {
            "logits": logits,
            "probs": probs,
            "value": value,
        }
    
    def target_to_bins(self, target: torch.Tensor) -> torch.Tensor:
        """
        Convert target values to bin distribution targets (two-hot encoding).
        
        Args:
            target: Target values (batch, ...)
        
        Returns:
            Bin distribution targets for cross-entropy loss
        """
        # Clamp target to valid range
        target = target.clamp(self.value_range[0], self.value_range[1])
        
        # Find bin indices
        bin_width = (self.value_range[1] - self.value_range[0]) / (self.num_bins - 1)
        bin_idx = (target - self.value_range[0]) / bin_width
        
        # Two-hot encoding
        lower_idx = bin_idx.floor().long().clamp(0, self.num_bins - 2)
        upper_idx = lower_idx + 1
        upper_weight = bin_idx - lower_idx.float()
        lower_weight = 1.0 - upper_weight
        
        # Create target distribution
        target_dist = torch.zeros(*target.shape, self.num_bins, device=target.device)
        target_dist.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        target_dist.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
        
        return target_dist


class RewardHead(nn.Module):
    """
    Reward head for reward prediction r(s).
    
    Predicts expected reward at each state.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_bins: int = 255,
        reward_range: Tuple[float, float] = (-10.0, 10.0),
    ):
        """
        Args:
            input_dim: Dimension of input latent representation
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            num_bins: Number of bins for distributional reward
            reward_range: (min, max) reward range
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.reward_range = reward_range
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output head
        self.reward_head = nn.Linear(hidden_dim, num_bins)
        
        # Precompute bin centers
        bin_centers = torch.linspace(reward_range[0], reward_range[1], num_bins)
        self.register_buffer("bin_centers", bin_centers)
    
    def forward(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute reward distribution and expected reward.
        
        Args:
            latents: Latent representation (batch, ..., input_dim)
        
        Returns:
            Dictionary containing:
                - logits: Distribution logits
                - probs: Distribution probabilities  
                - reward: Expected reward
        """
        original_shape = latents.shape[:-1]
        latents = latents.reshape(-1, self.input_dim)
        
        hidden = self.mlp(latents)
        logits = self.reward_head(hidden)
        
        probs = F.softmax(logits, dim=-1)
        reward = (probs * self.bin_centers).sum(dim=-1)
        
        # Reshape outputs
        logits = logits.reshape(*original_shape, self.num_bins)
        probs = probs.reshape(*original_shape, self.num_bins)
        reward = reward.reshape(*original_shape)
        
        return {
            "logits": logits,
            "probs": probs,
            "reward": reward,
        }
    
    def target_to_bins(self, target: torch.Tensor) -> torch.Tensor:
        """Convert target rewards to bin distribution targets."""
        target = target.clamp(self.reward_range[0], self.reward_range[1])
        
        bin_width = (self.reward_range[1] - self.reward_range[0]) / (self.num_bins - 1)
        bin_idx = (target - self.reward_range[0]) / bin_width
        
        lower_idx = bin_idx.floor().long().clamp(0, self.num_bins - 2)
        upper_idx = lower_idx + 1
        upper_weight = bin_idx - lower_idx.float()
        lower_weight = 1.0 - upper_weight
        
        target_dist = torch.zeros(*target.shape, self.num_bins, device=target.device)
        target_dist.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        target_dist.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
        
        return target_dist
