"""
Multi-Discrete Action Head for MineRL

Handles the MineRL action space:
- 23 binary keyboard actions (independent Bernoulli)
- 121-class categorical camera action (discretized pitch × yaw)

This replaces the simple PolicyHead for MineRL tasks.
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Bernoulli


class MultiDiscretePolicyHead(nn.Module):
    """
    Policy head for multi-discrete action spaces (MineRL).
    
    Outputs:
    - 23 independent Bernoulli distributions for keyboard actions
    - 1 categorical distribution (121 classes) for camera actions
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        keyboard_binary: int = 23,
        camera_discrete: int = 121,
        num_layers: int = 2,
    ):
        """
        Args:
            input_dim: Dimension of input latent representation
            hidden_dim: Hidden layer dimension
            keyboard_binary: Number of binary keyboard actions (23 for MineRL)
            camera_discrete: Number of camera action classes (121 for 11×11 bins)
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.keyboard_binary = keyboard_binary
        self.camera_discrete = camera_discrete
        
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
        # Keyboard: 23 independent binary outputs (logits for Bernoulli)
        self.keyboard_head = nn.Linear(hidden_dim, keyboard_binary)
        
        # Camera: 121-class categorical
        self.camera_head = nn.Linear(hidden_dim, camera_discrete)
    
    def forward(
        self,
        latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute action distributions from latent state.
        
        Args:
            latents: Latent representation (batch, ..., input_dim)
        
        Returns:
            Dictionary containing:
                - keyboard_logits: (..., 23) logits for binary actions
                - keyboard_probs: (..., 23) probabilities
                - camera_logits: (..., 121) logits for camera action
                - camera_probs: (..., 121) probabilities
        """
        # Flatten if needed
        original_shape = latents.shape[:-1]
        latents = latents.reshape(-1, self.input_dim)
        
        # MLP forward
        hidden = self.mlp(latents)
        
        # Keyboard outputs (23 binary)
        keyboard_logits = self.keyboard_head(hidden)  # (batch, 23)
        keyboard_probs = torch.sigmoid(keyboard_logits)  # Bernoulli probabilities
        
        # Camera output (121 categorical)
        camera_logits = self.camera_head(hidden)  # (batch, 121)
        camera_probs = F.softmax(camera_logits, dim=-1)
        
        # Reshape outputs
        keyboard_logits = keyboard_logits.reshape(*original_shape, self.keyboard_binary)
        keyboard_probs = keyboard_probs.reshape(*original_shape, self.keyboard_binary)
        camera_logits = camera_logits.reshape(*original_shape, self.camera_discrete)
        camera_probs = camera_probs.reshape(*original_shape, self.camera_discrete)
        
        return {
            "keyboard_logits": keyboard_logits,
            "keyboard_probs": keyboard_probs,
            "camera_logits": camera_logits,
            "camera_probs": camera_probs,
        }
    
    def sample(
        self,
        latents: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            latents: Latent representation
            deterministic: If True, return mode of distribution
        
        Returns:
            actions: Dictionary with 'keyboard' and 'camera' actions
            log_probs: Log probabilities of sampled actions
        """
        dist_params = self.forward(latents)
        
        # Sample keyboard actions (23 independent Bernoulli)
        keyboard_logits = dist_params["keyboard_logits"]
        keyboard_dist = Independent(Bernoulli(logits=keyboard_logits), 1)
        
        if deterministic:
            keyboard_actions = (keyboard_logits > 0).long()
        else:
            keyboard_actions = keyboard_dist.sample().long()
        
        keyboard_log_probs = keyboard_dist.log_prob(keyboard_actions.float())
        
        # Sample camera action (121 categorical)
        camera_logits = dist_params["camera_logits"]
        camera_dist = Categorical(logits=camera_logits)
        
        if deterministic:
            camera_actions = camera_logits.argmax(dim=-1)
        else:
            camera_actions = camera_dist.sample()
        
        camera_log_probs = camera_dist.log_prob(camera_actions)
        
        # Total log prob is sum
        total_log_probs = keyboard_log_probs + camera_log_probs
        
        actions = {
            "keyboard": keyboard_actions,
            "camera": camera_actions,
        }
        
        return actions, total_log_probs
    
    def log_prob(
        self,
        latents: torch.Tensor,
        actions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log probability of actions under the policy.
        
        Args:
            latents: Latent representation
            actions: Dictionary with 'keyboard' and 'camera' actions
                - keyboard: (..., 23) binary actions
                - camera: (...,) categorical action indices
        
        Returns:
            log_probs: Log probabilities
        """
        dist_params = self.forward(latents)
        
        # Keyboard log prob
        keyboard_logits = dist_params["keyboard_logits"]
        keyboard_dist = Independent(Bernoulli(logits=keyboard_logits), 1)
        keyboard_log_probs = keyboard_dist.log_prob(actions["keyboard"].float())
        
        # Camera log prob
        camera_logits = dist_params["camera_logits"]
        camera_dist = Categorical(logits=camera_logits)
        camera_log_probs = camera_dist.log_prob(actions["camera"])
        
        # Total log prob
        total_log_probs = keyboard_log_probs + camera_log_probs
        
        return total_log_probs
    
    def entropy(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the action distribution.
        
        Args:
            latents: Latent representation
        
        Returns:
            entropy: Entropy values
        """
        dist_params = self.forward(latents)
        
        # Keyboard entropy (sum of 23 independent Bernoulli entropies)
        keyboard_logits = dist_params["keyboard_logits"]
        keyboard_probs = dist_params["keyboard_probs"]
        keyboard_entropy = -(
            keyboard_probs * torch.log(keyboard_probs + 1e-8) +
            (1 - keyboard_probs) * torch.log(1 - keyboard_probs + 1e-8)
        ).sum(dim=-1)
        
        # Camera entropy
        camera_logits = dist_params["camera_logits"]
        camera_dist = Categorical(logits=camera_logits)
        camera_entropy = camera_dist.entropy()
        
        # Total entropy
        total_entropy = keyboard_entropy + camera_entropy
        
        return total_entropy
