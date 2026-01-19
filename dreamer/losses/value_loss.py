"""
TD(λ) Value Loss for DreamerV4 Phase 3.

Implements Equation 10 from the DreamerV4 paper:
- Lambda-weighted returns (exponentially weighted mix of n-step returns)
- Value regression to bootstrapped targets
- Distributional value learning with two-hot encoding

Key formula:
R_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} R_t^{(n)}
L_value = -log p(R_t^λ | s_t)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDLambdaLoss(nn.Module):
    """
    TD(λ) Value Loss for critic training.
    
    Uses generalized advantage estimation (GAE) and λ-returns
    for stable value function learning in imagination.
    
    Supports both:
    - Scalar value prediction with MSE loss
    - Distributional value prediction with cross-entropy loss
    """
    
    def __init__(
        self,
        discount: float = 0.997,
        lambda_: float = 0.95,
        loss_scale: float = 0.5,
        use_distributional: bool = True,
    ):
        """
        Args:
            discount: Discount factor γ
            lambda_: TD(λ) mixing parameter
            loss_scale: Scale factor for value loss
            use_distributional: Whether to use distributional value learning
        """
        super().__init__()
        
        self.discount = discount
        self.lambda_ = lambda_
        self.loss_scale = loss_scale
        self.use_distributional = use_distributional
    
    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        bootstrap_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute TD(λ) returns as per Equation 10.
        
        G_t^λ = r_t + γ[(1-λ)V(s_{t+1}) + λG_{t+1}^λ]
        
        Args:
            rewards: Predicted rewards (batch, horizon)
            values: Predicted values (batch, horizon)
            dones: Done flags (batch, horizon)
            bootstrap_value: Value at final state (batch,)
        
        Returns:
            Lambda returns for each timestep (batch, horizon)
        """
        batch_size, horizon = rewards.shape
        device = rewards.device
        
        returns = torch.zeros_like(rewards)
        
        # Initialize with bootstrap value
        next_return = bootstrap_value
        
        # Compute returns backwards
        for t in reversed(range(horizon)):
            mask = 1.0 - dones[:, t]
            
            # TD(λ) target
            # G_t^λ = r_t + γ * [(1-λ) * V(s_{t+1}) + λ * G_{t+1}^λ]
            if t == horizon - 1:
                next_value = bootstrap_value
            else:
                next_value = values[:, t + 1]
            
            returns[:, t] = rewards[:, t] + self.discount * mask * (
                (1 - self.lambda_) * next_value + self.lambda_ * next_return
            )
            
            next_return = returns[:, t]
        
        return returns
    
    def forward(
        self,
        value_head: nn.Module,
        latents: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        bootstrap_latent: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute TD(λ) value loss.
        
        Args:
            value_head: Value head network
            latents: Flattened latent states (batch, horizon, latent_dim)
            rewards: Predicted rewards (batch, horizon)
            dones: Done flags (batch, horizon)
            bootstrap_latent: Final state latent for bootstrapping
        
        Returns:
            Dictionary containing:
                - loss: Total value loss
                - value_loss: Main regression loss
                - mean_value: Mean predicted value
                - mean_return: Mean TD(λ) return
        """
        batch_size, horizon = rewards.shape
        device = latents.device
        
        # Get value predictions
        flat_latents = latents.reshape(-1, latents.shape[-1])
        value_output = value_head(flat_latents)
        values = value_output["value"].reshape(batch_size, horizon)
        
        # Compute bootstrap value
        if bootstrap_latent is not None:
            with torch.no_grad():
                bootstrap_output = value_head(bootstrap_latent)
                bootstrap_value = bootstrap_output["value"]
        else:
            # Use last predicted value as bootstrap
            bootstrap_value = values[:, -1].detach()
        
        # Compute TD(λ) returns
        with torch.no_grad():
            returns = self.compute_lambda_returns(
                rewards=rewards,
                values=values.detach(),
                dones=dones,
                bootstrap_value=bootstrap_value,
            )
        
        # Compute loss
        if self.use_distributional:
            # Distributional value loss with two-hot encoding
            flat_returns = returns.reshape(-1)
            target_dist = value_head.target_to_bins(flat_returns)
            
            value_logits = value_output["logits"].reshape(-1, value_output["logits"].shape[-1])
            
            # Cross-entropy loss
            value_loss = F.cross_entropy(
                value_logits,
                target_dist,
                reduction="mean",
            )
        else:
            # Simple MSE loss
            value_loss = F.mse_loss(values, returns)
        
        # Apply scale
        total_loss = self.loss_scale * value_loss
        
        return {
            "loss": total_loss,
            "value_loss": value_loss,
            "mean_value": values.mean(),
            "mean_return": returns.mean(),
            "value_std": values.std(),
            "return_std": returns.std(),
        }


class ValueSymlogLoss(nn.Module):
    """
    Alternative: Symlog-encoded value loss.
    
    Uses symlog transformation for better handling of
    large value magnitudes:
    symlog(x) = sign(x) * ln(|x| + 1)
    """
    
    def __init__(
        self,
        discount: float = 0.997,
        lambda_: float = 0.95,
        loss_scale: float = 0.5,
    ):
        super().__init__()
        
        self.discount = discount
        self.lambda_ = lambda_
        self.loss_scale = loss_scale
    
    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        """Symmetric logarithm: sign(x) * ln(|x| + 1)"""
        return torch.sign(x) * torch.log1p(torch.abs(x))
    
    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        """Inverse of symlog: sign(x) * (exp(|x|) - 1)"""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    
    def forward(
        self,
        value_head: nn.Module,
        latents: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute symlog value loss.
        
        Args:
            value_head: Value head (outputs scalar values)
            latents: Latent states
            returns: Pre-computed returns
        
        Returns:
            Loss dictionary
        """
        flat_latents = latents.reshape(-1, latents.shape[-1])
        flat_returns = returns.reshape(-1)
        
        value_output = value_head(flat_latents)
        predicted_values = value_output["value"].reshape(-1)
        
        # Symlog transform for stable learning
        symlog_pred = self.symlog(predicted_values)
        symlog_target = self.symlog(flat_returns)
        
        value_loss = F.mse_loss(symlog_pred, symlog_target)
        total_loss = self.loss_scale * value_loss
        
        return {
            "loss": total_loss,
            "value_loss": value_loss,
            "mean_value": predicted_values.mean(),
            "mean_return": flat_returns.mean(),
        }


class CombinedAgentLoss(nn.Module):
    """
    Combined loss for Phase 3 training.
    
    Combines:
    - PMPO policy loss
    - TD(λ) value loss
    - Reward prediction loss (optional)
    """
    
    def __init__(
        self,
        pmpo_loss: nn.Module,
        value_loss: nn.Module,
        policy_weight: float = 1.0,
        value_weight: float = 0.5,
        reward_weight: float = 0.1,
    ):
        super().__init__()
        
        self.pmpo_loss = pmpo_loss
        self.value_loss = value_loss
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.reward_weight = reward_weight
    
    def forward(
        self,
        policy_head: nn.Module,
        value_head: nn.Module,
        reward_head: nn.Module,
        trajectory_data: Dict[str, torch.Tensor],
        prior_head: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined agent loss.
        
        Args:
            policy_head: Policy network
            value_head: Value network
            reward_head: Reward network
            trajectory_data: Dictionary with trajectory tensors
            prior_head: Optional behavioral prior
        
        Returns:
            Combined loss dictionary
        """
        # Flatten latents
        latents = trajectory_data["latents"]
        batch_size, horizon = latents.shape[:2]
        flat_latents = latents.reshape(batch_size, horizon, -1)
        
        # Policy loss
        policy_result = self.pmpo_loss(
            policy_head=policy_head,
            latents=flat_latents,
            actions=trajectory_data["actions"],
            advantages=trajectory_data["advantages"],
            prior_head=prior_head,
        )
        
        # Value loss
        value_result = self.value_loss(
            value_head=value_head,
            latents=flat_latents,
            rewards=trajectory_data["rewards"],
            dones=trajectory_data["dones"],
        )
        
        # Combined loss
        total_loss = (
            self.policy_weight * policy_result["loss"]
            + self.value_weight * value_result["loss"]
        )
        
        return {
            "loss": total_loss,
            "policy_loss": policy_result["loss"],
            "value_loss": value_result["loss"],
            "entropy": policy_result.get("entropy", 0.0),
            "mean_value": value_result["mean_value"],
            "mean_return": value_result["mean_return"],
            "n_positive": policy_result.get("n_positive", 0),
            "n_negative": policy_result.get("n_negative", 0),
        }
