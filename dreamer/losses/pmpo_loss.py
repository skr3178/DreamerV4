"""
PMPO (Policy Mirror Proximal Policy Optimization) Loss for DreamerV4 Phase 3.

Implements Equation 11 from the DreamerV4 paper:
- Advantage binning into D+ (positive) and D- (negative) sets
- Weighted log probability updates
- KL regularization to behavioral prior

Key insight: PMPO pushes good actions up and bad actions down,
with robust advantage signs rather than magnitudes.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class PMPOLoss(nn.Module):
    """
    PMPO Policy Loss as per DreamerV4 Equation 11.
    
    Loss formula:
    L_policy = (1-α)/|D-| × Σ_{D-} log π(a|s) 
             - α/|D+| × Σ_{D+} log π(a|s) 
             + β/N × Σ KL[π(a|s) || π_prior]
    
    Where:
    - D+: Trajectories with positive advantage (good actions)
    - D-: Trajectories with negative advantage (bad actions)
    - α: Weight for positive advantage updates (typically 0.5)
    - β: Weight for KL regularization
    - π_prior: Behavioral cloning prior (prevents policy drift)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta_kl: float = 0.3,
        entropy_coef: float = 0.003,
        num_bins: int = 16,
        discrete_actions: bool = True,
    ):
        """
        Args:
            alpha: Weight for positive advantage updates (α in Eq.11)
            beta_kl: Weight for KL regularization to prior (β in Eq.11, paper default 0.3)
            entropy_coef: Entropy bonus coefficient for exploration
            num_bins: Number of bins for advantage binning
            discrete_actions: Whether action space is discrete
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta_kl = beta_kl
        self.entropy_coef = entropy_coef
        self.num_bins = num_bins
        self.discrete_actions = discrete_actions
    
    def bin_advantages(
        self,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bin advantages into D+ and D- sets.
        
        Args:
            advantages: Advantage values (batch * horizon,)
        
        Returns:
            positive_mask: Boolean mask for D+ (positive advantages, > 0)
            negative_mask: Boolean mask for D- (negative advantages, < 0)
        """
        positive_mask = advantages > 0
        negative_mask = advantages < 0  # Strictly negative (zero goes to neither, but rare)
        return positive_mask, negative_mask
    
    def compute_kl_divergence(
        self,
        policy_dist,
        prior_dist,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and prior distributions.
        
        Args:
            policy_dist: Current policy distribution (Categorical or Normal)
            prior_dist: Prior (BC) policy distribution (Categorical or Normal)
        
        Returns:
            KL divergence per sample (not averaged)
        """
        # Use PyTorch's built-in KL divergence for proper computation
        # Returns KL for each sample: [B*H] for batch of samples
        kl = torch.distributions.kl.kl_divergence(policy_dist, prior_dist)
        return kl
    
    def compute_entropy(
        self,
        policy_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute policy entropy for exploration bonus.
        
        Args:
            policy_output: Policy head output dict
        
        Returns:
            Mean entropy
        """
        if self.discrete_actions:
            logits = policy_output["logits"]
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
        else:
            std = policy_output["std"]
            # Entropy of Gaussian: 0.5 * log(2πe * σ²)
            entropy = 0.5 + 0.5 * torch.log(2 * torch.pi * std ** 2)
            entropy = entropy.sum(dim=-1)
        
        return entropy.mean()
    
    def forward(
        self,
        policy_head: nn.Module,
        latents: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        prior_head: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PMPO policy loss.
        
        Args:
            policy_head: Current policy head
            latents: Flattened latent states (batch, horizon, latent_dim)
            actions: Actions taken (batch, horizon) or (batch, horizon, action_dim)
            advantages: Advantage estimates (batch, horizon)
            prior_head: Optional prior policy (for KL regularization)
        
        Returns:
            Dictionary containing:
                - loss: Total PMPO loss
                - policy_loss: Main PMPO policy loss
                - kl_loss: KL divergence to prior
                - entropy_loss: Negative entropy (for bonus)
                - entropy: Policy entropy
        """
        batch_size, horizon = latents.shape[:2]
        device = latents.device
        
        # Flatten for processing
        flat_latents = latents.reshape(-1, latents.shape[-1])  # (B*H, latent_dim)
        flat_actions = actions.reshape(-1) if self.discrete_actions else actions.reshape(-1, actions.shape[-1])
        flat_advantages = advantages.reshape(-1)  # (B*H,)
        
        # Get current policy distribution
        policy_output = policy_head(flat_latents)
        
        # Compute log probabilities of taken actions
        if self.discrete_actions:
            dist = Categorical(logits=policy_output["logits"])
            log_probs = dist.log_prob(flat_actions)
        else:
            dist = Normal(policy_output["mean"], policy_output["std"])
            log_probs = dist.log_prob(flat_actions).sum(dim=-1)
        
        # Bin advantages into D+ and D- (Equation 11)
        positive_mask = flat_advantages > 0   # D+: good actions
        negative_mask = flat_advantages < 0    # D-: bad actions
        
        # Count samples in each set
        n_positive = positive_mask.sum().float().clamp(min=1.0)
        n_negative = negative_mask.sum().float().clamp(min=1.0)
        n_total = float(flat_latents.shape[0])
        
        # PMPO loss computation (Equation 11)
        # Positive advantages: decrease log_prob (negative coefficient)
        # This means: minimize -alpha/n_pos * sum(log_prob) → maximize log_prob for good actions
        pos_term = -(self.alpha / n_positive) * log_probs[positive_mask].sum()
        
        # Negative advantages: increase log_prob (positive coefficient)  
        # This means: minimize (1-alpha)/n_neg * sum(log_prob) → minimize log_prob for bad actions
        neg_term = ((1 - self.alpha) / n_negative) * log_probs[negative_mask].sum()
        
        policy_loss = pos_term + neg_term
        
        # KL regularization to prior (Equation 11: β/N * Σ KL[π || π_prior])
        kl_loss = torch.tensor(0.0, device=device)
        if prior_head is not None and self.beta_kl > 0:
            with torch.no_grad():
                prior_output = prior_head(flat_latents)
            
            # Create distributions for KL computation
            if self.discrete_actions:
                policy_dist = Categorical(logits=policy_output["logits"])
                prior_dist = Categorical(logits=prior_output["logits"])
            else:
                policy_dist = Normal(policy_output["mean"], policy_output["std"])
                prior_dist = Normal(prior_output["mean"], prior_output["std"])
            
            # Compute KL divergence per sample, then apply formula: β/N * Σ KL
            kl_per_sample = self.compute_kl_divergence(policy_dist, prior_dist)  # [B*H]
            kl_sum = kl_per_sample.sum()  # Σ KL
            kl_loss = (self.beta_kl / n_total) * kl_sum  # β/N * Σ KL
        
        # Entropy bonus for exploration
        entropy = self.compute_entropy(policy_output)
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + kl_loss + entropy_loss
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "positive_loss": pos_term,
            "negative_loss": neg_term,
            "kl_loss": kl_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "n_positive": n_positive,
            "n_negative": n_negative,
        }


class PPOClipLoss(nn.Module):
    """
    Alternative: Standard PPO clipped objective.
    
    Can be used instead of PMPO for comparison.
    L = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
    """
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.003,
        discrete_actions: bool = True,
    ):
        super().__init__()
        
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.discrete_actions = discrete_actions
    
    def forward(
        self,
        policy_head: nn.Module,
        latents: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO clipped policy loss.
        
        Args:
            policy_head: Current policy head
            latents: Flattened latent states
            actions: Actions taken
            advantages: Advantage estimates
            old_log_probs: Log probs from behavior policy
        
        Returns:
            Dictionary containing loss components
        """
        # Flatten
        flat_latents = latents.reshape(-1, latents.shape[-1])
        flat_actions = actions.reshape(-1) if self.discrete_actions else actions.reshape(-1, actions.shape[-1])
        flat_advantages = advantages.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        
        # Get current log probs
        policy_output = policy_head(flat_latents)
        
        if self.discrete_actions:
            dist = Categorical(logits=policy_output["logits"])
            log_probs = dist.log_prob(flat_actions)
        else:
            dist = Normal(policy_output["mean"], policy_output["std"])
            log_probs = dist.log_prob(flat_actions).sum(dim=-1)
        
        # Importance sampling ratio
        ratio = torch.exp(log_probs - flat_old_log_probs)
        
        # Clipped objective
        surr1 = ratio * flat_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * flat_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy
        probs = F.softmax(policy_output["logits"], dim=-1) if self.discrete_actions else None
        if probs is not None:
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
        else:
            entropy = (0.5 + 0.5 * torch.log(2 * torch.pi * policy_output["std"] ** 2)).sum(dim=-1).mean()
        
        entropy_loss = -self.entropy_coef * entropy
        
        total_loss = policy_loss + entropy_loss
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "approx_kl": ((ratio - 1) - (ratio.log())).mean(),
            "clip_fraction": ((ratio - 1).abs() > self.clip_epsilon).float().mean(),
        }
