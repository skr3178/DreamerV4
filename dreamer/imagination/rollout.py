"""
Imagination Rollout Generator for DreamerV4 Phase 3.

Generates imagined trajectories from the frozen world model.
Uses shortcut sampling for fast generation.

Key features:
- Autoregressive latent prediction with shortcut forcing
- Reward and value prediction along trajectory
- No environment interaction - all in imagination
"""

from typing import Dict, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn


class ImagineTrajectory(NamedTuple):
    """Container for imagined trajectory data."""
    latents: torch.Tensor      # (batch, horizon, num_latent, latent_dim)
    actions: torch.Tensor      # (batch, horizon) or (batch, horizon, action_dim)
    log_probs: torch.Tensor    # (batch, horizon)
    rewards: torch.Tensor      # (batch, horizon)
    values: torch.Tensor       # (batch, horizon)
    dones: torch.Tensor        # (batch, horizon)


class ImaginationRollout(nn.Module):
    """
    Generates imagined trajectories from frozen world model.
    
    Uses the dynamics model to predict future latent states,
    the policy head to sample actions, and reward/value heads
    to predict rewards and values along the trajectory.
    
    All world model components (tokenizer, dynamics, transformer)
    remain frozen during imagination rollouts.
    """
    
    def __init__(
        self,
        dynamics_model: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        reward_head: nn.Module,
        horizon: int = 15,
        num_denoising_steps: int = 4,
        discount: float = 0.997,
        discrete_actions: bool = True,
    ):
        """
        Args:
            dynamics_model: Frozen dynamics model
            policy_head: Policy head (trainable)
            value_head: Value head (trainable)
            reward_head: Reward head (trainable)
            horizon: Rollout horizon H
            num_denoising_steps: Number of shortcut denoising steps K
            discount: Discount factor γ
            discrete_actions: Whether action space is discrete
        """
        super().__init__()
        
        self.dynamics_model = dynamics_model
        self.policy_head = policy_head
        self.value_head = value_head
        self.reward_head = reward_head
        
        self.horizon = horizon
        self.num_denoising_steps = num_denoising_steps
        self.discount = discount
        self.discrete_actions = discrete_actions
        
        # Freeze dynamics model
        for param in self.dynamics_model.parameters():
            param.requires_grad = False
    
    def flatten_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Flatten latent tokens for head input."""
        # (batch, num_latent, latent_dim) -> (batch, num_latent * latent_dim)
        return latents.reshape(latents.shape[0], -1)
    
    @torch.no_grad()
    def generate_step(
        self,
        current_latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate next latent state using shortcut sampling.
        
        Args:
            current_latent: Current latent (batch, num_latent, latent_dim)
            action: Action to take (batch,) or (batch, action_dim)
        
        Returns:
            next_latent: Predicted next latent
        """
        batch_size = current_latent.shape[0]
        device = current_latent.device
        
        # Step size for shortcut sampling
        step_size_val = 1.0 / self.num_denoising_steps
        
        # Start from noise
        z = torch.randn_like(current_latent)
        
        # Prepare action
        if self.discrete_actions:
            action_t = action.unsqueeze(1)  # (batch, 1)
        else:
            action_t = action.unsqueeze(1)  # (batch, 1, action_dim)
        
        # Iterative denoising with shortcut forcing
        for step in range(self.num_denoising_steps):
            # τ is the signal level of current z BEFORE the update
            # For K=4: τ goes 0, 0.25, 0.5, 0.75 (z reaches τ=1 AFTER final update)
            # This matches dynamics.py generate() method
            tau = step * step_size_val
            tau_tensor = torch.full((batch_size,), tau, device=device)
            d_tensor = torch.full((batch_size,), step_size_val, device=device)
            
            # Prepare input
            z_input = z.unsqueeze(1)  # (batch, 1, num_latent, latent_dim)
            
            tokens = self.dynamics_model.prepare_sequence(
                z_input, action_t, tau_tensor, d_tensor, self.discrete_actions
            )
            
            # Get prediction
            output = self.dynamics_model.transformer(tokens)
            pred = self.dynamics_model.extract_latent_predictions(output, 1)[:, 0]
            
            # Update z using the shortcut update rule
            z = z + step_size_val * (pred - z)
        
        return z
    
    def rollout(
        self,
        initial_latents: torch.Tensor,
        deterministic: bool = False,
    ) -> ImagineTrajectory:
        """
        Generate a full imagined trajectory.
        
        Args:
            initial_latents: Starting latents (batch, num_latent, latent_dim)
            deterministic: If True, use mode of policy distribution
        
        Returns:
            ImagineTrajectory containing all trajectory data
        """
        batch_size = initial_latents.shape[0]
        device = initial_latents.device
        
        # Storage for trajectory
        latents_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        
        current_latent = initial_latents
        
        for t in range(self.horizon):
            # Store current latent
            latents_list.append(current_latent)

            # Flatten latent for heads
            flat_latent = self.flatten_latents(current_latent)

            # Predict value at current state
            value_out = self.value_head(flat_latent)
            values_list.append(value_out["value"])

            # Sample action from policy
            action, log_prob = self.policy_head.sample(
                flat_latent, deterministic=deterministic
            )
            # Handle MTP-mode policy: take only the first prediction (n=0)
            if action.dim() > 1:
                action = action[:, 0]
                log_prob = log_prob[:, 0]
            actions_list.append(action)
            log_probs_list.append(log_prob)

            # Generate next latent state
            with torch.no_grad():
                next_latent = self.generate_step(current_latent, action)

            # Predict reward for transition
            flat_next_latent = self.flatten_latents(next_latent)
            reward_out = self.reward_head(flat_next_latent)
            reward = reward_out["reward"]
            # Handle MTP-mode reward head: take only the first prediction (n=0)
            if reward.dim() > 1:
                reward = reward[:, 0]
            rewards_list.append(reward)
            
            # Move to next state
            current_latent = next_latent
        
        # Stack all trajectory data
        latents = torch.stack(latents_list, dim=1)  # (batch, horizon, num_latent, latent_dim)
        actions = torch.stack(actions_list, dim=1)  # (batch, horizon) or (batch, horizon, action_dim)
        log_probs = torch.stack(log_probs_list, dim=1)  # (batch, horizon)
        rewards = torch.stack(rewards_list, dim=1)  # (batch, horizon)
        values = torch.stack(values_list, dim=1)  # (batch, horizon)
        
        # No terminal states in imagination (could be extended to predict termination)
        dones = torch.zeros(batch_size, self.horizon, device=device)
        
        return ImagineTrajectory(
            latents=latents,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            values=values,
            dones=dones,
        )
    
    def compute_returns_and_advantages(
        self,
        trajectory: ImagineTrajectory,
        lambda_: float = 0.95,
        normalize_advantages: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute TD(λ) returns and advantages for the trajectory.
        
        Uses the generalized advantage estimation (GAE) with
        λ-weighted returns as per Equation 10.
        
        Args:
            trajectory: Imagined trajectory
            lambda_: TD(λ) mixing parameter
            normalize_advantages: Whether to normalize advantages
        
        Returns:
            returns: TD(λ) return targets (batch, horizon)
            advantages: Advantage estimates (batch, horizon)
        """
        rewards = trajectory.rewards
        values = trajectory.values
        dones = trajectory.dones
        
        batch_size, horizon = rewards.shape
        device = rewards.device
        
        # Bootstrap from last value
        # Get value at final state
        final_latent = trajectory.latents[:, -1]
        flat_final = self.flatten_latents(final_latent)
        with torch.no_grad():
            final_value_out = self.value_head(flat_final)
            bootstrap_value = final_value_out["value"]
        
        # Compute TD(λ) returns backwards
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Initialize with bootstrap
        next_value = bootstrap_value
        next_advantage = torch.zeros(batch_size, device=device)
        
        for t in reversed(range(horizon)):
            mask = 1.0 - dones[:, t]
            
            # TD error
            delta = rewards[:, t] + self.discount * mask * next_value - values[:, t]
            
            # GAE advantage
            advantage = delta + self.discount * lambda_ * mask * next_advantage
            advantages[:, t] = advantage
            
            # TD(λ) return
            returns[:, t] = advantage + values[:, t]
            
            # Update for next iteration
            next_value = values[:, t]
            next_advantage = advantage
        
        # Normalize advantages
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def forward(
        self,
        initial_latents: torch.Tensor,
        lambda_: float = 0.95,
        normalize_advantages: bool = True,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trajectory and compute all quantities needed for training.
        
        Args:
            initial_latents: Starting latents
            lambda_: TD(λ) parameter
            normalize_advantages: Whether to normalize advantages
            deterministic: Whether to use deterministic policy
        
        Returns:
            Dictionary with trajectory data and computed returns/advantages
        """
        # Generate trajectory
        trajectory = self.rollout(initial_latents, deterministic=deterministic)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            trajectory, lambda_=lambda_, normalize_advantages=normalize_advantages
        )
        
        return {
            "latents": trajectory.latents,
            "actions": trajectory.actions,
            "log_probs": trajectory.log_probs,
            "rewards": trajectory.rewards,
            "values": trajectory.values,
            "dones": trajectory.dones,
            "returns": returns,
            "advantages": advantages,
        }
