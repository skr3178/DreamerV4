"""
Agent Losses for DreamerV4 Phase 2 (Equation 9)

Behavior cloning and reward prediction losses for agent finetuning.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorCloningLoss(nn.Module):
    """
    Behavior Cloning Loss (part of Equation 9).
    
    Negative log-likelihood of actions over multiple steps.
    L_BC = -Σ_{k=1}^{K} log π(a_{t+k}|s_{t+k})
    
    Supports both discrete and continuous actions.
    """
    
    def __init__(
        self,
        num_prediction_steps: int = 8,
    ):
        """
        Args:
            num_prediction_steps: Number of future steps to predict (K)
        """
        super().__init__()
        self.num_prediction_steps = num_prediction_steps
    
    def forward(
        self,
        policy_output: Dict[str, torch.Tensor],
        target_actions: torch.Tensor,
        action_type: str = "discrete",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute behavior cloning loss.
        
        Args:
            policy_output: Output from PolicyHead (logits for discrete, mean/std for continuous)
            target_actions: Target actions (batch, time, ...) 
            action_type: "discrete" or "continuous"
        
        Returns:
            Dictionary with loss components
        """
        if action_type == "discrete":
            # Cross-entropy loss for discrete actions
            logits = policy_output["logits"]  # (batch, time, num_actions)
            
            # Flatten for cross-entropy
            batch_size, time_steps, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)
            targets_flat = target_actions.reshape(-1).long()
            
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
            
            # Compute accuracy for logging
            predicted = logits.argmax(dim=-1)
            accuracy = (predicted == target_actions).float().mean()
            
        else:
            # Gaussian NLL for continuous actions
            mean = policy_output["mean"]
            std = policy_output["std"]
            
            # Compute negative log-likelihood
            var = std ** 2
            log_prob = -0.5 * (
                ((target_actions - mean) ** 2) / var + 
                torch.log(var) + 
                torch.log(torch.tensor(2 * 3.14159265))
            )
            
            loss = -log_prob.mean()
            accuracy = torch.tensor(0.0)  # Not applicable for continuous
        
        return {
            "loss": loss,
            "accuracy": accuracy,
        }


class RewardPredictionLoss(nn.Module):
    """
    Reward Prediction Loss (part of Equation 9).
    
    Predicts rewards at each timestep using distributional learning.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        reward_output: Dict[str, torch.Tensor],
        target_rewards: torch.Tensor,
        reward_head: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reward prediction loss.
        
        Args:
            reward_output: Output from RewardHead (logits, probs, reward)
            target_rewards: Target rewards (batch, time)
            reward_head: RewardHead module (for target_to_bins)
        
        Returns:
            Dictionary with loss components
        """
        logits = reward_output["logits"]
        predicted_rewards = reward_output["reward"]
        
        # Convert targets to bin distributions
        target_bins = reward_head.target_to_bins(target_rewards)
        
        # Cross-entropy loss on distributions
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_bins * log_probs).sum(dim=-1).mean()
        
        # Compute MSE for logging
        mse = F.mse_loss(predicted_rewards, target_rewards)
        
        return {
            "loss": loss,
            "mse": mse,
        }


class AgentFinetuningLoss(nn.Module):
    """
    Combined loss for agent finetuning (Equation 9).
    
    L = L_BC + λ_reward * L_reward
    
    Where:
    - L_BC: Behavior cloning loss
    - L_reward: Reward prediction loss
    """
    
    def __init__(
        self,
        reward_weight: float = 1.0,
        num_prediction_steps: int = 8,
    ):
        """
        Args:
            reward_weight: Weight for reward prediction loss
            num_prediction_steps: Number of future steps to predict
        """
        super().__init__()
        self.reward_weight = reward_weight
        self.bc_loss = BehaviorCloningLoss(num_prediction_steps)
        self.reward_loss = RewardPredictionLoss()
    
    def forward(
        self,
        policy_head: nn.Module,
        reward_head: nn.Module,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        action_type: str = "discrete",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined agent finetuning loss.
        
        Args:
            policy_head: PolicyHead or MultiDiscretePolicyHead
            reward_head: RewardHead
            latents: Latent states (batch, time, ...)
            actions: Target actions (batch, time, ...) or Dict for multi-discrete
            rewards: Target rewards (batch, time)
            action_type: "discrete", "continuous", or "multi_discrete"
        
        Returns:
            Dictionary with all loss components
        """
        # Get policy and reward outputs
        policy_output = policy_head(latents)
        reward_output = reward_head(latents)
        
        # Behavior cloning loss
        if action_type == "multi_discrete":
            # Multi-discrete: actions is a Dict with 'keyboard' and 'camera'
            from torch.distributions import Independent, Bernoulli, Categorical
            
            keyboard_logits = policy_output["keyboard_logits"]
            camera_logits = policy_output["camera_logits"]
            
            # Keyboard BC loss (23 independent Bernoulli)
            keyboard_dist = Independent(Bernoulli(logits=keyboard_logits), 1)
            keyboard_log_probs = keyboard_dist.log_prob(actions["keyboard"].float())
            keyboard_bc_loss = -keyboard_log_probs.mean()
            
            # Camera BC loss (121 categorical)
            camera_dist = Categorical(logits=camera_logits)
            camera_log_probs = camera_dist.log_prob(actions["camera"])
            camera_bc_loss = -camera_log_probs.mean()
            
            bc_loss = keyboard_bc_loss + camera_bc_loss
            
            # Compute accuracy
            keyboard_pred = (keyboard_logits > 0).long()
            keyboard_acc = (keyboard_pred == actions["keyboard"]).float().mean()
            camera_pred = camera_logits.argmax(dim=-1)
            camera_acc = (camera_pred == actions["camera"]).float().mean()
            accuracy = (keyboard_acc + camera_acc) / 2.0
            
            bc_result = {
                "loss": bc_loss,
                "accuracy": accuracy,
            }
        else:
            bc_result = self.bc_loss(policy_output, actions, action_type)
        
        # Reward prediction loss
        reward_result = self.reward_loss(reward_output, rewards, reward_head)
        
        # Combined loss
        total_loss = bc_result["loss"] + self.reward_weight * reward_result["loss"]
        
        return {
            "loss": total_loss,
            "bc_loss": bc_result["loss"],
            "bc_accuracy": bc_result["accuracy"],
            "reward_loss": reward_result["loss"],
            "reward_mse": reward_result["mse"],
        }
