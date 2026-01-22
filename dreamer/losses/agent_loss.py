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
    Supports focal loss to handle imbalanced reward distributions.
    """
    
    def __init__(
        self,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        """
        Args:
            use_focal_loss: Whether to use focal loss for imbalanced rewards
            focal_gamma: Focal loss focusing parameter (higher = more focus on hard examples)
            focal_alpha: Focal loss balancing parameter
        """
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
    
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
        
        if self.use_focal_loss:
            # Focal loss: down-weight easy (common) examples
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Focal weight: (1 - p_t)^gamma
            # For correct class: p_t is the predicted probability of the target bin
            p_t = (probs * target_bins).sum(dim=-1, keepdim=True)  # (B*T, 1)
            focal_weight = (1 - p_t) ** self.focal_gamma
            
            # Focal loss: -alpha * (1-p_t)^gamma * log(p_t)
            loss = -(self.focal_alpha * focal_weight * (target_bins * log_probs).sum(dim=-1)).mean()
        else:
            # Standard cross-entropy loss
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(target_bins * log_probs).sum(dim=-1).mean()
        
        # Compute MSE for logging
        mse = F.mse_loss(predicted_rewards, target_rewards)
        
        # Compute reward statistics for monitoring
        nonzero_rewards = (target_rewards != 0).sum().float()
        total_rewards = target_rewards.numel()
        nonzero_ratio = nonzero_rewards / total_rewards if total_rewards > 0 else 0.0
        
        return {
            "loss": loss,
            "mse": mse,
            "nonzero_ratio": nonzero_ratio,
            "pred_std": predicted_rewards.std(),
        }


class AgentFinetuningLoss(nn.Module):
    """
    Combined loss for agent finetuning (Equation 9).
    
    Paper's Equation (9):
    L(θ) = - Σ_{n=0}^{L} ln p_θ(a_{t+n} | h_t) - Σ_{n=0}^{L} ln p_θ(r_{t+n} | h_t)
    
    Where:
    - h_t: Task output embedding at time t
    - L = 8: Multi-token prediction length
    - n=0 to L: Predicts current timestep (n=0) plus L future timesteps (n=1 to L)
    - Both terms have equal weight (no λ_reward)
    
    Supports both:
    - MTP mode (default): True multi-token prediction matching the paper
    - Standard mode: Per-timestep prediction (backward compatible)
    """
    
    def __init__(
        self,
        reward_weight: float = 1.0,
        num_prediction_steps: int = 8,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        use_mtp: bool = True,
        use_auxiliary_reconstruction: bool = True,
        reconstruction_weight: float = 0.1,
    ):
        """
        Args:
            reward_weight: Weight for reward prediction loss (only used in standard mode)
            num_prediction_steps: Number of future steps to predict (L in paper)
            use_focal_loss: Whether to use focal loss for imbalanced rewards
            focal_gamma: Focal loss focusing parameter
            focal_alpha: Focal loss balancing parameter
            use_mtp: Whether to use Multi-Token Prediction (matches paper Equation 9)
            use_auxiliary_reconstruction: Whether to use auxiliary reconstruction loss (Section 4.3)
            reconstruction_weight: Weight for auxiliary reconstruction loss
        """
        super().__init__()
        self.reward_weight = reward_weight
        self.mtp_length = num_prediction_steps  # L in paper
        self.use_mtp = use_mtp
        self.use_auxiliary_reconstruction = use_auxiliary_reconstruction
        self.reconstruction_weight = reconstruction_weight
        
        # For standard mode (backward compatibility)
        if not use_mtp:
            self.bc_loss = BehaviorCloningLoss(num_prediction_steps)
            self.reward_loss = RewardPredictionLoss(
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
            )
        else:
            # MTP mode uses focal loss if specified
            self.reward_loss = RewardPredictionLoss(
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
            )
    
    def forward(
        self,
        policy_head: nn.Module,
        reward_head: nn.Module,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        action_type: str = "discrete",
        tokenizer: Optional[nn.Module] = None,
        target_frames: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined agent finetuning loss.
        
        Args:
            policy_head: PolicyHead or MultiDiscretePolicyHead
            reward_head: RewardHead
            latents: Latent states 
                - Standard mode: (batch, time, input_dim)
                - MTP mode: (batch, time, input_dim) - will extract h_t for each t
            actions: Target actions (batch, time, ...) or Dict for multi-discrete
            rewards: Target rewards (batch, time)
            action_type: "discrete", "continuous", or "multi_discrete"
            tokenizer: Optional tokenizer for auxiliary reconstruction loss (Section 4.3)
            target_frames: Optional target frames for reconstruction loss
        
        Returns:
            Dictionary with all loss components
        """
        if self.use_mtp:
            result = self._forward_mtp(
                policy_head, reward_head, latents, actions, rewards, action_type
            )
        else:
            result = self._forward_standard(
                policy_head, reward_head, latents, actions, rewards, action_type
            )
        
        # Add auxiliary reconstruction loss (Section 4.3)
        if self.use_auxiliary_reconstruction and tokenizer is not None and target_frames is not None:
            recon_loss = self._compute_reconstruction_loss(
                tokenizer, latents, target_frames
            )
            result["loss"] = result["loss"] + self.reconstruction_weight * recon_loss
            result["reconstruction_loss"] = recon_loss
        
        return result
    
    def _compute_reconstruction_loss(
        self,
        tokenizer: nn.Module,
        latents: torch.Tensor,
        target_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary reconstruction loss (Section 4.3).
        
        Reconstructs frames from latents to ensure latents maintain good representations
        during phase 2 training.
        
        Args:
            tokenizer: Tokenizer module (frozen)
            latents: Flattened latents (batch, time, num_latent * latent_dim)
            target_frames: Target frames (batch, time, C, H, W) or (batch, C, T, H, W)
        
        Returns:
            Reconstruction loss (MSE)
        """
        # Reshape latents back to (batch, time, num_latent, latent_dim)
        # We need to know num_latent and latent_dim from tokenizer
        num_latent = tokenizer.num_latent_tokens
        latent_dim = tokenizer.latent_dim
        
        batch_size, time_steps = latents.shape[:2]
        latents_reshaped = latents.reshape(batch_size, time_steps, num_latent, latent_dim)
        
        # Decode latents to frames using tokenizer decoder
        # Note: This assumes tokenizer has a decode method
        # If not, we'll need to reconstruct patches and then unpatchify
        with torch.no_grad():
            # Get target patches from tokenizer
            if hasattr(tokenizer, 'patch_embed'):
                # Reshape frames if needed
                if target_frames.dim() == 5 and target_frames.shape[2] == 3:
                    # (B, C, T, H, W) -> (B, T, C, H, W)
                    target_frames = target_frames.permute(0, 2, 1, 3, 4)
                
                # Get patches for target frames
                target_patches = []
                for t in range(time_steps):
                    frame = target_frames[:, t]  # (B, C, H, W)
                    patches = tokenizer.patch_embed.patchify(frame)  # (B, num_patches, patch_dim)
                    target_patches.append(patches)
                target_patches = torch.stack(target_patches, dim=1)  # (B, T, num_patches, patch_dim)
            else:
                # Fallback: use MSE on latents directly
                return F.mse_loss(latents_reshaped, latents_reshaped.detach())
        
        # Reconstruct frames from latents using tokenizer decoder
        # Decode each timestep's latents to frames
        reconstructed_frames = []
        for t in range(time_steps):
            frame_latents = latents_reshaped[:, t]  # (batch, num_latent, latent_dim)
            # Use tokenizer's decode method to reconstruct frames
            if hasattr(tokenizer, 'decode'):
                reconstructed_frame = tokenizer.decode(frame_latents)  # (batch, C, H, W)
                reconstructed_frames.append(reconstructed_frame)
            else:
                # Fallback: return zero loss if decode not available
                return torch.tensor(0.0, device=latents.device)
        
        reconstructed_frames = torch.stack(reconstructed_frames, dim=1)  # (batch, time, C, H, W)
        
        # Ensure target_frames are in correct format
        if target_frames.dim() == 5 and target_frames.shape[1] == 3:
            # (B, C, T, H, W) -> (B, T, C, H, W)
            target_frames = target_frames.permute(0, 2, 1, 3, 4)
        elif target_frames.dim() == 4:
            # (B, C, H, W) -> (B, 1, C, H, W)
            target_frames = target_frames.unsqueeze(1)
        
        # Normalize target frames to [0, 1] if needed
        if target_frames.max() > 1.0:
            target_frames = target_frames.float() / 255.0
        
        # Compute MSE loss between reconstructed and target frames
        recon_loss = F.mse_loss(reconstructed_frames, target_frames)
        
        return recon_loss
    
    def _forward_mtp(
        self,
        policy_head: nn.Module,
        reward_head: nn.Module,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        action_type: str,
    ) -> Dict[str, torch.Tensor]:
        """
        MTP forward pass matching Equation (9).
        
        For each timestep t:
        1. Extract h_t = latents[t]
        2. Predict a_{t+n} and r_{t+n} for n=0 to L
        3. Compute loss: -Σ_{n=0}^{L} ln p(a_{t+n}|h_t) - Σ_{n=0}^{L} ln p(r_{t+n}|h_t)
        
        This matches the paper's Equation (9) exactly:
        L(θ) = - Σ_{n=0}^{L} ln p_θ(a_{t+n} | h_t) - Σ_{n=0}^{L} ln p_θ(r_{t+n} | h_t)
        """
        batch_size, time_steps, input_dim = latents.shape
        device = latents.device
        
        # Batch all h_t embeddings for efficient processing
        # Reshape latents: (batch, time, input_dim) -> (batch * time, input_dim)
        all_h_t = latents.reshape(-1, input_dim)  # (batch * time, input_dim)
        
        # Predict all actions and rewards in batch
        policy_output = policy_head(all_h_t)  # (batch * time, L+1, num_actions)
        reward_output = reward_head(all_h_t)  # (batch * time, L+1, num_bins)
        
        # Reshape back: (batch * time, L+1, ...) -> (batch, time, L+1, ...)
        policy_output_reshaped = {}
        for key, value in policy_output.items():
            policy_output_reshaped[key] = value.reshape(batch_size, time_steps, *value.shape[1:])
        
        reward_output_reshaped = {}
        for key, value in reward_output.items():
            reward_output_reshaped[key] = value.reshape(batch_size, time_steps, *value.shape[1:])
        
        all_bc_losses = []
        all_reward_losses = []
        all_accuracies = []
        all_pred_rewards = []
        all_target_rewards = []
        
        # Compute loss for each (t, n) pair where t+n < time_steps
        for t in range(time_steps):
            for n in range(self.mtp_length + 1):
                if t + n >= time_steps:
                    break  # Can't compute loss for future timesteps beyond sequence
                
                # Get target action and reward at timestep t+n
                target_action = actions[:, t + n]  # (batch,)
                target_reward = rewards[:, t + n]  # (batch,)
                
                # Get predicted outputs for timestep t, prediction n
                if action_type == "discrete":
                    pred_logits = policy_output_reshaped["logits"][:, t, n]  # (batch, num_actions)
                    
                    # Behavior cloning loss: -ln p(a_{t+n} | h_t)
                    bc_loss = F.cross_entropy(pred_logits, target_action, reduction="mean")
                    all_bc_losses.append(bc_loss)
                    
                    # Accuracy
                    pred_action = pred_logits.argmax(dim=-1)
                    accuracy = (pred_action == target_action).float().mean()
                    all_accuracies.append(accuracy)
                    
                elif action_type == "continuous":
                    pred_mean = policy_output_reshaped["mean"][:, t, n]  # (batch, action_dim)
                    pred_std = policy_output_reshaped["std"][:, t, n]  # (batch, action_dim)
                    
                    # Gaussian NLL: -ln p(a_{t+n} | h_t)
                    var = pred_std ** 2
                    log_prob = -0.5 * (
                        ((target_action - pred_mean) ** 2) / var +
                        torch.log(var) +
                        torch.log(torch.tensor(2 * 3.14159265, device=device))
                    )
                    bc_loss = -log_prob.sum(dim=-1).mean()
                    all_bc_losses.append(bc_loss)
                    all_accuracies.append(torch.tensor(0.0, device=device))
                
                # Reward prediction loss: -ln p(r_{t+n} | h_t)
                pred_logits = reward_output_reshaped["logits"][:, t, n]  # (batch, num_bins)
                target_bins = reward_head.target_to_bins(target_reward)  # (batch, num_bins)
                
                log_probs = F.log_softmax(pred_logits, dim=-1)
                reward_loss = -(target_bins * log_probs).sum(dim=-1).mean()
                all_reward_losses.append(reward_loss)
                
                # Collect for MSE computation
                pred_reward = reward_output_reshaped["reward"][:, t, n]  # (batch,)
                all_pred_rewards.append(pred_reward)
                all_target_rewards.append(target_reward)
        
        # Average losses over all valid (t, n) pairs
        bc_loss = torch.stack(all_bc_losses).mean() if all_bc_losses else torch.tensor(0.0, device=device)
        reward_loss = torch.stack(all_reward_losses).mean() if all_reward_losses else torch.tensor(0.0, device=device)
        accuracy = torch.stack(all_accuracies).mean() if all_accuracies else torch.tensor(0.0, device=device)
        
        # Combined loss: L_BC + L_reward (equal weights as per Equation 9)
        total_loss = bc_loss + reward_loss
        
        # Compute reward statistics
        nonzero_rewards = (rewards != 0).sum().float()
        total_rewards = rewards.numel()
        nonzero_ratio = nonzero_rewards / total_rewards if total_rewards > 0 else 0.0
        
        # Compute predicted reward statistics
        if all_pred_rewards:
            pred_rewards_tensor = torch.stack(all_pred_rewards)  # (num_predictions, batch)
            target_rewards_tensor = torch.stack(all_target_rewards)  # (num_predictions, batch)
            pred_std = pred_rewards_tensor.flatten().std()
            reward_mse = F.mse_loss(pred_rewards_tensor, target_rewards_tensor)
        else:
            pred_std = torch.tensor(0.0, device=device)
            reward_mse = torch.tensor(0.0, device=device)
        
        return {
            "loss": total_loss,
            "bc_loss": bc_loss,
            "bc_accuracy": accuracy,
            "reward_loss": reward_loss,
            "reward_mse": reward_mse,
            "nonzero_ratio": nonzero_ratio,
            "pred_std": pred_std,
        }
    
    def _forward_standard(
        self,
        policy_head: nn.Module,
        reward_head: nn.Module,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        action_type: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass (backward compatible).
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
            "nonzero_ratio": reward_result["nonzero_ratio"],
            "pred_std": reward_result["pred_std"],
        }
