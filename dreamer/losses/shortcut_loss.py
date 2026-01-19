"""
Shortcut Forcing Loss for DreamerV4 (Equation 7)

The shortcut forcing objective enables efficient generation with few sampling steps.

Key equations:
- Eq. 7: Full shortcut forcing loss
- Eq. 8: Ramp loss weight w(τ) = 0.9τ + 0.1
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortcutForcingLoss(nn.Module):
    """
    Shortcut Forcing Loss for dynamics model training.
    
    Equation 7 from the paper:
    L_shortcut = {
        ||ẑ₁ - z₁||²                    if d = d_min
        w(τ) · ||ẑ₁ - z_bootstrap||²    otherwise
    }
    
    Where:
    - z₁: Clean target latents
    - ẑ₁: Predicted clean latents
    - z_bootstrap: Bootstrapped target from Equation 3
    - w(τ) = 0.9τ + 0.1: Ramp loss weight (Equation 8)
    - d_min: Minimum step size
    
    The model predicts clean latents directly (x-prediction) rather than
    velocities (v-prediction) to avoid error accumulation in long rollouts.
    """
    
    def __init__(
        self,
        ramp_min: float = 0.1,
        ramp_scale: float = 0.9,
    ):
        """
        Args:
            ramp_min: Minimum weight in ramp function
            ramp_scale: Scale factor for ramp function
        """
        super().__init__()
        self.ramp_min = ramp_min
        self.ramp_scale = ramp_scale
    
    def ramp_weight(self, signal_level: torch.Tensor) -> torch.Tensor:
        """
        Compute ramp loss weight (Equation 8).
        
        w(τ) = 0.9τ + 0.1
        
        Higher weight when τ is larger (less noise, more informative).
        This focuses learning on semantically richer timesteps.
        
        Args:
            signal_level: τ values in [0, 1]
        
        Returns:
            Weights in [ramp_min, ramp_min + ramp_scale]
        """
        return self.ramp_scale * signal_level + self.ramp_min
    
    def compute_mse_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            predicted: Predicted latents
            target: Target latents
            weights: Optional per-sample weights
        
        Returns:
            Weighted MSE loss
        """
        # Compute per-sample MSE
        mse = F.mse_loss(predicted, target, reduction="none")
        
        # Average over all dimensions except batch
        mse = mse.mean(dim=tuple(range(1, mse.dim())))  # (batch,)
        
        # Apply weights if provided
        if weights is not None:
            mse = mse * weights
        
        return mse.mean()
    
    def forward(
        self,
        predicted_latents: torch.Tensor,
        target_latents: torch.Tensor,
        signal_level: torch.Tensor,
        step_size: torch.Tensor,
        d_is_min: torch.Tensor,
        bootstrap_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute shortcut forcing loss.
        
        Args:
            predicted_latents: Predicted clean latents (batch, time, num_latent, latent_dim)
            target_latents: Clean target latents (batch, time, num_latent, latent_dim)
            signal_level: τ values (batch,)
            step_size: d values (batch,)
            d_is_min: Boolean indicating if d is minimum (batch,)
            bootstrap_targets: Optional bootstrapped targets for non-minimum d
        
        Returns:
            Dictionary with loss components
        """
        batch_size = predicted_latents.shape[0]
        device = predicted_latents.device
        
        # Compute ramp weights
        weights = self.ramp_weight(signal_level)
        
        # Split batch into d_min and non-d_min samples
        d_min_mask = d_is_min.float()
        d_other_mask = 1.0 - d_min_mask
        
        # For d = d_min: use clean targets directly
        # For d > d_min: use bootstrapped targets (or clean targets if not provided)
        if bootstrap_targets is None:
            # If no bootstrap targets, use clean targets for all
            effective_targets = target_latents
        else:
            # Mix targets based on d_min mask
            d_min_mask_expanded = d_min_mask.view(batch_size, 1, 1, 1)
            effective_targets = (
                d_min_mask_expanded * target_latents + 
                (1 - d_min_mask_expanded) * bootstrap_targets
            )
        
        # Compute weighted MSE loss
        # For d_min: weight = 1 (no ramp)
        # For others: weight = w(τ)
        effective_weights = d_min_mask * 1.0 + d_other_mask * weights
        
        loss = self.compute_mse_loss(predicted_latents, effective_targets, effective_weights)
        
        # Compute component losses for logging
        d_min_loss = torch.tensor(0.0, device=device)
        d_other_loss = torch.tensor(0.0, device=device)
        
        if d_min_mask.sum() > 0:
            d_min_indices = d_is_min.nonzero(as_tuple=True)[0]
            d_min_loss = F.mse_loss(
                predicted_latents[d_min_indices],
                target_latents[d_min_indices],
            )
        
        if d_other_mask.sum() > 0:
            d_other_indices = (~d_is_min).nonzero(as_tuple=True)[0]
            d_other_loss = F.mse_loss(
                predicted_latents[d_other_indices],
                effective_targets[d_other_indices],
            )
        
        return {
            "loss": loss,
            "d_min_loss": d_min_loss,
            "d_other_loss": d_other_loss,
            "mean_weight": effective_weights.mean(),
            "mean_signal_level": signal_level.mean(),
            "mean_step_size": step_size.mean(),
        }


class BootstrapTargetComputer(nn.Module):
    """
    Computes bootstrap targets for shortcut forcing (Equation 3).
    
    For larger steps, the target is the average of two half-steps:
    v_target = (v₁ + v₂) / 2
    
    Where v₁ and v₂ are predictions with stop-gradient.
    This distills knowledge so the model can take bigger jumps.
    """
    
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def compute_bootstrap_target(
        self,
        model,
        noisy_latents: torch.Tensor,
        actions: torch.Tensor,
        signal_level: torch.Tensor,
        step_size: torch.Tensor,
        discrete_actions: bool = True,
    ) -> torch.Tensor:
        """
        Compute bootstrap targets using two half-steps.
        
        Args:
            model: Dynamics model
            noisy_latents: Noisy input latents
            actions: Actions
            signal_level: τ values
            step_size: d values (will be halved)
            discrete_actions: Whether actions are discrete
        
        Returns:
            Bootstrap target latents
        """
        # Half step size
        half_step = step_size / 2
        
        # First half step
        result1 = model(
            noisy_latents,
            actions,
            signal_level=signal_level,
            step_size=half_step,
            discrete_actions=discrete_actions,
            add_noise_to_latents=False,
        )
        pred1 = result1["predicted_latents"]
        
        # Intermediate latents after first half step
        intermediate = noisy_latents + half_step.view(-1, 1, 1, 1) * (pred1 - noisy_latents)
        
        # Second half step
        result2 = model(
            intermediate,
            actions,
            signal_level=signal_level + half_step,
            step_size=half_step,
            discrete_actions=discrete_actions,
            add_noise_to_latents=False,
        )
        pred2 = result2["predicted_latents"]
        
        # Average of two half-step predictions
        bootstrap_target = (pred1 + pred2) / 2
        
        return bootstrap_target
