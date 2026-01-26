"""
Tokenizer Loss for DreamerV4 (Equation 5)

Loss = MSE(predicted_patches, ground_truth_patches) + 0.2 × LPIPS(predicted, ground_truth)

Both losses are normalized via RMS (Root Mean Square).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenizerLoss(nn.Module):
    """
    Tokenizer loss combining MSE and LPIPS for masked autoencoding.
    
    Equation 5 from the paper:
    L_tokenizer = MSE(ẑ, z) + λ_LPIPS × LPIPS(ẑ, z)
    
    Where:
    - ẑ: Predicted patches
    - z: Ground truth patches
    - λ_LPIPS = 0.2
    - Losses normalized via RMS
    """
    
    def __init__(
        self,
        lpips_weight: float = 0.2,
        use_lpips: bool = True,
        lpips_net: str = "alex",  # "alex", "vgg", or "squeeze"
    ):
        """
        Args:
            lpips_weight: Weight for LPIPS loss (λ_LPIPS)
            use_lpips: Whether to use LPIPS loss
            lpips_net: Network to use for LPIPS
        """
        super().__init__()
        
        self.lpips_weight = lpips_weight
        self.use_lpips = use_lpips
        
        # Initialize LPIPS if requested
        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net=lpips_net, verbose=False)
                # Freeze LPIPS network
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: lpips not installed, falling back to MSE only")
                self.use_lpips = False
                self.lpips_fn = None
        else:
            self.lpips_fn = None
    
    def rms_normalize(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Normalize loss by RMS (Root Mean Square).

        This helps balance losses of different scales.
        The RMS is detached to preserve gradient flow through the loss.
        """
        rms = torch.sqrt(torch.mean(loss ** 2) + 1e-8)
        # Detach RMS to preserve gradients - we want to scale the loss
        # but not backprop through the normalization factor
        return loss / rms.detach()
    
    def compute_mse_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MSE loss, optionally only on masked patches.
        
        Args:
            predicted: Predicted patches (batch, num_patches, patch_dim)
            target: Target patches (batch, num_patches, patch_dim)
            mask: Boolean mask (batch, num_patches), True = compute loss
        
        Returns:
            MSE loss (scalar)
        """
        if mask is not None:
            # Only compute loss on masked patches
            mask = mask.unsqueeze(-1).expand_as(predicted)
            mse = F.mse_loss(predicted[mask], target[mask], reduction="mean")
        else:
            mse = F.mse_loss(predicted, target, reduction="mean")
        
        return mse
    
    def compute_lpips_loss(
        self,
        predicted_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LPIPS perceptual loss.
        
        Args:
            predicted_images: (batch, channels, height, width) in [-1, 1]
            target_images: (batch, channels, height, width) in [-1, 1]
        
        Returns:
            LPIPS loss (scalar)
        """
        if self.lpips_fn is None:
            return torch.tensor(0.0, device=predicted_images.device)
        
        # LPIPS expects input in [-1, 1]
        # Ensure proper range
        predicted_images = predicted_images.clamp(-1, 1)
        target_images = target_images.clamp(-1, 1)
        
        # Compute LPIPS
        lpips_loss = self.lpips_fn(predicted_images, target_images).mean()
        
        return lpips_loss
    
    def forward(
        self,
        predicted_patches: torch.Tensor,
        target_patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        predicted_images: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        decoder_images: Optional[torch.Tensor] = None,
        decoder_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total tokenizer loss.

        Args:
            predicted_patches: Predicted patches (batch, time, num_patches, patch_dim)
            target_patches: Target patches (batch, time, num_patches, patch_dim)
            mask: Optional mask (batch, time, num_patches)
            predicted_images: Optional reconstructed images for LPIPS (from patches)
            target_images: Optional target images for LPIPS
            decoder_images: Optional images from dedicated latent decoder (for auxiliary loss)
            decoder_weight: Weight for decoder auxiliary loss

        Returns:
            Dictionary with loss components
        """
        # Flatten time dimension if present
        if predicted_patches.dim() == 4:
            batch_size, time_steps = predicted_patches.shape[:2]
            predicted_patches = predicted_patches.reshape(batch_size * time_steps, -1, predicted_patches.shape[-1])
            target_patches = target_patches.reshape(batch_size * time_steps, -1, target_patches.shape[-1])
            if mask is not None:
                mask = mask.reshape(batch_size * time_steps, -1)

        # Compute MSE loss on patches
        mse_loss = self.compute_mse_loss(predicted_patches, target_patches, mask)

        # Compute LPIPS loss if images provided
        if self.use_lpips and predicted_images is not None and target_images is not None:
            # Flatten time dimension for images
            if predicted_images.dim() == 5:
                b, t = predicted_images.shape[:2]
                predicted_images = predicted_images.reshape(b * t, *predicted_images.shape[2:])
                target_images_flat = target_images.reshape(b * t, *target_images.shape[2:])
            else:
                target_images_flat = target_images

            # Normalize images to [-1, 1] if in [0, 1]
            pred_imgs_norm = predicted_images
            tgt_imgs_norm = target_images_flat
            if predicted_images.min() >= 0:
                pred_imgs_norm = predicted_images * 2 - 1
                tgt_imgs_norm = target_images_flat * 2 - 1

            lpips_loss = self.compute_lpips_loss(pred_imgs_norm, tgt_imgs_norm)
        else:
            lpips_loss = torch.tensor(0.0, device=predicted_patches.device)

        # Compute auxiliary decoder loss (MSE on images from dedicated latent decoder)
        decoder_loss = torch.tensor(0.0, device=predicted_patches.device)
        if decoder_images is not None and target_images is not None:
            # Flatten time dimension
            if decoder_images.dim() == 5:
                b, t = decoder_images.shape[:2]
                decoder_images_flat = decoder_images.reshape(b * t, *decoder_images.shape[2:])
                target_images_flat = target_images.reshape(b * t, *target_images.shape[2:])
            else:
                decoder_images_flat = decoder_images
                target_images_flat = target_images

            # MSE loss between decoder output and target images
            decoder_loss = F.mse_loss(decoder_images_flat, target_images_flat)

        # Normalize losses via RMS
        mse_loss_normalized = self.rms_normalize(mse_loss)
        lpips_loss_normalized = self.rms_normalize(lpips_loss) if lpips_loss.item() > 0 else lpips_loss
        decoder_loss_normalized = self.rms_normalize(decoder_loss) if decoder_loss.item() > 0 else decoder_loss

        # Total loss
        total_loss = (
            mse_loss_normalized
            + self.lpips_weight * lpips_loss_normalized
            + decoder_weight * decoder_loss_normalized
        )

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "lpips_loss": lpips_loss,
            "decoder_loss": decoder_loss,
            "mse_loss_normalized": mse_loss_normalized,
            "lpips_loss_normalized": lpips_loss_normalized,
            "decoder_loss_normalized": decoder_loss_normalized,
        }
