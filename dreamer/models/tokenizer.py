"""
Causal Tokenizer for DreamerV4

The tokenizer compresses raw video frames into continuous latent representations.
It uses the same block-causal transformer architecture as the dynamics model.

Key features:
- Masked autoencoding (MAE) training
- Bottleneck with tanh activation
- Block-causal attention for temporal causality
- Loss: MSE + 0.2 × LPIPS (Equation 5)
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .transformer import BlockCausalTransformer, RMSNorm
from .embeddings import PatchEmbedding, LatentTokenEmbedding, RegisterTokens


class CausalTokenizer(nn.Module):
    """
    Causal Tokenizer for DreamerV4.
    
    Architecture:
    1. Input: Video frames (B, T, C, H, W)
    2. Patchify: Convert to patches (B, T, num_patches, patch_dim)
    3. Encoder: Block-causal transformer processes patches + latent tokens
    4. Bottleneck: Project to low-dim latent with tanh
    5. Decoder: Reconstruct patches from latents
    
    The same block-causal transformer architecture is used as in the dynamics model.
    """
    
    def __init__(
        self,
        # Image dimensions
        image_height: int = 64,
        image_width: int = 64,
        in_channels: int = 3,
        patch_size: int = 8,
        # Model dimensions
        embed_dim: int = 256,
        latent_dim: int = 32,
        num_latent_tokens: int = 16,
        # Transformer config
        depth: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        # Register tokens
        num_registers: int = 4,
        # Training
        mask_ratio: float = 0.75,
    ):
        """
        Args:
            image_height: Height of input images
            image_width: Width of input images
            in_channels: Number of input channels
            patch_size: Size of each square patch
            embed_dim: Transformer embedding dimension
            latent_dim: Dimension of latent bottleneck
            num_latent_tokens: Number of latent tokens per frame
            depth: Number of transformer layers
            num_heads: Number of attention heads
            ffn_dim: FFN hidden dimension
            dropout: Dropout probability
            num_registers: Number of register tokens
            mask_ratio: Ratio of patches to mask during training
        """
        super().__init__()
        
        self.image_height = image_height
        self.image_width = image_width
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.mask_ratio = mask_ratio
        
        # Calculate patch counts
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # Latent tokens
        self.latent_tokens = LatentTokenEmbedding(
            num_latent_tokens=num_latent_tokens,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
        )
        
        # Register tokens for temporal consistency
        self.register_tokens = RegisterTokens(
            num_registers=num_registers,
            embed_dim=embed_dim,
        )
        
        # Mask token for masked patches
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Unified block-causal transformer (encoder)
        self.transformer = BlockCausalTransformer(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_qk_norm=True,
        )
        
        # Decoder projection (from embed_dim to patch_dim for reconstruction)
        self.decoder_norm = RMSNorm(embed_dim)
        self.decoder_proj = nn.Linear(embed_dim, self.patch_dim)
        
        # Block size for attention mask (patches + latents + registers per frame)
        self.tokens_per_frame = self.num_patches + num_latent_tokens + num_registers
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        # Initialize decoder projection to small values
        nn.init.normal_(self.decoder_proj.weight, std=0.02)
        nn.init.zeros_(self.decoder_proj.bias)
    
    def random_masking(
        self,
        patches: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches for MAE training.
        
        Args:
            patches: (batch, num_patches, embed_dim)
            mask_ratio: Fraction of patches to mask
        
        Returns:
            masked_patches: Patches with some replaced by mask token
            mask: Boolean mask (True = masked)
            ids_restore: Indices to restore original order
        """
        batch_size, num_patches, dim = patches.shape
        num_mask = int(num_patches * mask_ratio)
        
        # Random permutation for each batch element
        noise = torch.rand(batch_size, num_patches, device=patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Create mask: True = keep, False = mask
        mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=patches.device)
        mask[:, :num_mask] = False
        
        # Unshuffle mask to original positions
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Apply mask token to masked positions
        masked_patches = patches.clone()
        mask_tokens = self.mask_token.expand(batch_size, num_patches, -1)
        masked_patches = torch.where(
            mask.unsqueeze(-1),
            patches,
            mask_tokens,
        )
        
        return masked_patches, ~mask, ids_restore  # Return ~mask so True = masked
    
    def encode_frame(
        self,
        images: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a single frame (or batch of frames) to latent tokens.
        
        Args:
            images: (batch, channels, height, width)
            mask_ratio: If provided, apply random masking for training
        
        Returns:
            Dictionary containing:
                - latents: (batch, num_latent_tokens, latent_dim)
                - patches: Original patch embeddings
                - mask: Mask applied (if training)
                - reconstructed: Reconstructed patches
        """
        batch_size = images.shape[0]
        
        # 1. Patchify and embed
        patch_embeds = self.patch_embed(images)  # (batch, num_patches, embed_dim)
        
        # 2. Apply random masking if training
        mask = None
        if mask_ratio is not None and mask_ratio > 0:
            patch_embeds, mask, _ = self.random_masking(patch_embeds, mask_ratio)
        
        # 3. Get latent tokens and register tokens
        latent_tokens = self.latent_tokens(batch_size)  # (batch, num_latent, embed_dim)
        register_tokens = self.register_tokens(batch_size)  # (batch, num_reg, embed_dim)
        
        # 4. Concatenate: [patches, latent_tokens, registers]
        tokens = torch.cat([patch_embeds, latent_tokens, register_tokens], dim=1)
        
        # 5. Process through transformer
        # For single frame, we don't need block-causal masking
        tokens = self.transformer(tokens)
        
        # 6. Extract latent tokens and apply bottleneck
        latent_start = self.num_patches
        latent_end = latent_start + self.num_latent_tokens
        latent_output = tokens[:, latent_start:latent_end]
        latents = self.latent_tokens.to_bottleneck(latent_output)  # (batch, num_latent, latent_dim)
        
        # 7. Decode patches for reconstruction
        patch_output = tokens[:, :self.num_patches]
        reconstructed = self.decoder_proj(self.decoder_norm(patch_output))
        
        return {
            "latents": latents,
            "reconstructed": reconstructed,
            "mask": mask,
        }
    
    def encode(
        self,
        video: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode video frames to latent tokens.
        
        Args:
            video: (batch, channels, time, height, width) or (batch, time, channels, height, width)
            mask_ratio: Masking ratio for training
        
        Returns:
            Dictionary with latents and reconstruction info
        """
        # Handle different input formats
        if video.dim() == 5:
            if video.shape[1] == self.in_channels:
                # (batch, channels, time, height, width) -> (batch, time, channels, height, width)
                video = video.permute(0, 2, 1, 3, 4)
        
        batch_size, time_steps, channels, height, width = video.shape
        
        # Process each frame
        all_latents = []
        all_reconstructed = []
        all_masks = []
        
        for t in range(time_steps):
            frame = video[:, t]  # (batch, channels, height, width)
            result = self.encode_frame(frame, mask_ratio=mask_ratio)
            all_latents.append(result["latents"])
            all_reconstructed.append(result["reconstructed"])
            if result["mask"] is not None:
                all_masks.append(result["mask"])
        
        # Stack along time dimension
        latents = torch.stack(all_latents, dim=1)  # (batch, time, num_latent, latent_dim)
        reconstructed = torch.stack(all_reconstructed, dim=1)  # (batch, time, num_patches, patch_dim)
        
        result = {
            "latents": latents,
            "reconstructed": reconstructed,
        }
        
        if all_masks:
            result["mask"] = torch.stack(all_masks, dim=1)
        
        return result
    
    def decode_patches(self, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Convert reconstructed patches back to images.
        
        Args:
            reconstructed: (batch, num_patches, patch_dim)
        
        Returns:
            images: (batch, channels, height, width)
        """
        return self.patch_embed.unpatchify(reconstructed)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tokens back to images.
        
        Note: This requires running through the transformer again with
        the latents to reconstruct patches.
        
        Args:
            latents: (batch, num_latent_tokens, latent_dim)
        
        Returns:
            images: (batch, channels, height, width)
        """
        batch_size = latents.shape[0]
        
        # Project latents back to embed_dim
        latent_embeds = self.latent_tokens.from_bottleneck(latents)
        
        # Use mask tokens for patches (we don't have patch info in decode-only)
        patch_embeds = self.mask_token.expand(batch_size, self.num_patches, -1)
        patch_embeds = patch_embeds + self.patch_embed.pos_embed
        
        # Get register tokens
        register_tokens = self.register_tokens(batch_size)
        
        # Concatenate and process
        tokens = torch.cat([patch_embeds, latent_embeds, register_tokens], dim=1)
        tokens = self.transformer(tokens)
        
        # Extract and decode patches
        patch_output = tokens[:, :self.num_patches]
        reconstructed = self.decoder_proj(self.decoder_norm(patch_output))
        
        return self.decode_patches(reconstructed)
    
    def forward(
        self,
        video: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            video: Input video (batch, channels, time, height, width)
            mask_ratio: Masking ratio (default: self.mask_ratio)
        
        Returns:
            Dictionary with latents, reconstructed patches, and original patches
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Get original patches for loss computation
        if video.dim() == 5 and video.shape[1] == self.in_channels:
            video_reordered = video.permute(0, 2, 1, 3, 4)
        else:
            video_reordered = video
        
        batch_size, time_steps = video_reordered.shape[:2]
        
        # Get original patches
        original_patches = []
        for t in range(time_steps):
            frame = video_reordered[:, t]
            patches = self.patch_embed.patchify(frame)
            original_patches.append(patches)
        original_patches = torch.stack(original_patches, dim=1)
        
        # Encode
        result = self.encode(video, mask_ratio=mask_ratio)
        result["original_patches"] = original_patches
        
        return result
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
