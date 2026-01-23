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

from .transformer import BlockCausalTransformer, RMSNorm, create_block_causal_mask
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

        # Dedicated latent-to-image decoder for decode-only path
        # This bypasses the transformer and directly decodes from latent bottleneck
        # Architecture: latents (num_latent × latent_dim) → CNN → image (C × H × W)
        # Reshape latents to spatial grid: sqrt(num_latent) × sqrt(num_latent) × latent_dim
        self.latent_spatial_size = int(math.sqrt(num_latent_tokens))  # e.g., 4 for 16 tokens
        assert self.latent_spatial_size ** 2 == num_latent_tokens, \
            f"num_latent_tokens must be a perfect square, got {num_latent_tokens}"

        # CNN decoder: 4×4 → 8×8 → 16×16 → 32×32 → 64×64
        decoder_channels = embed_dim  # Use embed_dim as base channel count
        self.latent_decoder = nn.Sequential(
            # Project latent_dim to decoder channels
            nn.Conv2d(latent_dim, decoder_channels, kernel_size=1),
            nn.SiLU(),
            # 4×4 → 8×8
            nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # 8×8 → 16×16
            nn.ConvTranspose2d(decoder_channels // 2, decoder_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # 16×16 → 32×32
            nn.ConvTranspose2d(decoder_channels // 4, decoder_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # 32×32 → 64×64
            nn.ConvTranspose2d(decoder_channels // 8, in_channels, kernel_size=4, stride=2, padding=1),
        )

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        # Initialize decoder projection to small values
        nn.init.normal_(self.decoder_proj.weight, std=0.02)
        nn.init.zeros_(self.decoder_proj.bias)

        # Initialize latent decoder CNN
        for m in self.latent_decoder.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
        Encode video frames to latent tokens using block-causal attention across time.
        
        This method processes all frames together with block-causal attention, allowing
        tokens at time t to attend to all tokens at times <= t, ensuring temporal consistency.
        
        Args:
            video: (batch, channels, time, height, width) or (batch, time, channels, height, width)
            mask_ratio: Masking ratio for training
        
        Returns:
            Dictionary with latents and reconstruction info
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Handle different input formats
        if video.dim() == 5:
            if video.shape[1] == self.in_channels:
                # (batch, channels, time, height, width) -> (batch, time, channels, height, width)
                video = video.permute(0, 2, 1, 3, 4)
        
        batch_size, time_steps, channels, height, width = video.shape
        device = video.device
        
        # Build full sequence with block-causal structure
        # Sequence per timestep: [patches, latent_tokens, register_tokens]
        all_tokens = []
        all_original_patches = []
        all_masks = []
        
        for t in range(time_steps):
            frame = video[:, t]  # (batch, channels, height, width)
            
            # 1. Patchify and embed
            patch_embeds = self.patch_embed(frame)  # (batch, num_patches, embed_dim)
            
            # Store original patches for loss computation
            original_patches = self.patch_embed.patchify(frame)  # (batch, num_patches, patch_dim)
            all_original_patches.append(original_patches)
            
            # 2. Apply random masking if training
            mask = None
            if mask_ratio is not None and mask_ratio > 0:
                patch_embeds, mask, _ = self.random_masking(patch_embeds, mask_ratio)
            
            if mask is not None:
                all_masks.append(mask)
            
            # 3. Get latent tokens and register tokens for this timestep
            latent_tokens = self.latent_tokens(batch_size)  # (batch, num_latent, embed_dim)
            register_tokens = self.register_tokens(batch_size)  # (batch, num_reg, embed_dim)
            
            # 4. Concatenate tokens for this timestep: [patches, latent_tokens, registers]
            timestep_tokens = torch.cat([patch_embeds, latent_tokens, register_tokens], dim=1)
            all_tokens.append(timestep_tokens)
        
        # Stack all timestep tokens into full sequence
        # Shape: (batch, time_steps * tokens_per_frame, embed_dim)
        full_sequence = torch.cat(all_tokens, dim=1)
        
        # Create block-causal attention mask
        # Each block corresponds to one timestep's tokens
        block_size = self.tokens_per_frame
        attention_mask = create_block_causal_mask(
            seq_len=full_sequence.shape[1],
            block_size=block_size,
            device=device,
        )
        
        # Process through transformer with block-causal attention
        output = self.transformer(full_sequence, attention_mask=attention_mask)
        
        # Extract latents and reconstructed patches for each timestep
        all_latents = []
        all_reconstructed = []
        
        for t in range(time_steps):
            # Calculate indices for this timestep's tokens
            step_start = t * self.tokens_per_frame
            patch_start = step_start
            patch_end = patch_start + self.num_patches
            latent_start = patch_end
            latent_end = latent_start + self.num_latent_tokens
            
            # Extract patch output and reconstruct
            patch_output = output[:, patch_start:patch_end]  # (batch, num_patches, embed_dim)
            reconstructed = self.decoder_proj(self.decoder_norm(patch_output))  # (batch, num_patches, patch_dim)
            all_reconstructed.append(reconstructed)
            
            # Extract latent tokens and apply bottleneck
            latent_output = output[:, latent_start:latent_end]  # (batch, num_latent, embed_dim)
            latents = self.latent_tokens.to_bottleneck(latent_output)  # (batch, num_latent, latent_dim)
            all_latents.append(latents)
        
        # Stack along time dimension
        latents = torch.stack(all_latents, dim=1)  # (batch, time, num_latent, latent_dim)
        reconstructed = torch.stack(all_reconstructed, dim=1)  # (batch, time, num_patches, patch_dim)
        original_patches = torch.stack(all_original_patches, dim=1)  # (batch, time, num_patches, patch_dim)
        
        result = {
            "latents": latents,
            "reconstructed": reconstructed,
            "original_patches": original_patches,
        }

        if all_masks:
            result["mask"] = torch.stack(all_masks, dim=1)  # (batch, time, num_patches)

        # Also decode latents through the dedicated CNN decoder for auxiliary training
        # This trains the decoder to reconstruct images from the latent bottleneck
        decoder_images = []
        for t in range(time_steps):
            latent_t = latents[:, t]  # (batch, num_latent, latent_dim)
            decoded_t = self.decode(latent_t)  # (batch, C, H, W)
            decoder_images.append(decoded_t)
        result["decoder_images"] = torch.stack(decoder_images, dim=1)  # (batch, time, C, H, W)

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
        Decode latent tokens back to images using dedicated CNN decoder.

        This method uses a learned CNN decoder that directly maps from the
        latent bottleneck to image space, bypassing the transformer. This is
        necessary because the transformer-based reconstruction requires visible
        patches as context, which aren't available in decode-only scenarios
        (e.g., decoding dynamics model predictions).

        Args:
            latents: (batch, num_latent_tokens, latent_dim)

        Returns:
            images: (batch, channels, height, width)
        """
        batch_size = latents.shape[0]

        # Reshape latents to spatial grid: (B, num_latent, latent_dim) → (B, latent_dim, H, W)
        # where H = W = sqrt(num_latent_tokens)
        latents_spatial = latents.permute(0, 2, 1)  # (B, latent_dim, num_latent)
        latents_spatial = latents_spatial.view(
            batch_size,
            self.latent_dim,
            self.latent_spatial_size,
            self.latent_spatial_size
        )  # (B, latent_dim, 4, 4)

        # Decode through CNN
        images = self.latent_decoder(latents_spatial)  # (B, C, H, W)

        return images

    def decode_transformer(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Original transformer-based decode (for reference/comparison).

        WARNING: This method uses mask tokens for all patches and may not
        produce good reconstructions without visible patch context.

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
        # Encode with block-causal attention across time
        # The encode method now handles everything including original_patches
        result = self.encode(video, mask_ratio=mask_ratio)
        
        return result
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
