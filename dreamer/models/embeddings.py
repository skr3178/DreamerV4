"""
Embedding modules for DreamerV4

Contains:
- PatchEmbedding: Converts image patches to embeddings
- ActionEmbedding: Embeds discrete or continuous actions
- SignalEmbedding: Embeds signal level τ and step size d for shortcut forcing
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """
    Converts image patches to embeddings.
    
    For DreamerV4:
    - Takes input images and divides them into patches
    - Projects patches to embedding dimension
    - Used by the tokenizer to process visual input
    """
    
    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_size: int,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        """
        Args:
            image_height: Height of input images
            image_width: Width of input images
            patch_size: Size of each square patch
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches_h = image_height // patch_size
        self.num_patches_w = image_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch dimension
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Linear projection from patch to embedding
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        
        # Learnable position embeddings for spatial positions
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
    
    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            images: (batch, channels, height, width)
        
        Returns:
            patches: (batch, num_patches, patch_dim)
        """
        batch_size = images.shape[0]
        
        # Reshape to patches
        patches = rearrange(
            images,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        
        return patches
    
    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images.
        
        Args:
            patches: (batch, num_patches, patch_dim)
        
        Returns:
            images: (batch, channels, height, width)
        """
        images = rearrange(
            patches,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.num_patches_h,
            w=self.num_patches_w,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.in_channels,
        )
        return images
    
    def forward(
        self,
        images: torch.Tensor,
        add_pos_embed: bool = True,
    ) -> torch.Tensor:
        """
        Convert images to patch embeddings.
        
        Args:
            images: (batch, channels, height, width)
            add_pos_embed: Whether to add position embeddings
        
        Returns:
            embeddings: (batch, num_patches, embed_dim)
        """
        # Patchify
        patches = self.patchify(images)  # (batch, num_patches, patch_dim)
        
        # Project to embedding dimension
        embeddings = self.proj(patches)  # (batch, num_patches, embed_dim)
        
        # Add position embeddings
        if add_pos_embed:
            embeddings = embeddings + self.pos_embed
        
        return embeddings


class ActionEmbedding(nn.Module):
    """
    Embeds discrete or continuous actions.
    
    For DreamerV4:
    - Discrete actions: Embedding lookup
    - Continuous actions: Linear projection
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_discrete_actions: Optional[int] = None,
        continuous_action_dim: Optional[int] = None,
    ):
        """
        Args:
            embed_dim: Output embedding dimension
            num_discrete_actions: Number of discrete actions (if discrete)
            continuous_action_dim: Dimension of continuous actions (if continuous)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_discrete_actions = num_discrete_actions
        self.continuous_action_dim = continuous_action_dim
        
        if num_discrete_actions is not None:
            # Discrete action embedding
            self.discrete_embed = nn.Embedding(num_discrete_actions, embed_dim)
        
        if continuous_action_dim is not None:
            # Continuous action projection
            self.continuous_proj = nn.Linear(continuous_action_dim, embed_dim)
        
        # "No action" embedding for unlabeled data
        self.no_action_embed = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
    
    def forward(
        self,
        discrete_actions: Optional[torch.Tensor] = None,
        continuous_actions: Optional[torch.Tensor] = None,
        use_no_action: bool = False,
    ) -> torch.Tensor:
        """
        Embed actions.
        
        Args:
            discrete_actions: (batch, seq_len) discrete action indices
            continuous_actions: (batch, seq_len, action_dim) continuous actions
            use_no_action: If True, return "no action" embedding
        
        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        if use_no_action:
            # Return broadcasted "no action" embedding
            return self.no_action_embed
        
        if discrete_actions is not None:
            return self.discrete_embed(discrete_actions)
        
        if continuous_actions is not None:
            return self.continuous_proj(continuous_actions)
        
        raise ValueError("Must provide either discrete_actions, continuous_actions, or use_no_action=True")


class SignalEmbedding(nn.Module):
    """
    Embeds signal level τ and step size d for shortcut forcing.
    
    τ ∈ [0, 1]: Signal level (0 = pure noise, 1 = clean data)
    d ∈ {1, 1/2, 1/4, ...}: Step size for sampling
    
    Uses sinusoidal embeddings similar to diffusion timestep embeddings.
    """
    
    def __init__(self, embed_dim: int, max_period: float = 10000.0):
        """
        Args:
            embed_dim: Output embedding dimension
            max_period: Maximum period for sinusoidal embeddings
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # MLP to process combined τ and d embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
    
    def sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for continuous values.
        
        Args:
            t: (batch,) or (batch, 1) values in [0, 1]
        
        Returns:
            embeddings: (batch, embed_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) 
            * torch.arange(half_dim, device=t.device, dtype=t.dtype) 
            / half_dim
        )
        
        args = t * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Pad if embed_dim is odd
        if self.embed_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        
        return embedding
    
    def forward(
        self,
        signal_level: torch.Tensor,
        step_size: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed signal level τ and step size d.
        
        Args:
            signal_level: (batch,) τ values in [0, 1]
            step_size: (batch,) d values (step sizes)
        
        Returns:
            embeddings: (batch, embed_dim)
        """
        # Create sinusoidal embeddings for τ and d
        tau_embed = self.sinusoidal_embedding(signal_level)
        d_embed = self.sinusoidal_embedding(step_size)
        
        # Concatenate and process through MLP
        combined = torch.cat([tau_embed, d_embed], dim=-1)
        embedding = self.mlp(combined)
        
        return embedding


class LatentTokenEmbedding(nn.Module):
    """
    Learned latent tokens for the tokenizer.
    
    These tokens are used as the bottleneck representation
    in the causal tokenizer architecture.
    """
    
    def __init__(
        self,
        num_latent_tokens: int,
        embed_dim: int,
        latent_dim: int,
    ):
        """
        Args:
            num_latent_tokens: Number of latent tokens per frame
            embed_dim: Embedding dimension
            latent_dim: Dimension of latent bottleneck
        """
        super().__init__()
        
        self.num_latent_tokens = num_latent_tokens
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        # Learnable latent tokens
        self.latent_tokens = nn.Parameter(
            torch.randn(1, num_latent_tokens, embed_dim) * 0.02
        )
        
        # Bottleneck projection (to latent_dim with tanh)
        self.to_latent = nn.Linear(embed_dim, latent_dim)
        
        # Projection back from latent
        self.from_latent = nn.Linear(latent_dim, embed_dim)
    
    def to_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to low-dimensional bottleneck with tanh activation.
        
        Args:
            x: (batch, num_tokens, embed_dim)
        
        Returns:
            z: (batch, num_tokens, latent_dim)
        """
        return torch.tanh(self.to_latent(x))
    
    def from_bottleneck(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project from bottleneck back to embedding dimension.
        
        Args:
            z: (batch, num_tokens, latent_dim)
        
        Returns:
            x: (batch, num_tokens, embed_dim)
        """
        return self.from_latent(z)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get latent tokens expanded to batch size.
        
        Args:
            batch_size: Batch size
        
        Returns:
            tokens: (batch, num_latent_tokens, embed_dim)
        """
        return self.latent_tokens.expand(batch_size, -1, -1)


class RegisterTokens(nn.Module):
    """
    Learned register tokens for improving temporal consistency.
    
    As mentioned in the paper, register tokens act as "memory" 
    that helps maintain coherence in long sequences.
    """
    
    def __init__(self, num_registers: int, embed_dim: int):
        """
        Args:
            num_registers: Number of register tokens
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.num_registers = num_registers
        self.embed_dim = embed_dim
        
        # Learnable register tokens
        self.registers = nn.Parameter(
            torch.randn(1, num_registers, embed_dim) * 0.02
        )
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get register tokens expanded to batch size.
        
        Args:
            batch_size: Batch size
        
        Returns:
            registers: (batch, num_registers, embed_dim)
        """
        return self.registers.expand(batch_size, -1, -1)
