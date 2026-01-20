"""
Dynamics Model for DreamerV4

The dynamics model predicts future latent states given past latents and actions.
It uses the same block-causal transformer architecture as the tokenizer.

Key features:
- Interleaved action + latent sequence processing
- Shortcut forcing with x-prediction (predicts clean latents directly)
- Conditioned on signal level τ and step size d
- Same transformer architecture as tokenizer (unified design)
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .transformer import BlockCausalTransformer, RMSNorm, create_block_causal_mask
from .embeddings import ActionEmbedding, SignalEmbedding, RegisterTokens


class DynamicsModel(nn.Module):
    """
    Interactive Dynamics Model for DreamerV4.
    
    Predicts future latent states given past latents and actions using
    the shortcut forcing objective for efficient generation.
    
    Architecture:
    - Input: Interleaved sequence of [action_t, τ, d, z̃_t] per timestep
    - Processing: Block-causal transformer (same arch as tokenizer)
    - Output: Predicted clean latent ẑ₁
    
    The model uses x-prediction (predicts clean latents directly) instead of
    v-prediction (velocities) to avoid error accumulation in long rollouts.
    """
    
    def __init__(
        self,
        # Latent dimensions
        latent_dim: int = 32,
        num_latent_tokens: int = 16,
        # Model dimensions
        embed_dim: int = 256,
        # Transformer config (same as tokenizer)
        depth: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        # Action space
        num_discrete_actions: Optional[int] = None,
        continuous_action_dim: Optional[int] = None,
        # Register tokens
        num_registers: int = 4,
        # Shortcut forcing
        max_shortcut_steps: int = 6,  # K in paper (log2 of max steps)
        # Context corruption (Section 4.2)
        context_noise_level: float = 0.1,  # τ_ctx for context frames
    ):
        """
        Args:
            latent_dim: Dimension of latent tokens from tokenizer
            num_latent_tokens: Number of latent tokens per frame
            embed_dim: Transformer embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            ffn_dim: FFN hidden dimension
            dropout: Dropout probability
            num_discrete_actions: Number of discrete actions (if discrete)
            continuous_action_dim: Continuous action dimension (if continuous)
            num_registers: Number of register tokens
            max_shortcut_steps: Maximum log2 of shortcut steps (K)
            context_noise_level: Fixed noise level for context frames (τ_ctx, default 0.1)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.context_noise_level = context_noise_level
        self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.max_shortcut_steps = max_shortcut_steps
        
        # Latent projection (from latent_dim to embed_dim)
        self.latent_in = nn.Linear(latent_dim, embed_dim)
        self.latent_out = nn.Linear(embed_dim, latent_dim)
        
        # Action embedding
        self.action_embed = ActionEmbedding(
            embed_dim=embed_dim,
            num_discrete_actions=num_discrete_actions,
            continuous_action_dim=continuous_action_dim,
        )
        
        # Signal embedding (τ and d)
        self.signal_embed = SignalEmbedding(embed_dim=embed_dim)
        
        # Register tokens
        self.register_tokens = RegisterTokens(
            num_registers=num_registers,
            embed_dim=embed_dim,
        )
        
        # Learnable position embeddings for latent tokens within a timestep
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, num_latent_tokens, embed_dim) * 0.02
        )
        
        # Learnable position embeddings per frame (Section 3.1)
        # These distinguish different timesteps in the sequence
        # We use a large max_seq_len to handle variable-length sequences
        max_seq_len = 1024  # Maximum sequence length for position embeddings
        self.frame_pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )
        
        # Unified block-causal transformer (same architecture as tokenizer)
        self.transformer = BlockCausalTransformer(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_qk_norm=True,
        )
        
        # Output normalization
        self.output_norm = RMSNorm(embed_dim)
        
        # Tokens per timestep: 1 (action) + 1 (signal) + num_latent + num_registers
        self.tokens_per_step = 1 + 1 + num_latent_tokens + num_registers
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.latent_in.weight, std=0.02)
        nn.init.zeros_(self.latent_in.bias)
        nn.init.normal_(self.latent_out.weight, std=0.02)
        nn.init.zeros_(self.latent_out.bias)
    
    def add_noise(
        self,
        latents: torch.Tensor,
        signal_level: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to latents based on signal level τ.

        Interpolation: z̃ = (1 - τ) * noise + τ * z

        Args:
            latents: Clean latents (batch, ..., latent_dim)
            signal_level: τ values in [0, 1] (batch,)

        Returns:
            Noisy latents
        """
        noise = torch.randn_like(latents)

        # Expand signal_level for broadcasting
        while signal_level.dim() < latents.dim():
            signal_level = signal_level.unsqueeze(-1)

        noisy_latents = (1 - signal_level) * noise + signal_level * latents
        return noisy_latents

    def add_noise_with_context_corruption(
        self,
        latents: torch.Tensor,
        signal_level: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise with context corruption per Section 4.2.

        Context frames (all but last) get fixed noise at τ_ctx.
        Target frame (last) gets noise at sampled τ.

        This makes the model robust to its own prediction errors during
        autoregressive generation, since context comes from model predictions.

        Args:
            latents: Clean latents (batch, time, num_latent, latent_dim)
            signal_level: τ values for target frame (batch,)

        Returns:
            Noisy latents with context corruption
        """
        batch_size, time_steps, num_latent, latent_dim = latents.shape
        device = latents.device

        # Generate noise for all frames
        noise = torch.randn_like(latents)

        # Build per-timestep signal levels: τ_ctx for context, τ for target
        # Shape: (batch, time, 1, 1) for broadcasting
        tau_per_step = torch.full(
            (batch_size, time_steps, 1, 1),
            self.context_noise_level,
            device=device,
        )
        # Last timestep uses the sampled signal_level
        tau_per_step[:, -1, :, :] = signal_level.view(batch_size, 1, 1)

        # Apply noise: z̃ = (1 - τ) * noise + τ * z
        noisy_latents = (1 - tau_per_step) * noise + tau_per_step * latents

        return noisy_latents
    
    def sample_shortcut_params(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample shortcut forcing parameters (τ, d) for training.
        
        As per Equation 4:
        - d is sampled as powers of 2: d ~ {2^k : k ∈ {0, 1, ..., K}}
        - τ is sampled uniformly on the corresponding grid
        
        Args:
            batch_size: Number of samples
            device: Device to create tensors on
        
        Returns:
            signal_level: τ values (batch,)
            step_size: d values (batch,)
            d_is_min: Boolean indicating if d is minimum (for loss selection)
        """
        # Sample k uniformly from {0, 1, ..., K}
        k = torch.randint(0, self.max_shortcut_steps + 1, (batch_size,), device=device)

        # Compute step sizes: d = 1 / 2^k
        step_size = 1.0 / (2.0 ** k.float())

        # Sample τ from discrete grid per Equation 4: τ ~ U({0, 1/d, ..., 1 - 1/d})
        # For step_size d, valid τ values are: 0, d, 2d, ..., (1-d)
        # Number of valid grid points = 1/d = 2^k
        num_grid_points = (2.0 ** k.float()).long()  # 2^k points per sample

        # Sample random index from [0, num_grid_points) for each sample
        # Using uniform sampling and floor since each sample may have different grid size
        random_01 = torch.rand(batch_size, device=device)
        grid_indices = torch.floor(random_01 * num_grid_points.float()).long()

        # Compute τ = index * d (ensures τ is on the discrete grid)
        signal_level = grid_indices.float() * step_size
        
        # Check if d is minimum (d_min = 1/2^K)
        d_min = 1.0 / (2.0 ** self.max_shortcut_steps)
        d_is_min = step_size <= d_min + 1e-6
        
        return signal_level, step_size, d_is_min
    
    def prepare_sequence(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        signal_level: torch.Tensor,
        step_size: torch.Tensor,
        discrete_actions: bool = True,
    ) -> torch.Tensor:
        """
        Prepare interleaved input sequence for the transformer.
        
        Sequence structure per timestep:
        [action_embed, signal_embed, latent_1, ..., latent_N, register_1, ..., register_R]
        
        Args:
            latents: Noisy latents (batch, time, num_latent, latent_dim)
            actions: Actions (batch, time) for discrete or (batch, time, action_dim) for continuous
            signal_level: τ values (batch,) or (batch, time)
            step_size: d values (batch,) or (batch, time)
            discrete_actions: Whether actions are discrete
        
        Returns:
            tokens: Interleaved sequence (batch, seq_len, embed_dim)
        """
        batch_size, time_steps, num_latent, _ = latents.shape
        
        # Project latents to embed_dim
        latent_embeds = self.latent_in(latents)  # (batch, time, num_latent, embed_dim)
        latent_embeds = latent_embeds + self.latent_pos_embed
        
        # Embed actions
        if discrete_actions:
            action_embeds = self.action_embed(discrete_actions=actions)  # (batch, time, embed_dim)
        else:
            action_embeds = self.action_embed(continuous_actions=actions)
        
        # Ensure signal params have time dimension
        if signal_level.dim() == 1:
            signal_level = signal_level.unsqueeze(1).expand(-1, time_steps)
        if step_size.dim() == 1:
            step_size = step_size.unsqueeze(1).expand(-1, time_steps)
        
        # Embed signals per timestep
        signal_embeds = []
        for t in range(time_steps):
            sig_embed = self.signal_embed(signal_level[:, t], step_size[:, t])
            signal_embeds.append(sig_embed)
        signal_embeds = torch.stack(signal_embeds, dim=1)  # (batch, time, embed_dim)
        
        # Get register tokens
        registers = self.register_tokens(batch_size)  # (batch, num_reg, embed_dim)
        registers = registers.unsqueeze(1).expand(-1, time_steps, -1, -1)  # (batch, time, num_reg, embed_dim)
        
        # Build interleaved sequence with frame-level position embeddings
        all_tokens = []
        for t in range(time_steps):
            # Get frame position embedding for this timestep
            # Clamp to max_seq_len to handle long sequences
            frame_pos = self.frame_pos_embed[:, t % self.frame_pos_embed.shape[1], :]  # (1, embed_dim)
            
            # Action token (add frame position embedding)
            action_t = action_embeds[:, t:t+1] + frame_pos  # (batch, 1, embed_dim)
            all_tokens.append(action_t)
            
            # Signal token (add frame position embedding)
            signal_t = signal_embeds[:, t:t+1] + frame_pos  # (batch, 1, embed_dim)
            all_tokens.append(signal_t)
            
            # Latent tokens (add frame position embedding to each latent token)
            latent_t = latent_embeds[:, t] + frame_pos  # (batch, num_latent, embed_dim)
            all_tokens.append(latent_t)
            
            # Register tokens (add frame position embedding to each register)
            register_t = registers[:, t] + frame_pos  # (batch, num_reg, embed_dim)
            all_tokens.append(register_t)
        
        tokens = torch.cat(all_tokens, dim=1)  # (batch, time * tokens_per_step, embed_dim)
        
        return tokens
    
    def extract_latent_predictions(
        self,
        output: torch.Tensor,
        time_steps: int,
    ) -> torch.Tensor:
        """
        Extract latent predictions from transformer output.
        
        Args:
            output: Transformer output (batch, seq_len, embed_dim)
            time_steps: Number of timesteps
        
        Returns:
            predicted_latents: (batch, time, num_latent, latent_dim)
        """
        batch_size = output.shape[0]
        
        predicted_latents = []
        for t in range(time_steps):
            # Calculate start index for this timestep's latent tokens
            step_start = t * self.tokens_per_step
            latent_start = step_start + 2  # Skip action and signal tokens
            latent_end = latent_start + self.num_latent_tokens
            
            # Extract and project
            latent_output = output[:, latent_start:latent_end]  # (batch, num_latent, embed_dim)
            latent_output = self.output_norm(latent_output)
            pred_latent = self.latent_out(latent_output)  # (batch, num_latent, latent_dim)
            
            # Apply tanh to match tokenizer bottleneck
            pred_latent = torch.tanh(pred_latent)
            
            predicted_latents.append(pred_latent)
        
        return torch.stack(predicted_latents, dim=1)  # (batch, time, num_latent, latent_dim)
    
    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        signal_level: Optional[torch.Tensor] = None,
        step_size: Optional[torch.Tensor] = None,
        discrete_actions: bool = True,
        add_noise_to_latents: bool = True,
        use_context_corruption: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with shortcut forcing.

        Args:
            latents: Clean latent tokens (batch, time, num_latent, latent_dim)
            actions: Actions (batch, time) or (batch, time, action_dim)
            signal_level: Optional τ values (sampled if not provided)
            step_size: Optional d values (sampled if not provided)
            discrete_actions: Whether actions are discrete
            add_noise_to_latents: Whether to add noise based on signal level
            use_context_corruption: Whether to use context corruption (Section 4.2)
                If True, context frames get τ_ctx noise, target frame gets τ noise.
                If False, all frames get τ noise (original behavior).

        Returns:
            Dictionary containing:
                - predicted_latents: Predicted clean latents
                - target_latents: Target latents (for loss computation)
                - signal_level: τ values used
                - step_size: d values used
                - d_is_min: Whether d is minimum
        """
        batch_size, time_steps = latents.shape[:2]
        device = latents.device

        # Sample shortcut parameters if not provided
        if signal_level is None or step_size is None:
            signal_level, step_size, d_is_min = self.sample_shortcut_params(batch_size, device)
        else:
            d_min = 1.0 / (2.0 ** self.max_shortcut_steps)
            d_is_min = step_size <= d_min + 1e-6

        # Store clean latents as targets
        target_latents = latents.clone()

        # Add noise to latents based on signal level
        if add_noise_to_latents:
            if use_context_corruption and time_steps > 1:
                # Context corruption: τ_ctx for context frames, τ for target frame
                noisy_latents = self.add_noise_with_context_corruption(latents, signal_level)
            else:
                # Original behavior: same τ for all frames
                noisy_latents = self.add_noise(latents, signal_level)
        else:
            noisy_latents = latents
        
        # Prepare interleaved sequence
        tokens = self.prepare_sequence(
            noisy_latents, actions, signal_level, step_size, discrete_actions
        )
        
        # Create block-causal attention mask
        block_size = self.tokens_per_step
        attention_mask = create_block_causal_mask(
            seq_len=tokens.shape[1],
            block_size=block_size,
            device=device,
        )
        
        # Process through transformer
        output = self.transformer(tokens, attention_mask=attention_mask)
        
        # Extract latent predictions
        predicted_latents = self.extract_latent_predictions(output, time_steps)
        
        return {
            "predicted_latents": predicted_latents,
            "target_latents": target_latents,
            "signal_level": signal_level,
            "step_size": step_size,
            "d_is_min": d_is_min,
        }
    
    @torch.no_grad()
    def generate(
        self,
        initial_latents: torch.Tensor,
        actions: torch.Tensor,
        num_steps: int = 4,
        discrete_actions: bool = True,
    ) -> torch.Tensor:
        """
        Generate future latent states autoregressively.
        
        Uses shortcut forcing with the specified number of denoising steps.
        
        Args:
            initial_latents: Starting latents (batch, num_latent, latent_dim)
            actions: Future actions (batch, horizon) or (batch, horizon, action_dim)
            num_steps: Number of denoising steps (K)
            discrete_actions: Whether actions are discrete
        
        Returns:
            generated_latents: (batch, horizon, num_latent, latent_dim)
        """
        batch_size = initial_latents.shape[0]
        horizon = actions.shape[1]
        device = initial_latents.device
        
        # Step size for sampling
        step_size_val = 1.0 / num_steps
        
        generated = []
        current_latent = initial_latents
        
        for t in range(horizon):
            # Start from noise
            z = torch.randn_like(current_latent)
            
            # Get action for this step
            if discrete_actions:
                action_t = actions[:, t:t+1]
            else:
                action_t = actions[:, t:t+1]
            
            # Iterative denoising
            for step in range(num_steps):
                tau = step * step_size_val
                tau_tensor = torch.full((batch_size,), tau, device=device)
                d_tensor = torch.full((batch_size,), step_size_val, device=device)
                
                # Prepare input
                z_input = z.unsqueeze(1)  # (batch, 1, num_latent, latent_dim)
                
                tokens = self.prepare_sequence(
                    z_input, action_t, tau_tensor, d_tensor, discrete_actions
                )
                
                # Get prediction
                output = self.transformer(tokens)
                pred = self.extract_latent_predictions(output, 1)[:, 0]
                
                # Update z
                z = z + step_size_val * (pred - z)
            
            generated.append(z)
            current_latent = z
        
        return torch.stack(generated, dim=1)
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
