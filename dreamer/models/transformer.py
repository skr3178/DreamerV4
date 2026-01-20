"""
Unified Block-Causal Transformer for DreamerV4

This is the core architecture shared by both the Tokenizer and Dynamics Model.
Based on the paper: "Training Agents Inside of Scalable World Models"

Key components:
- Pre-layer RMSNorm
- Block-causal attention (attend to same + past timesteps)  
- RoPE (Rotary Position Embedding)
- SwiGLU activation
- QK-Norm for stability
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    More stable than LayerNorm, commonly used in modern transformers.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return x_normed * self.weight


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Encodes position through rotation of feature dimensions.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for max sequence length
        self._update_cos_sin_cache(max_seq_len)
    
    def _update_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin values for the given sequence length."""
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (not used, just for device)
            seq_len: Sequence length to get embeddings for
        Returns:
            cos, sin: Position embeddings of shape (seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to queries and keys.
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        cos: Cosine embeddings of shape (seq_len, head_dim)
        sin: Sine embeddings of shape (seq_len, head_dim)
    """
    # Reshape cos and sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    Combines Swish activation with Gated Linear Unit.
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = False):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4 * 2 / 3)
        # Make hidden_dim a multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)  # Output
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # Up projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(W1 * x) * (W3 * x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class QKNorm(nn.Module):
    """
    Query-Key Normalization for attention stability.
    Normalizes Q and K before computing attention scores.
    """
    
    def __init__(self, head_dim: int):
        super().__init__()
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


def create_block_causal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a block-causal attention mask.

    In block-causal attention:
    - Tokens within the same block can attend to each other (bidirectional)
    - Tokens can attend to all tokens in previous blocks
    - Tokens cannot attend to future blocks

    Args:
        seq_len: Total sequence length
        block_size: Size of each block (e.g., spatial tokens per timestep)
        device: Device to create mask on

    Returns:
        Attention mask of shape (seq_len, seq_len)
        True = attend, False = mask out
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    num_blocks = (seq_len + block_size - 1) // block_size

    for i in range(num_blocks):
        block_start = i * block_size
        block_end = min((i + 1) * block_size, seq_len)

        # Attend to all previous blocks
        mask[block_start:block_end, :block_end] = True

    return mask


def create_spatial_mask(
    seq_len: int,
    block_size: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a spatial attention mask (Section 3.2).

    Spatial attention: tokens only attend within their own block (timestep).
    This is bidirectional within the block, no cross-timestep attention.

    Args:
        seq_len: Total sequence length
        block_size: Size of each block (tokens per timestep)
        device: Device to create mask on

    Returns:
        Attention mask of shape (seq_len, seq_len)
        True = attend, False = mask out
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    num_blocks = (seq_len + block_size - 1) // block_size

    for i in range(num_blocks):
        block_start = i * block_size
        block_end = min((i + 1) * block_size, seq_len)

        # Only attend within the same block (bidirectional)
        mask[block_start:block_end, block_start:block_end] = True

    return mask


def create_temporal_mask(
    seq_len: int,
    block_size: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a temporal attention mask (Section 3.2).

    Temporal attention: tokens attend to the same position across timesteps.
    This is causal across time (can only attend to past timesteps).

    For example, with block_size=4:
    - Token 0 attends to tokens 0, 4, 8, ... (same position in each block)
    - Token 1 attends to tokens 1, 5, 9, ...
    - But only causally (past and current blocks)

    Args:
        seq_len: Total sequence length
        block_size: Size of each block (tokens per timestep)
        device: Device to create mask on

    Returns:
        Attention mask of shape (seq_len, seq_len)
        True = attend, False = mask out
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    num_blocks = (seq_len + block_size - 1) // block_size

    for i in range(seq_len):
        # Which block is this token in?
        current_block = i // block_size
        # Position within the block
        pos_in_block = i % block_size

        # Attend to same position in current and all past blocks
        for b in range(current_block + 1):
            target_idx = b * block_size + pos_in_block
            if target_idx < seq_len:
                mask[i, target_idx] = True

    return mask


class BlockCausalAttention(nn.Module):
    """
    Block-Causal Multi-Head Attention with optional Grouped Query Attention (GQA).

    Implements the attention pattern from DreamerV4:
    - Tokens within the same timestep can attend to each other
    - Tokens can attend to all past timesteps
    - Cannot attend to future timesteps

    GQA (Section 3.2): Uses fewer KV heads than query heads to reduce KV cache size.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_flash_attention: bool = True,
        allow_flash_only_for_standard_causal: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # GQA: num_kv_heads can be less than num_heads
        # Default to num_heads (standard MHA) if not specified
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm
        self.use_flash_attention = use_flash_attention
        self.allow_flash_only_for_standard_causal = allow_flash_only_for_standard_causal

        # Validate GQA config
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        self.num_queries_per_kv = num_heads // self.num_kv_heads

        # Separate Q and KV projections for GQA
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * self.num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        # QK normalization
        if use_qk_norm:
            self.qk_norm = QKNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            rope_cos: Rotary position embedding cosines
            rope_sin: Rotary position embedding sines
            attention_mask: Boolean mask (True = attend, False = mask)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q (all heads)
        q = self.q_proj(x)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads, d=self.head_dim)

        # Compute K, V (fewer heads for GQA)
        kv = self.kv_proj(x)
        kv = rearrange(
            kv,
            "b s (two h d) -> two b h s d",
            two=2,
            h=self.num_kv_heads,
            d=self.head_dim
        )
        k, v = kv[0], kv[1]

        # Apply QK normalization (before repeating KV for GQA)
        if self.use_qk_norm:
            q, k = self.qk_norm(q, k)

        # Apply rotary position embeddings (before repeating KV for GQA)
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # GQA: Repeat K and V to match number of query heads
        if self.num_kv_heads < self.num_heads:
            # Repeat each KV head to serve multiple query heads
            k = repeat(k, "b h s d -> b (h r) s d", r=self.num_queries_per_kv)
            v = repeat(v, "b h s d -> b (h r) s d", r=self.num_queries_per_kv)
        
        # ────────────────────────────────────────────────────────────────
        # Attention computation
        # ────────────────────────────────────────────────────────────────
        # Flash attention only supports standard causal (is_causal=True) 
        # or full attention — never custom block-causal patterns.
        # Therefore we only allow flash when NO custom mask is provided.
        if (
            self.use_flash_attention 
            and attention_mask is None 
            and self.allow_flash_only_for_standard_causal
        ):
            # Standard causal attention via flash (past + current allowed)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention — required for block-causal or any custom mask
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                # Use provided block-causal (or other custom) mask
                attn_mask = torch.zeros_like(attn_weights)
                # attention_mask shape: (seq_len, seq_len) -> broadcast to (B, H, S, S)
                attn_mask.masked_fill_(~attention_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                attn_weights = attn_weights + attn_mask
            else:
                # No mask provided → apply **standard causal** mask manually
                # (this is fallback when flash was requested but we cannot use it)
                if self.use_flash_attention:
                    warnings.warn(
                        "Flash attention requested but custom/block-causal mask required → "
                        "falling back to manual attention (slower).",
                        RuntimeWarning
                    )
                causal_mask = torch.tril(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
                )  # lower-triangular: True = allowed
                attn_mask = torch.zeros_like(attn_weights)
                attn_mask.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                attn_weights = attn_weights + attn_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.out_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """
    Single Transformer Block with Pre-Norm architecture.

    Structure:
    1. RMSNorm -> Block-Causal Attention -> Residual
    2. RMSNorm -> SwiGLU FFN -> Residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        # Pre-norm for attention
        self.norm1 = RMSNorm(dim)

        # Block-causal attention with optional GQA
        self.attention = BlockCausalAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
        )
        
        # Pre-norm for FFN
        self.norm2 = RMSNorm(dim)
        
        # SwiGLU FFN
        self.ffn = SwiGLU(dim, ffn_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            rope_cos: Rotary position embedding cosines
            rope_sin: Rotary position embedding sines
            attention_mask: Boolean attention mask
        
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Attention with residual
        x = x + self.dropout(
            self.attention(
                self.norm1(x),
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                attention_mask=attention_mask,
            )
        )
        
        # FFN with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x


class BlockCausalTransformer(nn.Module):
    """
    Block-Causal Transformer - Core unified architecture for DreamerV4.

    This is the shared architecture used by both:
    - Causal Tokenizer (for encoding/decoding frames)
    - Dynamics Model (for predicting future states)

    Key features:
    - Pre-layer RMSNorm
    - Block-causal attention pattern
    - RoPE position embeddings
    - SwiGLU activation
    - QK-Norm for stability
    - Space/time factorized attention (Section 3.2)
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_spacetime_factorization: bool = False,
        temporal_layer_interval: int = 4,
    ):
        """
        Args:
            dim: Model dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads for GQA (default: same as num_heads)
            head_dim: Dimension per head (default: dim // num_heads)
            ffn_dim: FFN hidden dimension (default: ~2.67 * dim for SwiGLU)
            max_seq_len: Maximum sequence length for position embeddings
            dropout: Dropout probability
            use_qk_norm: Whether to use QK normalization
            use_spacetime_factorization: Whether to use space/time factorized attention (Section 3.2)
            temporal_layer_interval: Apply temporal attention every N layers (default: 4)
        """
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or dim // num_heads
        self.use_spacetime_factorization = use_spacetime_factorization
        self.temporal_layer_interval = temporal_layer_interval

        # Rotary position embeddings
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
        )

        # Transformer layers with optional GQA
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(depth)
        ])

        # Final normalization
        self.final_norm = RMSNorm(dim)

    def _is_temporal_layer(self, layer_idx: int) -> bool:
        """Check if this layer should use temporal attention."""
        # Temporal attention every N layers (0-indexed: 3, 7, 11, ... for interval=4)
        return (layer_idx + 1) % self.temporal_layer_interval == 0

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Optional custom attention mask (overrides factorization)
            block_size: Block size for creating masks (tokens per timestep)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        _, seq_len, _ = x.shape

        # Get rotary embeddings
        rope_cos, rope_sin = self.rope(x, seq_len)

        # Create masks for space/time factorized attention if enabled
        if self.use_spacetime_factorization and block_size is not None and attention_mask is None:
            spatial_mask = create_spatial_mask(
                seq_len=seq_len,
                block_size=block_size,
                device=x.device,
            )
            temporal_mask = create_temporal_mask(
                seq_len=seq_len,
                block_size=block_size,
                device=x.device,
            )

            # Forward through transformer layers with factorized attention
            for layer_idx, layer in enumerate(self.layers):
                if self._is_temporal_layer(layer_idx):
                    layer_mask = temporal_mask
                else:
                    layer_mask = spatial_mask

                x = layer(
                    x,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    attention_mask=layer_mask,
                )
        else:
            # Original behavior: same mask for all layers
            if block_size is not None and attention_mask is None:
                attention_mask = create_block_causal_mask(
                    seq_len=seq_len,
                    block_size=block_size,
                    device=x.device,
                )

            # Forward through transformer layers
            for layer in self.layers:
                x = layer(
                    x,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    attention_mask=attention_mask,
                )

        # Final normalization
        x = self.final_norm(x)

        return x

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
