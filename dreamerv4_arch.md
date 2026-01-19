DreamerV4 Data Pipeline
INPUT: Raw MineRL Data
From extracted files (data/mineRL_extracted/):
frames.npy: (T, 64, 64, 3) - RGB frames, uint8 [0-255]
actions.npz: Dictionary with action arrays
action$forward, action$left, etc.: (T,) binary int64
action$camera: (T, 2) float32 - pitch, yaw
rewards.npy: (T,) float32 - scalar rewards
observations.npz: Additional observations (inventory, compass, etc.)
STAGE 1: DataLoader (MineRLDataset)
Input:
Episode directories with frames.npy, actions.npz, rewards.npy
Processing:
Loads sequences of length 16 (configurable)
Samples overlapping sequences from episodes
Normalizes frames: frames / 255.0 → [0, 1] range
Combines actions into single discrete action tensor
Output (per batch):
{    "frames": (B, T, C, H, W)  # (batch, 16, 3, 64, 64) - float32 [0, 1]    "actions": (B, T)          # (batch, 16) - int64 discrete actions    "rewards": (B, T)           # (batch, 16) - float32}
PHASE 1: World Model Pretraining
Step 1A: Tokenizer Training
Input:
frames: (B, T, C, H, W) = (batch, 16, 3, 64, 64)
Processing:
Patchify: 64×64 image → 8×8 patches = 64 patches per frame
Mask 75% of patches randomly
Encode: Transformer → latent tokens
Decode: Reconstruct masked patches
Output:
{    "latents": (B, T, num_latent, latent_dim)  # (batch, 16, 16, 32)    "reconstructed": (B, T, num_patches, patch_dim)  # Reconstructed patches    "mask": (B, T, num_patches)  # Which patches were masked}
Loss: MSE + 0.2 × LPIPS on reconstructed patches
Step 1B: Dynamics Model Training
Input:
latents: (B, T, 16, 32) - from tokenizer (frozen)
actions: (B, T) - discrete actions
signal_level τ: (B,) - noise level [0, 1]
step_size d: (B,) - shortcut step size
Processing:
Add noise to latents: z̃ = (1-τ)z₀ + τz₁
Create interleaved sequence: [action, τ, d, latent_1, ..., latent_16]
Transformer predicts clean latent ẑ₁ directly (x-prediction)
Output:
{    "predicted_latents": (B, T, 16, 32)  # Predicted clean latents    "noisy_latents": (B, T, 16, 32)     # Input noisy latents}
Loss: Shortcut forcing loss (weighted by signal level)
PHASE 2: Agent Finetuning
Input:
latents: (B, T, 16, 32) - from frozen tokenizer
actions: (B, T) - ground truth actions
rewards: (B, T) - ground truth rewards
Processing:
Policy head: predicts action distribution from latents
Reward head: predicts reward distribution from latents
Value head: predicts value distribution from latents
Output:
{    "policy_logits": (B, T, num_actions)  # (batch, 16, 144)    "reward_logits": (B, T, num_bins)    # (batch, 16, 255)    "value_logits": (B, T, num_bins)     # (batch, 16, 255)}
Loss: Behavior cloning (action prediction) + reward prediction
PHASE 3: Imagination Training (RL)
Input:
Initial latents: (B, 16, 32) - from real data
Policy head (trained in Phase 2)
Value head (trained in Phase 2)
Dynamics model (frozen from Phase 1)
Processing:
Imagination rollout:
Sample action from policy: a_t ~ π(·|s_t)
Predict next latent: s_{t+1} = dynamics(s_t, a_t)
Predict reward: r_t = reward_head(s_t)
Rollout for H=15 steps
Compute advantages: TD(λ) returns - values
PMPO loss: Bin trajectories by advantage, update policy
Output:
}
{    "imagined_latents": (B, H, 16, 32)      # Rolled out states    "imagined_actions": (B, H)              # Sampled actions    "imagined_rewards": (B, H)              # Predicted rewards    "advantages": (B, H)                    # Advantage estimates    "returns": (B, H)                       # TD(λ) returns}
Losses:
Policy: PMPO (weighted by advantage bins)
Value: TD(λ) distributional loss
Summary Flow
Raw Data (frames.npy, actions.npz, rewards.npy)    ↓DataLoader: Normalize & batch sequences    ↓[B, T, C, H, W] frames → Tokenizer → [B, T, 16, 32] latents    ↓[B, T, 16, 32] latents + [B, T] actions → Dynamics → [B, T, 16, 32] predicted latents    ↓[B, T, 16, 32] latents → Policy/Value/Reward Heads → Action/Value/Reward predictions    ↓Imagination: Rollout in latent space → PMPO + TD(λ) training
Key Points:
All processing happens in latent space (32-dim vectors) after tokenization
Actions are discretized to 144 classes (8 binary + 121 camera + 15 additional)
Rewards/values use distributional learning (255 bins)
Imagination happens entirely in latent space, no real environment needed
The pipeline converts raw video → compressed latents → learned world model → RL policy, all trained offline from the MineRL dataset.


Transformer Model Dimensions
Yes, they are the same
The tokenizer and dynamics model (world model) use the same transformer architecture and dimensions. This is a design choice in DreamerV4.
Current Configuration (from configs/minerl.yaml):
Tokenizer & Dynamics Model (Shared):├── embed_dim: 256          # Transformer embedding dimension├── depth: 6                 # Number of transformer layers├── num_heads: 8             # Attention heads├── head_dim: 32             # 256 / 8 = 32 per head├── ffn_dim: ~704            # Calculated as: int(256 * 4 * 2 / 3) = 682 → rounded to 704 (multiple of 64)├── num_registers: 4          # Register tokens for temporal consistency└── dropout: 0.0             # No dropout
Detailed Breakdown:
Attention:
embed_dim = 256
num_heads = 8
head_dim = 256 / 8 = 32 per head
QKV projection: 256 → 3 × 8 × 32 = 768 parameters
Feed-Forward Network (SwiGLU):
ffn_dim = int(256 × 4 × 2 / 3) = 682
Rounded to nearest multiple of 64: 704
Structure:
W1 (gate): 256 → 704
W3 (up): 256 → 704
W2 (out): 704 → 256
Total FFN params per layer: 256×704 + 256×704 + 704×256 = 1,081,344
Per Transformer Layer:
Attention: ~197K parameters
FFN: ~1.08M parameters
Norms + residuals: ~1K parameters
Total per layer: ~1.28M parameters
Full Transformer (6 layers):
Total: ~7.7M parameters
Same for both tokenizer and dynamics model
Differences (not in transformer dimensions):
Tokenizer:
Input: Image patches (64 patches × patch_dim)
Output: Latent tokens (16 tokens × 32 dim)
Has encoder + decoder paths
Dynamics Model:
Input: Latent tokens (16 tokens × 32 dim) + actions + signals
Output: Predicted latent tokens (16 tokens × 32 dim)
Processes interleaved sequences: [action, τ, d, latent_1, ..., latent_16]
Why Same Dimensions?
Unified architecture: same transformer for both
Shared weights: can share some components (not implemented here, but possible)
Consistent representation: same latent space dimensions
Efficiency: reuse the same transformer implementation
The transformer backbone is identical; only the input/output processing differs.

