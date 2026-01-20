# DreamerV4 Implementation Review - Deep Dive Analysis

## Overview

This document provides a detailed technical analysis of the DreamerV4 implementation compared to the paper "Training Agents Inside of Scalable World Models". The implementation is adapted for the MineRL dataset with reduced model capacity.

---

## Issue 1: Tokenizer Temporal Encoding

### Problem Description

The tokenizer processes frames **independently** in a loop rather than using block-causal attention across the temporal dimension.

### Code Evidence

**Current Implementation** (`tokenizer.py:271-278`):
```python
def encode(self, video, mask_ratio=None):
    # ...
    for t in range(time_steps):
        frame = video[:, t]  # Process each frame independently
        result = self.encode_frame(frame, mask_ratio=mask_ratio)
        all_latents.append(result["latents"])
```

**Paper's Specification**:
> "The tokenizer uses block-causal attention where spatial tokens at time t can attend to all tokens at times ≤ t"

### Technical Analysis

**What's Missing**:
1. The `encode_frame()` method processes a single frame through the transformer
2. No temporal context is passed between frames
3. Each frame's latent representation is computed in isolation

**What the Paper Describes**:
```
Input sequence for T frames:
[patch_1^t1, ..., patch_64^t1, latent_1^t1, ..., latent_16^t1, reg_1^t1, ..., reg_4^t1,
 patch_1^t2, ..., patch_64^t2, latent_1^t2, ..., latent_16^t2, reg_1^t2, ..., reg_4^t2,
 ...]

With block-causal mask:
- Tokens at t2 can attend to all tokens at t1 and t2
- Tokens at t3 can attend to all tokens at t1, t2, and t3
```

### Impact Assessment

| Metric | Per-Frame | Block-Causal | Impact |
|--------|-----------|--------------|--------|
| Temporal consistency | Low | High | Objects may "flicker" between frames |
| Compression quality | Moderate | Higher | Less redundancy exploitation |
| Decoder coherence | Low | High | Reconstructed videos less smooth |
| Dynamics learning | Harder | Easier | Dynamics must learn all temporal patterns |

### Severity: **MEDIUM**

For MineRL with 64x64 frames and short sequences (T=16), this may be acceptable because:
- The dynamics model can compensate by learning temporal patterns
- Small images have less temporal redundancy to exploit
- Computational savings are significant

### Recommendation

If temporal artifacts appear in reconstructions or generations:

```python
def encode_temporal(self, video, mask_ratio=None):
    """Encode video with block-causal attention across time."""
    B, T, C, H, W = video.shape

    # Patchify all frames
    all_patches = []
    for t in range(T):
        patches = self.patch_embed(video[:, t])  # (B, 64, embed_dim)
        all_patches.append(patches)

    # Get latent and register tokens for all timesteps
    latent_tokens = self.latent_tokens(B).unsqueeze(1).expand(-1, T, -1, -1)
    register_tokens = self.register_tokens(B).unsqueeze(1).expand(-1, T, -1, -1)

    # Build full sequence: (B, T * tokens_per_frame, embed_dim)
    tokens_per_frame = self.num_patches + self.num_latent_tokens + self.num_registers
    full_sequence = []
    for t in range(T):
        full_sequence.extend([
            all_patches[t],
            latent_tokens[:, t],
            register_tokens[:, t]
        ])
    full_sequence = torch.cat(full_sequence, dim=1)

    # Create block-causal mask
    mask = create_block_causal_mask(
        seq_len=T * tokens_per_frame,
        block_size=tokens_per_frame,
        device=video.device
    )

    # Process through transformer with temporal attention
    output = self.transformer(full_sequence, attention_mask=mask)

    # Extract latents for each timestep
    latents = []
    for t in range(T):
        start = t * tokens_per_frame + self.num_patches
        end = start + self.num_latent_tokens
        latent_output = output[:, start:end]
        latents.append(self.latent_tokens.to_bottleneck(latent_output))

    return torch.stack(latents, dim=1)  # (B, T, num_latent, latent_dim)
```

---

## Issue 2: Shortcut Forcing Tau Scheduling

### Problem Description

The generation loop has an **off-by-one error** in tau scheduling, causing the model to never reach τ=1 (fully clean) during generation.

### Code Evidence

**Training** (`dynamics.py:188-192`):
```python
# Sample τ uniformly in [0, 1]
signal_level = torch.rand(batch_size, device=device)

# Ensure τ + d <= 1
signal_level = signal_level * (1.0 - step_size)
```

This means during training, τ can be in `[0, 1-d]` for a given step size d.

**Generation** (`rollout.py:116-133` and `dynamics.py:422-439`):
```python
for step in range(self.num_denoising_steps):  # K=4 steps
    tau = step * step_size_val  # τ = 0, 0.25, 0.5, 0.75
    # ...
    z = z + step_size_val * (pred - z)
```

### Technical Analysis

**The Issue**:
```
K=4 denoising steps, step_size = 0.25

Step 0: tau = 0.00, model sees noise
Step 1: tau = 0.25, model sees 75% noise + 25% signal
Step 2: tau = 0.50, model sees 50% noise + 50% signal
Step 3: tau = 0.75, model sees 25% noise + 75% signal

Final output: z after step 3 (never evaluated at tau=1.0)
```

**Paper's Intent** (Shortcut Forcing):
The model should predict the **clean latent z₁** at each step. The final output should be the prediction at or near τ=1.

**Why This Matters**:
1. The model was trained to predict clean latents from various noise levels
2. At τ=0.75, there's still 25% noise mixed in
3. The final `z` is the result of Euler steps, not a direct prediction at τ=1

### Correct Implementation

Looking at the paper's Equation 6 (shortcut sampling):
```
z_{τ+d} = z_τ + d · f_θ(z_τ, τ, d)

Where f_θ predicts the direction from z_τ toward z_1
```

The update rule `z = z + d * (pred - z)` is correct **if** `pred` is the model's estimate of z₁.

**The Fix**: The issue is not the update rule but the tau values used. Two options:

**Option A: Shift tau by 1 step**
```python
for step in range(self.num_denoising_steps):
    tau = (step + 1) * step_size_val  # τ = 0.25, 0.5, 0.75, 1.0
```

**Option B: Use tau as the "current" noise level (more consistent with training)**
```python
for step in range(self.num_denoising_steps):
    tau = step * step_size_val  # τ = 0, 0.25, 0.5, 0.75 (current noise level)
    tau_target = tau + step_size_val  # Target: 0.25, 0.5, 0.75, 1.0

    # Model predicts z_1 given z_tau
    # Update moves z toward z_1 by step_size
```

The current implementation actually uses Option B's logic but doesn't clearly document it. The `pred` IS the estimate of z₁, and the update moves z toward it.

### Verification Test

```python
def verify_generation_quality():
    """Test that generation produces clean outputs."""
    # Generate from random noise
    z = torch.randn(1, 16, 32)  # (batch, num_latent, latent_dim)

    # After K denoising steps
    generated = dynamics.generate(initial_latent, action, num_steps=4)

    # Check: generated should have similar statistics to encoded real frames
    real_latents = tokenizer.encode(real_frames)

    print(f"Real latents - mean: {real_latents.mean():.3f}, std: {real_latents.std():.3f}")
    print(f"Generated - mean: {generated.mean():.3f}, std: {generated.std():.3f}")

    # Both should be similar (within tanh bounds [-1, 1])
```

### Severity: **MEDIUM**

The current implementation may work because:
1. The Euler update converges toward the prediction
2. 4 steps of refinement reduces noise significantly
3. The tanh activation bounds the output

However, it's not optimal and may cause slightly blurry/noisy generations.

---

## Issue 3: PMPO with Sparse MineRL Rewards

### Problem Description

MineRL has extremely sparse rewards. PMPO relies on advantage signs to determine D+ and D- sets, which may collapse when all advantages are near zero.

### Code Evidence

**Advantage Binning** (`pmpo_loss.py:171-177`):
```python
positive_mask = flat_advantages > 0   # D+: good actions
negative_mask = flat_advantages < 0   # D-: bad actions

n_positive = positive_mask.sum().float().clamp(min=1.0)
n_negative = negative_mask.sum().float().clamp(min=1.0)
```

**Advantage Computation** (`rollout.py:257-265`):
```python
for t in reversed(range(horizon)):
    delta = rewards[:, t] + self.discount * mask * next_value - values[:, t]
    advantage = delta + self.discount * lambda_ * mask * next_advantage
    advantages[:, t] = advantage
```

### Technical Analysis

**Scenario: Sparse Rewards in MineRL**

Consider a trajectory of 15 steps with rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:

```python
# TD(λ) computation with γ=0.997, λ=0.95
# All rewards are 0, so returns depend entirely on value bootstrap

# If value_head predicts V(s) ≈ 0 for all states:
advantages ≈ [0, 0, 0, 0, ...]  # All near zero!

# Result:
n_positive ≈ 0  # No good actions
n_negative ≈ 0  # No bad actions
```

**PMPO Loss Degeneracy**:
```python
# When advantages are all ~0:
positive_mask.sum() → 0 (clamped to 1)
negative_mask.sum() → 0 (clamped to 1)

# Loss becomes:
pos_term = -0.5 * log_probs[empty_mask].sum()  # ≈ 0
neg_term = 0.5 * log_probs[empty_mask].sum()   # ≈ 0

# Total policy_loss ≈ 0 → No learning signal!
```

### Empirical Detection

Add this monitoring code to `train_phase3.py`:

```python
# After computing rollout_data
advantages = rollout_data["advantages"]
rewards = rollout_data["rewards"]

# Monitor reward statistics
print(f"Rewards - min: {rewards.min():.4f}, max: {rewards.max():.4f}, "
      f"mean: {rewards.mean():.4f}, nonzero: {(rewards != 0).sum()}")

# Monitor advantage distribution
print(f"Advantages - min: {advantages.min():.4f}, max: {advantages.max():.4f}, "
      f"std: {advantages.std():.4f}")

# Critical check
n_pos = (advantages > 0).sum()
n_neg = (advantages < 0).sum()
n_zero = (advantages == 0).sum()
print(f"Advantage split: D+={n_pos}, D-={n_neg}, zero={n_zero}")

# WARNING if degenerate
if n_pos < 10 and n_neg < 10:
    print("WARNING: Degenerate advantage distribution - no learning signal!")
```

### Severity: **HIGH**

This is a fundamental issue for MineRL because:
1. Diamond-obtaining rewards are extremely rare
2. Most intermediate states have zero reward
3. Without reward signal, PMPO cannot distinguish good from bad actions

### Solutions

**Solution A: Intrinsic Motivation / Curiosity**

Add exploration bonuses based on prediction error:

```python
class IntrinsicReward(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.predictor = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, current_latent, next_latent, action_embed):
        # Predict next latent from current + action
        pred_next = self.predictor(torch.cat([current_latent, action_embed], dim=-1))

        # Intrinsic reward = prediction error (novelty)
        intrinsic = F.mse_loss(pred_next, next_latent.detach(), reduction='none')
        return intrinsic.mean(dim=-1)  # (batch,)

# In rollout:
intrinsic_reward = intrinsic_model(latent_t, latent_t1, action_embed)
total_reward = extrinsic_reward + 0.01 * intrinsic_reward
```

**Solution B: Reward Shaping from Minecraft Heuristics**

```python
def shaped_reward(obs_dict):
    """Add intermediate rewards for Minecraft progress."""
    reward = 0.0

    # Inventory-based rewards
    inventory = obs_dict.get('inventory', {})
    reward += 0.001 * inventory.get('log', 0)      # Collecting wood
    reward += 0.005 * inventory.get('planks', 0)   # Crafting planks
    reward += 0.01 * inventory.get('stick', 0)     # Making sticks
    reward += 0.05 * inventory.get('wooden_pickaxe', 0)
    reward += 0.1 * inventory.get('stone_pickaxe', 0)
    reward += 0.5 * inventory.get('iron_pickaxe', 0)
    reward += 1.0 * inventory.get('diamond', 0)

    return reward
```

**Solution C: Advantage Clipping and Scaling**

```python
def robust_advantage_binning(advantages, percentile=10):
    """More robust binning that handles sparse rewards."""
    # Use percentiles instead of zero threshold
    threshold_pos = torch.quantile(advantages, 1 - percentile/100)
    threshold_neg = torch.quantile(advantages, percentile/100)

    positive_mask = advantages > threshold_pos  # Top 10%
    negative_mask = advantages < threshold_neg  # Bottom 10%

    return positive_mask, negative_mask
```

---

## Issue 4: Reward Prediction Collapse

### Problem Description

The reward head, trained on sparse rewards in Phase 2, may learn to always predict zero, making imagination-based RL impossible.

### Code Evidence

**Phase 2 Training** (`train_phase2.py:133`):
```python
reward_output = heads["reward"](latents_flat)
```

**Reward Loss** (`agent_loss.py:119-124`):
```python
# Convert targets to bin distributions
target_bins = reward_head.target_to_bins(target_rewards)

# Cross-entropy loss
log_probs = F.log_softmax(logits, dim=-1)
loss = -(target_bins * log_probs).sum(dim=-1).mean()
```

**Dataset Default** (`minerl_dataset.py:247`):
```python
return torch.tensor(0.0)  # Default reward when not found
```

### Technical Analysis

**The Problem**:
With MineRL rewards, the training distribution might be:
```
reward = 0: 99.9% of samples
reward > 0: 0.1% of samples
```

The optimal solution for cross-entropy loss becomes:
```
Predict reward = 0 for everything → 99.9% accuracy!
```

**Checking for Collapse**:

```python
def diagnose_reward_head(reward_head, dataloader, device):
    """Check if reward head has collapsed."""
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            latents = encode_batch(batch)  # Get latents
            pred = reward_head(latents)["reward"]

            all_predictions.append(pred)
            all_targets.append(batch["rewards"])

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    print(f"Predictions - mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
    print(f"Targets - mean: {targets.mean():.4f}, std: {targets.std():.4f}")

    # Check for collapse
    if predictions.std() < 0.01:
        print("WARNING: Reward head has collapsed to constant prediction!")

    # Check correlation with actual non-zero rewards
    nonzero_mask = targets != 0
    if nonzero_mask.sum() > 0:
        corr = torch.corrcoef(torch.stack([
            predictions[nonzero_mask],
            targets[nonzero_mask]
        ]))[0, 1]
        print(f"Correlation on non-zero rewards: {corr:.4f}")
```

### Severity: **HIGH**

If the reward head collapses:
1. All imagined rewards ≈ 0
2. All TD returns ≈ bootstrap value
3. All advantages ≈ 0
4. PMPO has no learning signal
5. Policy never improves

### Solutions

**Solution A: Focal Loss for Imbalanced Rewards**

```python
class FocalRewardLoss(nn.Module):
    """Focal loss to handle class imbalance in rewards."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, reward_head):
        target_bins = reward_head.target_to_bins(targets)
        probs = F.softmax(logits, dim=-1)

        # Focal weight: down-weight easy (common) examples
        focal_weight = (1 - probs) ** self.gamma

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(self.alpha * focal_weight * target_bins * log_probs).sum(dim=-1)

        return loss.mean()
```

**Solution B: Oversampling Non-Zero Rewards**

```python
class BalancedMineRLDataset(MineRLDataset):
    def __init__(self, *args, oversample_factor=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.oversample_factor = oversample_factor
        self._build_reward_index()

    def _build_reward_index(self):
        """Find samples with non-zero rewards."""
        self.rewarded_samples = []
        for idx in range(len(self.samples)):
            ep_idx, start = self.samples[idx]
            episode = self.episodes[ep_idx]
            # Check if any reward in sequence is non-zero
            if self._has_nonzero_reward(episode, start):
                self.rewarded_samples.append(idx)

        print(f"Found {len(self.rewarded_samples)} samples with rewards")

    def __getitem__(self, idx):
        # With probability p, sample from rewarded examples
        if random.random() < 0.5 and self.rewarded_samples:
            idx = random.choice(self.rewarded_samples)
        return super().__getitem__(idx)
```

**Solution C: Auxiliary Reward Prediction Task**

Train the reward head on multiple objectives:

```python
class MultiTaskRewardHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_bins):
        super().__init__()
        self.mlp = nn.Sequential(...)

        # Multiple prediction heads
        self.reward_head = nn.Linear(hidden_dim, num_bins)  # Distributional
        self.binary_head = nn.Linear(hidden_dim, 1)  # Is reward > 0?
        self.magnitude_head = nn.Linear(hidden_dim, 1)  # Reward magnitude (regression)

    def forward(self, latents):
        hidden = self.mlp(latents)
        return {
            "logits": self.reward_head(hidden),
            "is_positive": self.binary_head(hidden),
            "magnitude": self.magnitude_head(hidden),
        }

# Training loss
binary_loss = F.binary_cross_entropy_with_logits(
    pred["is_positive"],
    (targets > 0).float()
)
magnitude_loss = F.mse_loss(
    pred["magnitude"][targets > 0],
    targets[targets > 0]
)
distributional_loss = cross_entropy_loss(pred["logits"], target_bins)

total_loss = distributional_loss + 0.1 * binary_loss + 0.1 * magnitude_loss
```

---

## Summary and Priority

| Issue | Severity | Effort to Fix | Priority |
|-------|----------|---------------|----------|
| Reward Prediction Collapse | HIGH | Medium | **1st** |
| PMPO Sparse Rewards | HIGH | Medium | **2nd** |
| Shortcut Tau Scheduling | MEDIUM | Low | **3rd** |
| Tokenizer Temporal | MEDIUM | High | **4th** |

### Recommended Action Plan

1. **Immediate**: Add monitoring for reward/advantage statistics
2. **Phase 2**: Implement balanced sampling or focal loss for rewards
3. **Phase 3**: Add intrinsic motivation or reward shaping
4. **Optional**: Fix tau scheduling if generation quality is poor
5. **Later**: Implement temporal tokenizer if coherence issues appear

---

## Verification Checklist

Before training, verify:

- [ ] Dataset has non-zero rewards (print statistics)
- [ ] Reward head doesn't collapse (monitor Phase 2)
- [ ] Advantages have meaningful variance (monitor Phase 3)
- [ ] D+ and D- sets are both populated
- [ ] Value predictions correlate with returns
- [ ] Generated latents have reasonable statistics
