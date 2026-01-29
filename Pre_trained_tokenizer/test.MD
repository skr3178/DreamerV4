# DreamerV4 with Pretrained Cosmos Tokenizer

Plan saved to: ~/.claude/plans/velvet-tinkering-owl.md

---

## Process Overview

### Temporal Compression (Causal)

Cosmos CV8x8x8 is a **causal** video tokenizer. The first frame gets its own latent step (no future context), then subsequent frames are grouped by 8:

```
T_latent = 1 + ceil((T_frames - 1) / 8)

Example: 32 frames → 1 + ceil(31/8) = 1 + 4 = 5 latent steps
```

This prevents future information leakage — critical for real-time interaction.

---

```
PHASE 1: World Model Training
══════════════════════════════════════════════════════════════════════════════

  Video (B,3,32,64,64)
    │
    ▼ Upsample (bicubic)
  (B,3,32,256,256)
    │
    ▼ Cosmos Tokenizer (FROZEN)
  Raw Latents (B,16,5,32,32)          ← C=16, T_lat=5, H/8=32, W/8=32
    │
    ▼ Reshape
  (B,5,1024,16)                       ← 32×32 = 1024 spatial tokens
    │
    ▼ Spatial Pooling (avg_pool 8×8)
  Pooled Latents (B,5,16,16)          ← 4×4 = 16 tokens, dim=16
    │
    ▼ Dynamics Model (TRAINABLE)
    │
    ▼ Shortcut Forcing Loss (predict future latents)

══════════════════════════════════════════════════════════════════════════════


PHASE 2: Agent Finetuning (Behavior Cloning)
══════════════════════════════════════════════════════════════════════════════

  Video → Tokenizer (FROZEN) → Latents (B,5,16,16)
    │
    ▼ Flatten
  (B,5,256)                           ← 16 tokens × 16 dim = 256
    │
    ├──► Policy Head (TRAINABLE)      → BC loss (predict actions)
    │
    └──► Reward Head (TRAINABLE)      → Symlog cross-entropy (predict rewards)

  Dynamics: FROZEN
  Value Head: initialized (not primary focus)

  [Optional for multi-task: inject TASK_TOKEN at sequence start]

══════════════════════════════════════════════════════════════════════════════


PHASE 3: PMPO Imagination RL
══════════════════════════════════════════════════════════════════════════════

  Real Frame → Tokenizer (FROZEN) → Initial Latent z₀
    │
    ▼ Imagination Rollout (H steps)
    ┌─────────────────────────────────────────────────────────┐
    │  for t in range(H):                                     │
    │      zₜ  ──► Policy Head (TRAINABLE) ──► aₜ            │
    │      (zₜ, aₜ) ──► Dynamics (FROZEN) ──► zₜ₊₁           │
    │      zₜ  ──► Reward Head (FROZEN) ──► r̂ₜ               │
    └─────────────────────────────────────────────────────────┘
    │
    ▼ Compute Advantages
  Âₜ = sign(GAE returns)              ← Binarized for PMPO
    │
    ├──► Policy: PMPO loss            -(Âₜ · log π(aₜ|zₜ))
    │
    └──► Value Head: TD(λ) loss       (TRAINABLE)

  ⚠️  IMPORTANT: No gradients flow through world model!
      Only policy & value heads receive gradients.
      Reward head predictions come from Phase 2 (frozen).

══════════════════════════════════════════════════════════════════════════════
```

### Summary Table

| Phase | Trainable | Frozen | Objective |
|-------|-----------|--------|-----------|
| **1** | Dynamics | Cosmos Tokenizer | Predict future latents (world model) |
| **2** | Policy, Reward, Value Heads | Tokenizer, Dynamics | Behavior cloning + reward prediction |
| **3** | Policy, Value Heads | Tokenizer, Dynamics, Reward | Maximize imagined returns (PMPO) |

### Data Flow in One Sentence

**Real video → frozen tokenizer → latents → dynamics predicts futures → heads output actions/values/rewards → policy optimized in imagination without real environment.**

---

## Preprocessing: Upsampling for Cosmos

Cosmos CV8x8x8 requires 256x256 input. MineRL frames are 64x64, so we upsample.

```python
import torch.nn.functional as F

# Single frame preprocessing
def preprocess_frame(frame_64):  # frame_64: [3, 64, 64]
    # Bicubic preserves block edges better than bilinear
    frame_256 = F.interpolate(
        frame_64.unsqueeze(0),
        size=(256, 256),
        mode='bicubic',
        align_corners=False
    ).clamp(0, 1)  # Prevent overshoot artifacts
    return frame_256.squeeze(0)  # [3, 256, 256]


# Batch preprocessing (for dataloader)
def preprocess_batch(frames_64):  # frames_64: [B, T, C, 64, 64] or [B, C, T, 64, 64]
    B, C, T, H, W = frames_64.shape
    # Reshape for interpolation
    frames_flat = frames_64.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    frames_256 = F.interpolate(
        frames_flat,
        size=(256, 256),
        mode='bicubic',
        align_corners=False
    ).clamp(0, 1)
    # Reshape back
    return frames_256.reshape(B, T, C, 256, 256).permute(0, 2, 1, 3, 4)  # [B, C, T, 256, 256]
```

**Note**: The `CosmosTokenizerWrapper` already handles upsampling internally in `_preprocess()`, so you don't need to do this manually unless you want to pre-upsample in the dataloader.

---

## Debugging Guide for Each Phase

### Phase 1: World Model Training

| Check | How | Red Flags |
|-------|-----|-----------|
| **Tokenizer output** | Log `latents.shape`, `latents.mean()`, `latents.std()` | Shape not `(B,5,16,16)`, std=0 or NaN |
| **Temporal dim** | Verify `T_lat = 1 + ceil((T-1)/8)` | Wrong T_lat = non-causal encoder or bug |
| **Reconstruction** | Decode latents → save images periodically | Blurry/wrong colors = pooling too aggressive |
| **Dynamics loss curve** | TensorBoard `loss/dynamics/total` | Flat line = not learning; spikes = LR too high |
| **Gradient norms** | Log `torch.nn.utils.clip_grad_norm_()` return value | >100 = exploding; <1e-6 = vanishing |
| **Action alignment** | Verify `actions.shape[1]` matches `latents.shape[1]` | Mismatch = temporal subsampling bug |
| **VRAM usage** | `nvidia-smi` or log `torch.cuda.memory_allocated()` | >11GB on 12GB card = OOM risk |

```python
# Quick sanity check for Phase 1
with torch.no_grad():
    out = cosmos_tokenizer.encode(frames)
    T_frames = frames.shape[2]  # or frames.shape[1] depending on format
    T_lat_expected = 1 + math.ceil((T_frames - 1) / 8)
    print(f"Latent shape: {out['latents'].shape}")  # Expect (B, T_lat_expected, 16, 16)
    print(f"Expected T_lat: {T_lat_expected}, Got: {out['latents'].shape[1]}")
    print(f"Latent stats: mean={out['latents'].mean():.3f}, std={out['latents'].std():.3f}")
```

---

### Phase 2: Agent Finetuning

| Check | How | Red Flags |
|-------|-----|-----------|
| **BC accuracy** | Log `bc_accuracy` metric | <5% after 1k steps = head not learning |
| **Action distribution** | Log `policy_head(latents)["probs"].max()` | Always ~1.0 = collapsed to deterministic |
| **Reward prediction std** | Log `pred_std` in loss | <0.01 = reward head collapsed to constant |
| **Class imbalance** | Check action histogram from dataset | 90%+ one action = need focal loss |
| **Latent frozen** | Verify `latents.requires_grad == False` | True = tokenizer not frozen properly |
| **Checkpoint loading** | Print loaded keys vs model keys | Missing keys = architecture mismatch |

```python
# Check for collapsed policy
with torch.no_grad():
    policy_out = policy_head(latents_flat)
    entropy = -(policy_out["probs"] * policy_out["probs"].log()).sum(-1).mean()
    print(f"Policy entropy: {entropy:.3f}")  # Should be >1.0, not ~0
```

---

### Phase 3: Imagination RL

| Check | How | Red Flags |
|-------|-----|-----------|
| **Advantage distribution** | Log `adv_mean`, `adv_std`, `n_positive`, `n_negative` | All zeros = no learning signal |
| **Imagined rewards** | Log `rollout_data["rewards"].mean()` | Always 0 = reward head broken |
| **Value predictions** | Compare `value_head` output to actual returns | Large gap = value not calibrated |
| **Policy entropy** | Log entropy from PMPO loss | <0.1 = premature convergence |
| **D+/D- samples** | Log `n_positive`, `n_negative` from PMPO | Both <10 = degenerate binning |
| **Imagination drift** | Visualize decoded latents at horizon H | Garbage = dynamics diverging |
| **Gradient flow** | Verify no grads on tokenizer/dynamics/reward | Grads present = freezing failed |

```python
# Check imagination rollout sanity
rollout = imagination(initial_latents, lambda_=0.95)
print(f"Rewards: mean={rollout['rewards'].mean():.4f}, std={rollout['rewards'].std():.4f}")
print(f"Advantages: mean={rollout['advantages'].mean():.4f}, std={rollout['advantages'].std():.4f}")
print(f"Non-zero rewards: {(rollout['rewards'] != 0).float().mean():.2%}")

# Verify no gradients through world model
print(f"Tokenizer grads: {any(p.grad is not None for p in tokenizer.parameters())}")  # Should be False
print(f"Dynamics grads: {any(p.grad is not None for p in dynamics.parameters())}")    # Should be False
print(f"Reward head grads: {any(p.grad is not None for p in reward_head.parameters())}")  # Should be False
```

---

### Common Issues Across All Phases

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| NaN loss | LR too high, missing normalization | Reduce LR, add LayerNorm, check inputs |
| Loss doesn't decrease | Frozen layers not frozen, wrong optimizer | Verify `requires_grad`, check param groups |
| OOM | Batch too large, no gradient accumulation | Reduce batch, enable accumulation |
| Slow training | DataLoader bottleneck | Increase `num_workers`, enable `pin_memory` |
| Checkpoint won't load | `torch.compile()` prefix mismatch | Use `strip_compiled_prefix()` |
| Wrong T_lat dimension | Non-causal assumption | Use `1 + ceil((T-1)/8)` formula |

---

### Quick Diagnostic Script

```python
import math

def diagnose_phase(phase, model_dict, sample_batch, device):
    """Run diagnostics for any phase."""
    print(f"=== Phase {phase} Diagnostics ===")

    # 1. Check all models on correct device
    for name, model in model_dict.items():
        param = next(model.parameters(), None)
        if param is not None:
            print(f"{name}: device={param.device}, requires_grad={param.requires_grad}")

    # 2. Forward pass check
    try:
        with torch.no_grad():
            frames = sample_batch["frames"].to(device)
            latents = model_dict["tokenizer"].encode(frames)["latents"]

            # Check temporal dimension (causal formula)
            T_frames = frames.shape[2] if frames.shape[1] == 3 else frames.shape[1]
            T_lat_expected = 1 + math.ceil((T_frames - 1) / 8)
            T_lat_actual = latents.shape[1]

            print(f"Tokenizer output: {latents.shape}, finite={latents.isfinite().all()}")
            print(f"Temporal: {T_frames} frames → {T_lat_actual} latents (expected {T_lat_expected})")
    except Exception as e:
        print(f"Forward pass failed: {e}")

    # 3. Memory check
    if device.type == "cuda":
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory used: {mem:.2f} GB")
```

---

## Commands to Run

```bash
# Phase 1: World Model Training
python train_phase1_cosmos.py --config configs/minerl_cosmos.yaml

# Phase 2: Agent Finetuning
python train_phase2.py --config configs/minerl_cosmos.yaml --checkpoint checkpoints/dreamerv4_minerl_cosmos/phase1_cosmos_final.pt

# Phase 3: Imagination RL
python train_phase3.py --config configs/minerl_cosmos.yaml --phase2-checkpoint checkpoints/dreamerv4_minerl_cosmos/phase2_final.pt
```
