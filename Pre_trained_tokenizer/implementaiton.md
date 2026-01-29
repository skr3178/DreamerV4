# Implementation Review Checklist

## Status Legend
- **DONE** - Implemented and verified
- **PARTIAL** - Partially implemented or needs verification
- **NOT IMPLEMENTED** - Missing, needs to be added
- **NOT NEEDED** - Intentionally skipped or not applicable

---

## Core Integration Checks

### 1. Tokenizer Integration
**Status: DONE**

- `CosmosTokenizerWrapper` exists at `dreamer/models/cosmos_tokenizer_wrapper.py`
- Handles upsampling (64→256), encoding, spatial pooling (1024→16 tokens)
- Output shape: `(B, T_lat, 16, 16)` where `T_lat = 1 + ceil((T-1)/8)`
- Frozen by default with `requires_grad=False`

### 2. Upsampling Correctness
**Status: DONE**

- Bicubic upsampling in `_preprocess()` method (line 148)
- Normalizes to [-1, 1] for Cosmos input
- Clamps output to prevent artifacts

### 3. Dynamics Model Input Shape
**Status: DONE**

- Dynamics accepts pooled latents `(B, T_lat, 16, 16)`
- Flattened to `(B, T_lat, 256)` for head input
- Config: `num_latent_tokens=16`, `latent_dim=16`

### 4. x-prediction (NOT v-prediction)
**Status: DONE**

- Verified in `dreamer/losses/shortcut_loss.py` (line 35-36):
  > "The model predicts clean latents directly (x-prediction) rather than velocities (v-prediction)"
- Debug utility exists: `dreamer/utils/debug_phase1.py:verify_x_prediction()`

### 5. Phase 3 Gradient Flow (ONLY policy/value heads get gradients)
**Status: DONE**

- `train_phase3.py:freeze_world_model()` (line 167-188):
  - Freezes tokenizer, dynamics, reward_head
  - Sets all to `.eval()` mode
- Imagination rollout: `torch.no_grad()` wraps `generate_step()` (line 190)
- PMPO loss only backprops through policy head log_probs

```python
# CRITICAL CHECKS (these are valid):
assert policy_head.weight.grad is not None   # Policy head SHOULD train
assert value_head.weight.grad is not None    # Value head SHOULD train
assert dynamics grads is None                # Dynamics FROZEN in PMPO
assert reward_head grads is None             # Reward head FROZEN in PMPO
```

---

## Pitfall Checklist

| # | Pitfall | Status | Notes |
|---|---------|--------|-------|
| 1 | Using v-prediction instead of x-prediction | **DONE** | x-prediction implemented in shortcut_loss.py |
| 2 | Forgetting to detach signed_adv in PMPO | **NOT NEEDED** | PMPO uses boolean masks (`adv > 0`), not `sign(adv) * loss`. Boolean ops don't backprop through mask values. |
| 3 | Hardcoding T_lat=4 (ignoring causal 5 steps) | **DONE** | Dynamic `T_lat = latents.shape[1]` used everywhere. Helper method `compute_temporal_latent_steps()` added. |
| 4 | Tokenizing 64×64 directly (no upsampling) | **DONE** | Upsampling in `_preprocess()` method, bicubic to 256×256 |
| 5 | Training dynamics during PMPO | **DONE** | `freeze_world_model()` called in Phase 3. `torch.no_grad()` wraps dynamics in rollout. |

---

## Grok Fixes Review

### 1. No CosmosTokenizerWrapper.py visible
**Status: DONE**

- File exists: `dreamer/models/cosmos_tokenizer_wrapper.py` (372 lines)
- Exported in `dreamer/models/__init__.py`
- Factory function: `create_cosmos_tokenizer()`

### 2. train_phase1_cosmos.py is incomplete / minimal
**Status: DONE**

- Full implementation exists (539 lines)
- Frozen Cosmos loading via `create_cosmos_tokenizer_from_config()`
- `torch.no_grad()` wraps tokenizer encode (line 122-124)
- Shortcut forcing loss only (no tokenizer loss)

### 3. Pretokenization script and dataset loader
**Status: DONE**

- `pretokenize_dataset.py` - Script to pre-encode videos with Cosmos
- `dreamer/data/pretokenized_dataset.py` - Dataset loader for pretokenized latents
- Trade-off: ~2-3× speedup vs storage cost (~100MB-10GB depending on dataset)

**Usage:**
```bash
# Pretokenize full dataset
python pretokenize_dataset.py --data-path data/mineRL_extracted --output-path data/pretokenized

# Pretokenize subset for testing
python pretokenize_dataset.py --data-path data/mineRL_extracted --output-path data/pretokenized_test --max-episodes 10
```

**Loading pretokenized data:**
```python
from dreamer.data import PretokenizedDataset, create_pretokenized_dataloader

dataloader = create_pretokenized_dataloader(
    data_path="data/pretokenized",
    batch_size=16,
    sequence_length=5,  # In latent steps (not frames!)
)
```

### 4. Dimension / shape handling risks
**Status: DONE**

- Dynamic `T_lat = latents.shape[1]` used consistently
- Causal formula: `T_lat = 1 + ceil((T - 1) / 8)` documented
- Helper method added: `CosmosTokenizerWrapper.compute_temporal_latent_steps()`
- Action alignment uses `torch.linspace()` for subsampling

### 5. Config & freezing
**Status: DONE**

- `configs/minerl_cosmos.yaml` has:
  ```yaml
  cosmos_tokenizer:
    enabled: true
    checkpoint_path: "cosmos_tokenizer/CV8x8x8"
  ```
- Freezing verified:
  - `cosmos_tokenizer_wrapper.py` line 108-109: `requires_grad = False`
  - `train_phase3.py` line 177-182: explicit freezing of tokenizer, dynamics, reward_head

---

## Verification Commands

```python
# Check trainable params (should only be dynamics in Phase 1, heads in Phase 2/3)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {trainable:,}, Frozen: {frozen:,}")

# Verify causal temporal compression
from dreamer.models import CosmosTokenizerWrapper
T_lat = CosmosTokenizerWrapper.compute_temporal_latent_steps(32)
assert T_lat == 5, f"Expected 5, got {T_lat}"

# Verify gradient flow in Phase 3 (after backward pass)
assert all(p.grad is None for p in tokenizer.parameters()), "Tokenizer should be frozen!"
assert all(p.grad is None for p in dynamics.parameters()), "Dynamics should be frozen!"
assert all(p.grad is None for p in reward_head.parameters()), "Reward head should be frozen!"
assert any(p.grad is not None for p in policy_head.parameters()), "Policy head should train!"
assert any(p.grad is not None for p in value_head.parameters()), "Value head should train!"
```

---

## Summary

| Category | Done | Partial | Not Implemented | Not Needed |
|----------|------|---------|-----------------|------------|
| Core Integration | 5 | 0 | 0 | 0 |
| Pitfalls | 4 | 0 | 0 | 1 |
| Grok Fixes | 5 | 0 | 0 | 0 |
| **Total** | **14** | **0** | **0** | **1** |

**All features implemented!**
